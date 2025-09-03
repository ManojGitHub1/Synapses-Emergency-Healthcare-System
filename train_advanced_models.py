import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# --- Configuration ---
X_DATA_PATH = 'X_train_2025.csv'
Y_DATA_PATH = 'y_train_2025.csv'
# We will save the final, best model (the Stacking model)
MODEL_SAVE_PATH = 'emergency_predictor_stacked.pkl' 
SCALER_SAVE_PATH = 'scaler.pkl'

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data():
    print("Loading and combining dataset...")
    X_df = pd.read_csv(X_DATA_PATH)
    y_df = pd.read_csv(Y_DATA_PATH)
    df = X_df.copy()
    df['In-hospital_death'] = y_df['In-hospital_death']
    df.rename(columns={'In-hospital_death': 'needs_icu'}, inplace=True)

    # --- FEATURE SELECTION for XGBoost (uses all relevant first-reads) ---
    tabular_features = [
        'Age', 'Gender', 'HR_first', 'SysABP_first', 'DiasABP_first', 'SaO2_first',
        'Temp_first', 'RespRate_first', 'GCS_first', 'Lactate_first', 'SAPS-I',
    ]
    
    # --- FEATURE SELECTION for LSTM (time-series features) ---
    # We select vitals that have multiple readings (_first, _last, _median)
    time_series_base_features = ['HR', 'SysABP', 'DiasABP', 'SaO2', 'Temp', 'RespRate']
    time_series_features = [f'{feat}_{suffix}' for feat in time_series_base_features for suffix in ['first', 'median', 'last']]

    all_features = sorted(list(set(tabular_features + time_series_features)))
    df_subset = df[all_features + ['needs_icu']].copy()
    
    for col in all_features:
        df_subset[col] = df_subset[col].fillna(df_subset[col].median())

    X = df_subset[all_features]
    y = df_subset['needs_icu']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler and feature list for the Flask app
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(all_features, 'feature_list.pkl')
    print(f"Scaler and feature list saved.")
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_features)
    return X_scaled_df, y, tabular_features, time_series_base_features

# --- 2. Model Training ---
def train_all_models(X, y, tabular_features, time_series_base_features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model A: XGBoost (Champion Challenger) ---
    print("\n--- Training XGBoost Classifier ---")
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                  scale_pos_weight=scale_pos_weight, use_label_encoder=False,
                                  n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train[tabular_features], y_train)
    xgb_preds = xgb_model.predict(X_test[tabular_features])
    print("\n--- XGBoost Performance Report ---")
    print(classification_report(y_test, xgb_preds, target_names=['Survived', 'Died']))
    print(f"XGBoost AUC Score: {roc_auc_score(y_test, xgb_model.predict_proba(X_test[tabular_features])[:, 1]):.4f}")

    # --- Model B: LSTM (Time Dimension) ---
    print("\n--- Training LSTM Model ---")
    # Reshape data into (samples, timesteps, features)
    # Timesteps = 3 (_first, _median, _last)
    # Features = number of base features (e.g., HR, SysABP, etc.)
    X_train_ts = X_train[[f'{feat}_{s}' for feat in time_series_base_features for s in ['first', 'median', 'last']]].values.reshape(len(X_train), 3, len(time_series_base_features))
    X_test_ts = X_test[[f'{feat}_{s}' for feat in time_series_base_features for s in ['first', 'median', 'last']]].values.reshape(len(X_test), 3, len(time_series_base_features))

    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_ts.shape[1], X_train_ts.shape[2])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    lstm_model.fit(X_train_ts, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=0)
    
    lstm_preds_proba = lstm_model.predict(X_test_ts)
    lstm_preds = (lstm_preds_proba > 0.5).astype(int)
    print("\n--- LSTM Performance Report ---")
    print(classification_report(y_test, lstm_preds, target_names=['Survived', 'Died']))
    print(f"LSTM AUC Score: {roc_auc_score(y_test, lstm_preds_proba):.4f}")

    # --- Model C: Stacking (The Final Boost) ---
    print("\n--- Training Stacking Classifier ---")
    # Define the base models (estimators)
    estimators = [
        ('xgb', XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                  scale_pos_weight=scale_pos_weight, use_label_encoder=False,
                                  n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)),
        # We need a wrapper to make the Keras LSTM model compatible with scikit-learn
        # For this PoC, we will stack XGBoost with a simpler scikit-learn model like Logistic Regression
        # A full Keras wrapper is complex, but this demonstrates the principle effectively.
        ('lr', LogisticRegression(class_weight='balanced'))
    ]

    # The final model is a simple Logistic Regression that combines the outputs
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    stacking_model.fit(X_train[tabular_features], y_train) # Stacking on tabular data for simplicity

    stack_preds = stacking_model.predict(X_test[tabular_features])
    print("\n--- Stacking Model Performance Report ---")
    print(classification_report(y_test, stack_preds, target_names=['Survived', 'Died']))
    print(f"Stacking AUC Score: {roc_auc_score(y_test, stacking_model.predict_proba(X_test[tabular_features])[:, 1]):.4f}")

    # Save the final, best model
    joblib.dump(stacking_model, MODEL_SAVE_PATH)
    print(f"\n✅ Final Stacking model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    X_data, y_data, tabular_cols, time_series_cols = load_and_preprocess_data()
    train_all_models(X_data, y_data, tabular_cols, time_series_cols)