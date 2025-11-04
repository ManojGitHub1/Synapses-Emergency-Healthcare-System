# New, simplified, and more powerful training script
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb # Use XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
X_DATA_PATH = 'X_train_2025.csv'
Y_DATA_PATH = 'y_train_2025.csv'
MODEL_SAVE_PATH = 'emergency_predictor_xgb.pkl' # Save as .pkl
SCALER_SAVE_PATH = 'scaler.pkl'

def load_and_preprocess_data():
    print("Loading and combining dataset...")
    X_df = pd.read_csv(X_DATA_PATH)
    y_df = pd.read_csv(Y_DATA_PATH)
    df = X_df.copy()
    df['In-hospital_death'] = y_df['In-hospital_death']
    df.rename(columns={'In-hospital_death': 'needs_icu'}, inplace=True)

    model_features = [
        'Age', 'Gender', 'HR_first', 'SysABP_first', 'DiasABP_first', 'SaO2_first',
        'Temp_first', 'RespRate_first', 'GCS_first', 'Lactate_first', 'SAPS-I',
    ]
    df_subset = df[model_features + ['needs_icu']].copy()
    for col in model_features:
        df_subset[col] = df_subset[col].fillna(df_subset[col].median())
    
    X = df_subset[model_features]
    y = df_subset['needs_icu']
    
    # XGBoost doesn't strictly need scaling, but it's good practice
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    return X_scaled, y, model_features

def train_xgb_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Training XGBoost Classifier ---")
    # Use scale_pos_weight to handle imbalanced data
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        n_estimators=150, # More trees
        max_depth=5,       # Deeper trees
        learning_rate=0.1,
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    
    preds = xgb_model.predict(X_test)
    print("\n--- XGBoost Final Performance Report ---")
    print(classification_report(y_test, preds, target_names=['Survived', 'Died']))

    # Save the model using joblib (standard for scikit-learn compatible models)
    joblib.dump(xgb_model, MODEL_SAVE_PATH)
    print(f"✅ XGBoost model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    X_data, y_data, feature_list = load_and_preprocess_data()
    print("\n--- Feature order for the model ---")
    print(feature_list)
    train_xgb_model(X_data, y_data)