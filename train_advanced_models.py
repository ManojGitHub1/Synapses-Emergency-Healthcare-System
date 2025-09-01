import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
X_DATA_PATH = 'X_train_2025.csv'
Y_DATA_PATH = 'y_train_2025.csv'
MODEL_SAVE_PATH = 'emergency_predictor_ann.h5'
SCALER_SAVE_PATH = 'scaler.pkl'

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data():
    print("Loading and combining dataset...")
    X_df = pd.read_csv(X_DATA_PATH)
    y_df = pd.read_csv(Y_DATA_PATH)

    # --- COMBINING THE DATA ---
    # Since y_train has no ID, we assume the rows are perfectly aligned.
    # We will take the target column from y_df and add it to X_df.
    df = X_df.copy() # Make a copy to avoid modifying the original dataframe
    df['In-hospital_death'] = y_df['In-hospital_death']

    # Now 'df' is the complete, combined dataframe.
    # The rest of the logic remains the same.
    
    df.rename(columns={'In-hospital_death': 'needs_icu'}, inplace=True)
    print(f"Dataset combined successfully with {df.shape[0]} records.")

    # --- FEATURE SELECTION ---
    model_features = [
        'Age', 'Gender',
        'HR_first',             # Map to heart_rate
        'SysABP_first',         # Map to systolic_blood_pressure
        'DiasABP_first',        # Map to diastolic_blood_pressure
        'SaO2_first',           # Map to oxygen_saturation
        'Temp_first',           # Map to temperature
        'RespRate_first',       # Map to respiratory_rate
        'GCS_first',            # Glasgow Coma Scale - a key indicator
        'Lactate_first',        # Important lab value
        'SAPS-I',               # Severity score
    ]
    target = 'needs_icu'
    
    # Check if all required features exist in the dataframe
    for feature in model_features:
        if feature not in df.columns:
            raise ValueError(f"CRITICAL ERROR: Feature '{feature}' not found in the dataset columns.")

    df_subset = df[model_features + [target]].copy()

    # Simple imputation: fill missing values with the median
    for col in model_features:
        df_subset[col].fillna(df_subset[col].median(), inplace=True)

    X = df_subset[model_features]
    y = df_subset[target]

    # Scale the data for the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")
    
    return X_scaled, y, model_features

# --- 2. Model Training ---
def train_ann_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Training Artificial Neural Network (ANN) ---")
    ann_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    neg, pos = np.bincount(y_train)
    class_weight = {0: (1 / neg) * (len(y_train) / 2.0), 1: (1 / pos) * (len(y_train) / 2.0)}
    
    ann_model.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.2, 
                  class_weight=class_weight, verbose=1)
    
    ann_preds_proba = ann_model.predict(X_test)
    ann_preds = (ann_preds_proba > 0.5).astype(int)
    print("\n--- ANN Final Performance Report ---")
    print(classification_report(y_test, ann_preds, target_names=['Survived', 'Died']))

    # Use the explicit saving function for Keras 3
    ann_model.save(MODEL_SAVE_PATH)
    print(f"✅ ANN model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    X_data, y_data, feature_list = load_and_preprocess_data()
    print("\n--- Feature order for the model ---")
    print(feature_list)
    train_ann_model(X_data, y_data)