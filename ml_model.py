import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class EmergencyPredictor:
    def __init__(self, model_path='emergency_predictor.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'age', 'gender_male', 'heart_rate', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'oxygen_saturation', 'temperature', 'respiratory_rate'
        ]
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
    
    def save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            return True
        return False
    
    def train_model(self, data_path='sample_data.csv'):
        if not os.path.exists(data_path):
            print("Sample data not found, generating new data...")
            df = self.generate_sample_data()
            df.to_csv(data_path, index=False)
        else:
            df = pd.read_csv(data_path)
        
        X = df[self.feature_names]
        y = df['needs_icu']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        
        self.save_model()
        return accuracy
    
    def predict(self, features):
        if not self.model:
            if not self.load_model():
                raise Exception("Model not found. Please train the model first.")
        
        features_2d = np.array(features).reshape(1, -1)
        
        prediction = self.model.predict(features_2d)[0]
        probability = self.model.predict_proba(features_2d)[0]
        
        return {
            'needs_icu': bool(prediction),
            'risk_score': float(probability[1]),
            'confidence': float(max(probability))
        }
    
    def generate_sample_data(self, n_samples=500):
        np.random.seed(42)
        data = []
        for _ in range(n_samples):
            age = np.random.randint(18, 90)
            gender_male = np.random.choice([0, 1])
            heart_rate = np.clip(np.random.normal(75, 25), 40, 200)
            bp_systolic = np.clip(np.random.normal(120, 30), 80, 200)
            bp_diastolic = np.clip(np.random.normal(80, 15), 50, 120)
            oxygen_sat = np.clip(np.random.normal(97, 3), 85, 100)
            temperature = np.clip(np.random.normal(98.6, 1.5), 95, 105)
            resp_rate = np.clip(np.random.normal(16, 6), 8, 30)
            
            risk_score = 0
            if age > 70: risk_score += 0.3
            if heart_rate > 120 or heart_rate < 50: risk_score += 0.4
            if bp_systolic < 90 or bp_systolic > 160: risk_score += 0.3
            if oxygen_sat < 94: risk_score += 0.5
            if resp_rate > 24 or resp_rate < 12: risk_score += 0.3
            
            needs_icu = 1 if risk_score > 0.5 else 0
            
            data.append([age, gender_male, heart_rate, bp_systolic, bp_diastolic,
                         oxygen_sat, temperature, resp_rate, needs_icu])
        
        columns = self.feature_names + ['needs_icu']
        return pd.DataFrame(data, columns=columns).round(2)