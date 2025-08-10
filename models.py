from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.timestamp)
    
    # Basic Info
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    
    # Vital Signs
    heart_rate = db.Column(db.Integer, nullable=False)
    blood_pressure_systolic = db.Column(db.Integer, nullable=False)
    blood_pressure_diastolic = db.Column(db.Integer, nullable=False)
    oxygen_saturation = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    respiratory_rate = db.Column(db.Integer, nullable=False)
    
    # Additional Info
    paramedic_notes = db.Column(db.Text)
    location = db.Column(db.String(200))
    
    # ML Predictions
    predicted_icu_need = db.Column(db.Boolean, default=False)
    risk_score = db.Column(db.Float)
    prediction_confidence = db.Column(db.Float)
    
    def get_features_for_ml(self):
        """Return features in the format expected by the ML model."""
        # Note the order must match the training script's feature_names
        return [
            self.age,
            1 if self.gender.lower() == 'male' else 0,
            self.heart_rate,
            self.blood_pressure_systolic,
            self.blood_pressure_diastolic,
            self.oxygen_saturation,
            self.temperature,
            self.respiratory_rate
        ]