import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, Patient
from transformers import pipeline

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_that_should_be_changed'
db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

try:
    os.makedirs(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance'))
except OSError:
    pass

db.init_app(app)

# --- Load AI Models and Scaler ---
MODEL_PATH = 'emergency_predictor_ann.h5'
SCALER_PATH = 'scaler.pkl'

# Use the explicit loading function for Keras 3
predictor = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
summarizer = pipeline("summarization", model="t5-small")
print("✅ All AI models and scaler loaded successfully.")

MODEL_FEATURES = [
    'Age', 'Gender', 'HR_first', 'SysABP_first', 'DiasABP_first', 'SaO2_first',
    'Temp_first', 'RespRate_first', 'GCS_first', 'Lactate_first', 'SAPS-I'
]

DEFAULT_VALUES = { 'GCS_first': 14.0, 'Lactate_first': 2.0, 'SAPS-I': 38.0 }

with app.app_context():
    db.create_all()

def make_prediction(form_data):
    patient_features = {
        'Age': float(form_data['age']), 'Gender': 1 if form_data['gender'].lower() == 'male' else 0,
        'HR_first': float(form_data['heart_rate']), 'SysABP_first': float(form_data['systolic_blood_pressure']),
        'DiasABP_first': float(form_data['diastolic_blood_pressure']), 'SaO2_first': float(form_data['oxygen_saturation']),
        'Temp_first': float(form_data['temperature']), 'RespRate_first': float(form_data['respiratory_rate']),
        **DEFAULT_VALUES
    }
    df = pd.DataFrame([patient_features], columns=MODEL_FEATURES)
    scaled_features = scaler.transform(df)
    risk_proba = predictor.predict(scaled_features)[0][0]
    prompt = f"""Summarize this patient case for an ER doctor based on initial field data: 
    A {form_data['age']}-year-old {form_data['gender']} presents with critical vitals.
    - Heart Rate: {form_data['heart_rate']} bpm
    - Blood Pressure: {form_data['systolic_blood_pressure']}/{form_data['diastolic_blood_pressure']} mmHg
    - O2 Saturation: {form_data['oxygen_saturation']}%
    - Notes: {form_data.get('paramedic_notes', 'N/A')}
    """
    summary_result = summarizer(prompt, max_length=60, min_length=20, do_sample=False)
    return { 'risk_score': float(risk_proba), 'predicted_icu_need': bool(risk_proba > 0.5), 'summary': summary_result[0]['summary_text'] }

@app.route('/report', methods=['GET', 'POST'])
def report():
    if request.method == 'POST':
        try:
            prediction = make_prediction(request.form)
            new_patient = Patient(
                age=int(request.form['age']), gender=request.form['gender'],
                heart_rate=int(request.form['heart_rate']),
                blood_pressure_systolic=int(request.form['systolic_blood_pressure']),
                blood_pressure_diastolic=int(request.form['diastolic_blood_pressure']),
                oxygen_saturation=float(request.form['oxygen_saturation']),
                temperature=float(request.form['temperature']),
                respiratory_rate=int(request.form['respiratory_rate']),
                paramedic_notes=request.form.get('paramedic_notes', ''),
                location=request.form.get('location', ''),
                predicted_icu_need=prediction['predicted_icu_need'],
                risk_score=prediction['risk_score'],
                generative_summary=prediction['summary']
            )
            db.session.add(new_patient)
            db.session.commit()
            flash(f"Patient report #{new_patient.id} analyzed successfully.", 'success')
            return redirect(url_for('patient_detail', patient_id=new_patient.id))
        except Exception as e:
            db.session.rollback()
            flash(f"Error submitting report: {e}", 'error')
            print(f"FATAL ERROR in /report: {e}")
            return redirect(url_for('report'))
    return render_template('report.html')

@app.route('/')
def index(): return render_template('index.html')
    
@app.route('/dashboard')
def dashboard():
    patients = Patient.query.order_by(Patient.timestamp.desc()).all()
    return render_template('dashboard.html', patients=patients)

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    patient = db.get_or_404(Patient, patient_id)
    return render_template('patient.html', patient=patient)
    
if __name__ == '__main__':
    app.run(debug=True)