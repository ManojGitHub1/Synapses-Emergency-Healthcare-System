# Final app.py to work with the Stacking model
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, Patient
from transformers import pipeline

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key'
db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

try:
    os.makedirs(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance'))
except OSError:
    pass
db.init_app(app)

# --- Load AI Models and Scaler ---
MODEL_PATH = 'emergency_predictor_stacked.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURE_LIST_PATH = 'feature_list.pkl' # Load the saved feature list

predictor = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_features = joblib.load(FEATURE_LIST_PATH)
summarizer = pipeline("summarization", model="t5-small")
print("✅ Stacking Model, Scaler, and Summarizer loaded successfully.")

# Default values for features not on the web form
DEFAULT_VALUES = {
    'GCS_first': 14.0, 'Lactate_first': 2.0, 'SAPS-I': 38.0,
    # Add defaults for _median and _last values for LSTM features
    'HR_median': 85.0, 'HR_last': 85.0,
    'SysABP_median': 110.0, 'SysABP_last': 110.0,
    'DiasABP_median': 60.0, 'DiasABP_last': 60.0,
    'SaO2_median': 98.0, 'SaO2_last': 98.0,
    'Temp_median': 37.0, 'Temp_last': 37.0,
    'RespRate_median': 18.0, 'RespRate_last': 18.0
}

with app.app_context():
    db.create_all()

def make_prediction(form_data):
    patient_features = {
        'Age': float(form_data['age']),
        'Gender': 1 if form_data['gender'].lower() == 'male' else 0,
        'HR_first': float(form_data['heart_rate']),
        'SysABP_first': float(form_data['systolic_blood_pressure']),
        'DiasABP_first': float(form_data['diastolic_blood_pressure']),
        'SaO2_first': float(form_data['oxygen_saturation']),
        'Temp_first': float(form_data['temperature']),
        'RespRate_first': float(form_data['respiratory_rate']),
        **DEFAULT_VALUES
    }
    df = pd.DataFrame([patient_features], columns=model_features) # Use the loaded feature list for order
    scaled_features = scaler.transform(df)
    
    # Stacking model uses the same prediction methods as other sklearn models
    prediction_result = predictor.predict(scaled_features)[0]
    risk_proba = predictor.predict_proba(scaled_features)[0][1]

    prompt = f"Summarize this patient case for an ER doctor: ..." # Summarizer logic remains the same
    summary_result = summarizer(prompt, max_length=60, min_length=20, do_sample=False)
    
    return {
        'risk_score': float(risk_proba),
        'predicted_icu_need': bool(prediction_result == 1),
        'summary': summary_result[0]['summary_text']
    }

# --- Routes (No changes needed below this line) ---
@app.route('/report', methods=['GET', 'POST'])
def report():
    if request.method == 'POST':
        try:
            prediction = make_prediction(request.form)
            # ... (rest of the route is the same)
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