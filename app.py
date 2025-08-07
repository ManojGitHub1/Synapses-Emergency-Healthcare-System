import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, Patient
from ml_model import EmergencyPredictor

# --- App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_that_should_be_changed'

# Configure database
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'instance', 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure instance folder exists for the database
try:
    os.makedirs(os.path.join(basedir, 'instance'))
except OSError:
    pass

# Initialize database and ML model
db.init_app(app)
predictor = EmergencyPredictor(model_path='emergency_predictor.pkl')

# --- Model Loading and Database Creation ---
with app.app_context():
    db.create_all()
    if predictor.load_model():
        print("✅ Machine Learning model loaded successfully.")
    else:
        print("⚠️ WARNING: ML model not found. Run 'python train_model.py' to generate it.")

# --- Routes ---
@app.route('/')
def index():
    patient_count = Patient.query.count()
    high_risk_count = Patient.query.filter_by(predicted_icu_need=True).count()
    return render_template('index.html', patient_count=patient_count, high_risk_count=high_risk_count)

@app.route('/report', methods=['GET', 'POST'])
def report():
    if request.method == 'POST':
        try:
            new_patient = Patient(
                age=int(request.form['age']),
                gender=request.form['gender'],
                heart_rate=int(request.form['heart_rate']),
                blood_pressure_systolic=int(request.form['blood_pressure_systolic']),
                blood_pressure_diastolic=int(request.form['blood_pressure_diastolic']),
                oxygen_saturation=float(request.form['oxygen_saturation']),
                temperature=float(request.form['temperature']),
                respiratory_rate=int(request.form['respiratory_rate']),
                paramedic_notes=request.form.get('paramedic_notes', ''),
                location=request.form.get('location', '')
            )
            
            features = new_patient.get_features_for_ml()
            prediction_result = predictor.predict(features)
            
            new_patient.predicted_icu_need = prediction_result['needs_icu']
            new_patient.risk_score = prediction_result['risk_score']
            new_patient.prediction_confidence = prediction_result['confidence']
            
            db.session.add(new_patient)
            db.session.commit()
            
            flash(f"Patient #{new_patient.id} submitted. Risk Score: {new_patient.risk_score:.2f}", 'success')
            return redirect(url_for('patient_detail', patient_id=new_patient.id))

        except Exception as e:
            db.session.rollback()
            flash(f"Error submitting report: {e}", 'error')
            return redirect(url_for('report'))

    return render_template('report.html')

@app.route('/dashboard')
def dashboard():
    patients = Patient.query.order_by(Patient.timestamp.desc()).all()
    total_patients = len(patients)
    high_risk_count = sum(1 for p in patients if p.predicted_icu_need)
    time_24_hours_ago = datetime.utcnow() - timedelta(hours=24)
    recent_count = Patient.query.filter(Patient.timestamp >= time_24_hours_ago).count()
        
    return render_template(
        'dashboard.html', 
        patients=patients,
        total_patients=total_patients,
        high_risk_count=high_risk_count,
        recent_count=recent_count
    )

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    patient = db.get_or_404(Patient, patient_id)
    return render_template('patient.html', patient=patient)

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists('emergency_predictor.pkl'):
        print("="*50)
        print("ERROR: Model file 'emergency_predictor.pkl' not found.")
        print("Please run the training script first by executing:")
        print("python train_model.py")
        print("="*50)
    else:
        app.run(debug=True)