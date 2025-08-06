# 🏥 Emergency Response Prediction System

A full-stack web application that enables paramedics to submit patient data from the field, predicts whether the patient needs ICU attention using a machine learning model, and presents the information on a centralized hospital dashboard.

---

## 📌 Project Goals

- Build an end-to-end emergency response platform.
- Use machine learning to predict ICU needs based on patient vitals.
- Create an intuitive UI for paramedic data entry and hospital staff monitoring.
- Deploy a responsive, production-ready web app.

---

## 🧰 Technology Stack

| Layer       | Tool / Library             |
|-------------|----------------------------|
| Backend     | Flask (Python)             |
| Frontend    | HTML, CSS, JavaScript      |
| Database    | SQLite (dev), PostgreSQL (prod) |
| ML Model    | Scikit-learn, XGBoost      |
| Deployment  | Heroku / PythonAnywhere / AWS / GCP |
| Others      | Flask-Mail, Joblib, Pandas |

---

## 📅 Project Timeline

### ✅ Phase 1: Setup and Foundation (Week 1)

- **Set up workspace**:
  - Install Python
  - Create virtual environment: `python -m venv venv`
  - Install libraries:  
    ```
    pip install Flask Flask-SQLAlchemy scikit-learn pandas xgboost
    ```

- **Build Flask Skeleton**:
  - Create `app.py` with a basic "Hello, World!" app
  - Define the `Patient` database model with fields:
    - `id`, `age`, `gender`, `heart_rate`, `blood_pressure`, `oxygen_saturation`, `paramedic_notes`, `timestamp`

- ✅ **Outcome**: Running Flask app with basic DB model.

---

### 🚑 Phase 2: Core Web Application (Weeks 2–4)

- **Paramedic Data Entry**:
  - HTML form at `/report` with patient input fields.
  - Flask route: display + receive + save to DB.

- **Hospital Dashboard**:
  - Route `/dashboard` to fetch all patient records.
  - HTML dashboard displays records in table/cards.
  - Add `View Details` button → `/patient/<int:id>`.

- ✅ **Outcome**: End-to-end data flow from paramedic form → database → dashboard.

---

### 🧠 Phase 3: Machine Learning Model (Weeks 5–6)

- **Create Sample Dataset**:
  - CSV with 100+ rows, fields matching DB, `needs_icu` column as target.

- **Train Model (train_model.py)**:
  - Use `pandas` to load data
  - Train/test split
  - Train using XGBoost
  - Save model:
    ```python
    import joblib
    joblib.dump(model, 'emergency_predictor.pkl')
    ```

- ✅ **Outcome**: Trained model saved as `emergency_predictor.pkl`.

---

### 🔗 Phase 4: Integration (Week 7)

- **Load Model in Flask**:
  ```python
  model = joblib.load('emergency_predictor.pkl')
