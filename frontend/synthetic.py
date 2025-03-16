from flask import Flask, render_template, jsonify
import random
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained models
with open('diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('stroke_model.pkl', 'rb') as f:
    stroke_model = pickle.load(f)

# Pre-defined patient data
patients = [
    {"id": 1, "name": "Patient A", "age": 45, "gender": "Male"},
    {"id": 2, "name": "Patient B", "age": 60, "gender": "Female"},
    {"id": 3, "name": "Patient C", "age": 52, "gender": "Male"},
    {"id": 4, "name": "Patient D", "age": 70, "gender": "Female"}
]

# Generate vitals and backend data for a patient
def generate_patient_data(patient):
    vitals = {
        "heart_rate": random.randint(60, 100),  # Normal range: 60-100 BPM
        "spo2": random.randint(90, 100),       # Normal range: 95-100%
        "systolic_bp": random.randint(90, 140),  # Systolic BP
        "diastolic_bp": random.randint(60, 90)   # Diastolic BP
    }

    # Backend variables for diabetes and stroke models
    diabetes_features = [
        random.randint(0, 10),               # Pregnancies
        random.randint(70, 200),             # Glucose
        vitals['systolic_bp'],               # BloodPressure
        random.randint(10, 50),              # SkinThickness
        random.randint(15, 300),             # Insulin
        random.uniform(18, 40),              # BMI
        random.uniform(0.1, 2.5),            # DiabetesPedigreeFunction
        patient['age']                       # Age
    ]

    stroke_features = {
        "gender": 1 if patient['gender'] == "Male" else 0,
        "age": patient['age'],
        "hypertension": random.randint(0, 1),
        "heart_disease": random.randint(0, 1),
        "ever_married": 1,
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": random.uniform(70, 200),
        "bmi": random.uniform(18, 40),
        "smoking_status": "never smoked"
    }
    
    stroke_features_df = pd.DataFrame([stroke_features])
    stroke_features_df = pd.get_dummies(stroke_features_df, drop_first=True)
    missing_cols = set(stroke_model.feature_names_in_) - set(stroke_features_df.columns)
    for col in missing_cols:
        stroke_features_df[col] = 0
    stroke_features_df = stroke_features_df[stroke_model.feature_names_in_]

    # Predictions
    diabetes_risk = diabetes_model.predict_proba([diabetes_features])[0][1]
    stroke_risk = stroke_model.predict_proba(stroke_features_df)[0][1]

    return {**vitals, "diabetes_risk": diabetes_risk, "stroke_risk": stroke_risk}

# Dashboard route
@app.route('/')
def dashboard():
    # Generate vitals and predictions for each patient
    patient_data = []
    for patient in patients:
        data = generate_patient_data(patient)
        patient_data.append({**patient, **data})

    return render_template('dashboard.html', patients=patient_data)

# Fetch new vitals for a specific patient
@app.route('/fetch_vitals/<int:patient_id>')
def fetch_vitals(patient_id):
    # Find the patient and generate new data
    patient = next(p for p in patients if p["id"] == patient_id)
    new_data = generate_patient_data(patient)
    return jsonify(new_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
