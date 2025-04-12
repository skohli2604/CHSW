from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import random
import pickle
import pandas as pd
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Load pre-trained models
with open('diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('stroke_model.pkl', 'rb') as f:
    stroke_model = pickle.load(f)

# Predefined patient data with FIXED attributes
patients = [
    {"id": 1, "name": "Patient A", "age": 45, "gender": "Male", "bmi": 25.5, "work_type": "Private", "residence": "Urban", "ever_married": 1, "smoking_status": "never smoked", "diabetes_pedigree": 0.8, "pregnancies": 2,"hypertension": 0, "heart_disease": 1},
    {"id": 2, "name": "Patient B", "age": 60, "gender": "Female", "bmi": 28.7, "work_type": "Self-employed", "residence": "Rural", "ever_married": 1, "smoking_status": "formerly smoked", "diabetes_pedigree": 1.2, "pregnancies": 4,"hypertension": 1, "heart_disease": 0,},
    {"id": 3, "name": "Patient C", "age": 52, "gender": "Male", "bmi": 26.3, "work_type": "Govt_job", "residence": "Urban", "ever_married": 1, "smoking_status": "never smoked", "diabetes_pedigree": 0.6, "pregnancies": 1,"hypertension": 0, "heart_disease": 0},
    {"id": 4, "name": "Patient D", "age": 70, "gender": "Female", "bmi": 30.1, "work_type": "Private", "residence": "Rural", "ever_married": 1, "smoking_status": "smokes", "diabetes_pedigree": 0.9, "pregnancies": 3,"hypertension": 1, "heart_disease": 1}
]

# Generate dynamic vitals
def generate_dynamic_vitals(patient):
    vitals = {
        "heart_rate": random.randint(60, 100),
        "spo2": random.randint(90, 100),
        "systolic_bp": random.randint(90, 140),
        "diastolic_bp": random.randint(60, 90),
        "glucose": random.randint(70, 200),
        "insulin": random.randint(15, 300),
        "skin_thickness": random.randint(10, 50),
        "avg_glucose_level": random.uniform(70, 200),
    }

    # Diabetes model prediction
    diabetes_features = [
        patient["pregnancies"],  # Fixed
        vitals["glucose"],       # Dynamic
        vitals["systolic_bp"],   # Dynamic
        vitals["skin_thickness"], # Dynamic
        vitals["insulin"],       # Dynamic
        patient["bmi"],          # Fixed
        patient["diabetes_pedigree"],  # Fixed
        patient["age"]           # Fixed
    ]
    diabetes_risk = diabetes_model.predict_proba([diabetes_features])[0][1]

    # Stroke model prediction
    stroke_features = {
        "gender": 1 if patient["gender"] == "Male" else 0,  # Fixed
        "age": patient["age"],  # Fixed
        "hypertension": patient["hypertension"],  # Fixed
        "heart_disease": patient["heart_disease"],  # Fixed
        "ever_married": patient["ever_married"],  # Fixed
        "work_type": patient["work_type"],  # Fixed
        "Residence_type": patient["residence"],  # Fixed
        "avg_glucose_level": vitals["avg_glucose_level"],  # Dynamic
        "bmi": patient["bmi"],  # Fixed
        "smoking_status": patient["smoking_status"],  # Fixed
    }
    
    stroke_features_df = pd.DataFrame([stroke_features])
    stroke_features_df = pd.get_dummies(stroke_features_df, drop_first=True)
    missing_cols = set(stroke_model.feature_names_in_) - set(stroke_features_df.columns)
    for col in missing_cols:
        stroke_features_df[col] = 0
    stroke_features_df = stroke_features_df[stroke_model.feature_names_in_]
    stroke_risk = stroke_model.predict_proba(stroke_features_df)[0][1]

    return {**vitals, "diabetes_risk": diabetes_risk, "stroke_risk": stroke_risk}

# Store dynamic vitals separately
dynamic_vitals = {patient["id"]: generate_dynamic_vitals(patient) for patient in patients}

# Real-time update function
def update_vitals():
    while True:
        for patient in patients:
            patient_id = patient["id"]
            dynamic_vitals[patient_id] = generate_dynamic_vitals(patient)
            socketio.emit("update_vitals", {"patient_id": patient_id, **dynamic_vitals[patient_id]})
        time.sleep(5)  # Update every 5 seconds

# Start background thread for real-time updates
threading.Thread(target=update_vitals, daemon=True).start()

# Dashboard route
@app.route('/')
def dashboard():
    patient_data = [{**patient, **dynamic_vitals[patient["id"]]} for patient in patients]
    return render_template('dashboardtest.html', patients=patient_data)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8085)