from flask import Flask, render_template
from flask_socketio import SocketIO
import random
import pickle
import pandas as pd

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')  # async mode required

# Load pre-trained models
with open('diabetes_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('stroke_model.pkl', 'rb') as f:
    stroke_model = pickle.load(f)

# Predefined patient data
patients = [
    {"id": 1, "name": "Patient A", "age": 45, "gender": "Male", "bmi": 25.5, "work_type": "Private", "residence": "Urban", "ever_married": 1, "smoking_status": "never smoked", "diabetes_pedigree": 0.8, "pregnancies": 2, "hypertension": 0, "heart_disease": 1},
    {"id": 2, "name": "Patient B", "age": 60, "gender": "Female", "bmi": 28.7, "work_type": "Self-employed", "residence": "Rural", "ever_married": 1, "smoking_status": "formerly smoked", "diabetes_pedigree": 1.2, "pregnancies": 4, "hypertension": 1, "heart_disease": 0},
    {"id": 3, "name": "Patient C", "age": 52, "gender": "Male", "bmi": 26.3, "work_type": "Govt_job", "residence": "Urban", "ever_married": 1, "smoking_status": "never smoked", "diabetes_pedigree": 0.6, "pregnancies": 1, "hypertension": 0, "heart_disease": 0},
    {"id": 4, "name": "Patient D", "age": 70, "gender": "Female", "bmi": 30.1, "work_type": "Private", "residence": "Rural", "ever_married": 1, "smoking_status": "smokes", "diabetes_pedigree": 0.9, "pregnancies": 3, "hypertension": 1, "heart_disease": 1}
]

# Dynamic vitals store
dynamic_vitals = {}

# Function to generate vitals + model predictions
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

    diabetes_features = [
        patient["pregnancies"],
        vitals["glucose"],
        vitals["systolic_bp"],
        vitals["skin_thickness"],
        vitals["insulin"],
        patient["bmi"],
        patient["diabetes_pedigree"],
        patient["age"]
    ]
    diabetes_risk = diabetes_model.predict_proba([diabetes_features])[0][1]

    stroke_features = {
        "gender": 1 if patient["gender"] == "Male" else 0,
        "age": patient["age"],
        "hypertension": patient["hypertension"],
        "heart_disease": patient["heart_disease"],
        "ever_married": patient["ever_married"],
        "work_type": patient["work_type"],
        "Residence_type": patient["residence"],
        "avg_glucose_level": vitals["avg_glucose_level"],
        "bmi": patient["bmi"],
        "smoking_status": patient["smoking_status"],
    }

    stroke_df = pd.DataFrame([stroke_features])
    stroke_df = pd.get_dummies(stroke_df, drop_first=True)
    missing_cols = set(stroke_model.feature_names_in_) - set(stroke_df.columns)
    for col in missing_cols:
        stroke_df[col] = 0
    stroke_df = stroke_df[stroke_model.feature_names_in_]
    stroke_risk = stroke_model.predict_proba(stroke_df)[0][1]

    return {**vitals, "diabetes_risk": diabetes_risk, "stroke_risk": stroke_risk}

# Update loop — emits data every 5 seconds
def update_vitals_loop():
    while True:
        for patient in patients:
            patient_id = patient["id"]
            dynamic_data = generate_dynamic_vitals(patient)
            dynamic_vitals[patient_id] = dynamic_data
            socketio.emit("update_vitals", {"patient_id": patient_id, **dynamic_data})
        socketio.sleep(5)

# Start background task when a client connects
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.start_background_task(update_vitals_loop)

# Dashboard route
@app.route('/')
def dashboard():
    # Initial data to render
    patient_data = [{**patient, **dynamic_vitals.get(patient["id"], generate_dynamic_vitals(patient))} for patient in patients]
    return render_template('dashboardtest.html', patients=patient_data)

# Run the app
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5050)
