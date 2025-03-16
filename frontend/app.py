from flask import Flask, jsonify, request, render_template
import pymysql
import pickle
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# MySQL connection
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="ShreyaRocks",
    database="disease_prediction"
)

# Train the diabetes model and save it
def train_diabetes_model():
    if not os.path.exists('diabetes_model.pkl'):
        data = pd.read_csv('/Users/shreyakohli/Downloads/diabetes.csv')
        X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        y = data['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        with open('diabetes_model.pkl', 'wb') as file:
            pickle.dump(model, file)

# Train the stroke model and save it
# Train the stroke model and save it
def train_stroke_model():
    if not os.path.exists('stroke_model.pkl'):
        # Load the stroke dataset
        data = pd.read_csv('/Users/shreyakohli/Downloads/stroke.csv')
        
        # Check for missing or invalid values
        data = data.dropna()  # Drop rows with missing values
        data = data.replace([float('inf'), float('-inf')], float('nan'))  # Replace infinities with NaN
        data = data.dropna()  # Drop rows with NaN values after replacement
        
        # Feature selection
        X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                  'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
        y = data['stroke']

        # Convert categorical variables to numerical values using one-hot encoding
        X = pd.get_dummies(X, drop_first=True)

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence
        model.fit(X_train, y_train)

        # Save the trained model
        with open('stroke_model.pkl', 'wb') as file:
            pickle.dump(model, file)




train_diabetes_model()
train_stroke_model()

# Load the models
with open('diabetes_model.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('stroke_model.pkl', 'rb') as file:
    stroke_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Diabetes prediction
    diabetes_features = [
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['blood_pressure']),
        float(data['skin_thickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['diabetes_pedigree_function']),
        float(data['age'])
    ]
    diabetes_prediction = diabetes_model.predict([diabetes_features])[0]
    diabetes_risk_probability = diabetes_model.predict_proba([diabetes_features])[0][1]

    cursor = connection.cursor()
    diabetes_query = """
        INSERT INTO diabetes_predictions (
            pregnancies, glucose, blood_pressure, skin_thickness, insulin,
            bmi, diabetes_pedigree_function, age, outcome, risk_probability
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(diabetes_query, (*diabetes_features, int(diabetes_prediction), float(diabetes_risk_probability)))

    # Stroke prediction
    stroke_features = {
        'gender': data['gender'],
        'age': float(data['age']),
        'hypertension': int(data['hypertension']),
        'heart_disease': int(data['heart_disease']),
        'ever_married': data['ever_married'],
        'work_type': data['work_type'],
        'Residence_type': data['residence_type'],
        'avg_glucose_level': float(data['avg_glucose_level']),
        'bmi': float(data['bmi']),
        'smoking_status': data['smoking_status']
    }
    stroke_features_df = pd.DataFrame([stroke_features])
    stroke_features_df = pd.get_dummies(stroke_features_df, drop_first=True)
    missing_cols = set(stroke_model.feature_names_in_) - set(stroke_features_df.columns)
    for col in missing_cols:
        stroke_features_df[col] = 0
    stroke_features_df = stroke_features_df[stroke_model.feature_names_in_]

    stroke_prediction = stroke_model.predict(stroke_features_df)[0]
    stroke_risk_probability = stroke_model.predict_proba(stroke_features_df)[0][1]

    stroke_query = """
        INSERT INTO stroke_predictions (
            gender, age, hypertension, heart_disease, ever_married, work_type,
            residence_type, avg_glucose_level, bmi, smoking_status, outcome, risk_probability
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(stroke_query, (
        data['gender'], float(data['age']), int(data['hypertension']), int(data['heart_disease']),
        data['ever_married'], data['work_type'], data['residence_type'],
        float(data['avg_glucose_level']), float(data['bmi']), data['smoking_status'],
        int(stroke_prediction), float(stroke_risk_probability)
    ))

    connection.commit()
    cursor.close()

    return jsonify({
        "diabetes_prediction": {
            "outcome": int(diabetes_prediction),
            "risk_probability": round(float(diabetes_risk_probability), 2)
        },
        "stroke_prediction": {
            "outcome": int(stroke_prediction),
            "risk_probability": round(float(stroke_risk_probability), 2)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

