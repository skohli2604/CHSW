import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

print("Starting script...")

# Load Diabetes Dataset
print("Loading Diabetes dataset...")
diabetes_data = pd.read_csv('/Users/shreyakohli/Downloads/diabetes.csv')
X_diabetes = diabetes_data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y_diabetes = diabetes_data['Outcome']
print(f"Diabetes dataset loaded with shape {X_diabetes.shape}")

# Load Stroke Dataset
print("Loading Stroke dataset...")
stroke_data = pd.read_csv('/Users/shreyakohli/Downloads/stroke.csv')
stroke_data = stroke_data.dropna()
stroke_data = pd.get_dummies(stroke_data, drop_first=True)
X_stroke = stroke_data.drop(columns=['stroke'])
y_stroke = stroke_data['stroke']
print(f"Stroke dataset loaded with shape {X_stroke.shape}")

# Standardization
print("Standardizing datasets...")
scaler = StandardScaler()
X_diabetes = scaler.fit_transform(X_diabetes)
X_stroke = scaler.fit_transform(X_stroke)

# Train-Test Split
print("Splitting datasets into train and test sets...")
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_stroke, y_stroke, test_size=0.2, random_state=42)

# Train Logistic Regression Models
print("Training Logistic Regression models...")
diabetes_model = LogisticRegression()
diabetes_model.fit(X_train_d, y_train_d)

stroke_model = LogisticRegression(max_iter=1000)
stroke_model.fit(X_train_s, y_train_s)
print("Models trained successfully!")

# Save models as Pickle
print("Saving models as Pickle files...")
pickle.dump(diabetes_model, open("diabetes_model.pkl", "wb"))
pickle.dump(stroke_model, open("stroke_model.pkl", "wb"))
print("Models saved as Pickle!")

# Convert to TensorFlow Keras models
def sklearn_to_tf_keras(sklearn_model, input_shape):
    print(f"Converting model to TensorFlow Keras format with input shape {input_shape}...")
    keras_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.Constant(sklearn_model.coef_.T))
    ])
    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return keras_model

print("Converting models to TensorFlow Lite format...")

# Convert and save TFLite models
def convert_to_tflite(tf_model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted and saved to {filename}")

# Convert Sklearn models to TF Keras
tf_diabetes_model = sklearn_to_tf_keras(diabetes_model, X_test_d.shape[1])
tf_stroke_model = sklearn_to_tf_keras(stroke_model, X_test_s.shape[1])

# Save as TFLite
convert_to_tflite(tf_diabetes_model, "diabetes_model.tflite")
convert_to_tflite(tf_stroke_model, "stroke_model.tflite")

print("All conversions completed successfully!")

