import os
import sys
import json
import time
import random
import threading
import pandas as pd
from flask import Flask, render_template, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.sql.functions import udf, col, avg, count

# -------------------------------
#  Configuration and Environment
# -------------------------------

# Set environment variables for PySpark if needed (adjust paths as appropriate)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# -------------------------------
#  Flask and Spark Initialization
# -------------------------------

app = Flask(__name__)

# Initialize Spark session (local mode)
spark = SparkSession.builder \
    .appName("HealthcareSparkApp") \
    .master("local[*]") \
    .getOrCreate()

# -------------------------------
#  Simulated Patient Data and Models
# -------------------------------

# Hardcode 100 patient records (here we use a simple list; for demo, we generate 10 for brevity)
patients = []
for i in range(1, 101):
    patients.append({
        "id": i,
        "name": f"Patient {i}",
        "age": random.randint(30, 80),
        "gender": random.choice(["Male", "Female"]),
        "bmi": round(random.uniform(18, 35), 1)
        # add more static fields as needed
    })

# For demonstration, we define a dummy model as a function
# This "model" returns a risk score based on glucose and systolic blood pressure.
def dummy_risk_model(glucose, systolic_bp):
    # A simple formula for risk – this is only a placeholder
    risk = (glucose / 200) * 0.6 + (systolic_bp / 140) * 0.4
    return float(round(risk, 2))

# Register the dummy risk model as a Spark UDF
risk_udf = udf(dummy_risk_model, FloatType())

# -------------------------------
#  Simulated Streaming Data Generation
# -------------------------------

# Directory where simulated streaming JSON files will be written
STREAMING_DIR = "streaming_data"
if not os.path.exists(STREAMING_DIR):
    os.makedirs(STREAMING_DIR)

def simulate_streaming_data():
    """
    Every few seconds, write a JSON file into the streaming_data folder.
    Each JSON file contains simulated vitals for a random patient.
    """
    while True:
        # Choose a random patient
        patient = random.choice(patients)
        vitals = {
            "patient_id": patient["id"],
            "glucose": random.randint(70, 200),
            "systolic_bp": random.randint(90, 140),
            "diastolic_bp": random.randint(60, 90),
            "heart_rate": random.randint(60, 100),
            "timestamp": int(time.time())
        }
        # Write a single JSON record to a new file (use timestamp to avoid collision)
        filename = os.path.join(STREAMING_DIR, f"data_{time.time()}.json")
        with open(filename, "w") as f:
            json.dump(vitals, f)
        time.sleep(3)  # simulate data every 3 seconds

# Start the streaming simulation in a background thread
threading.Thread(target=simulate_streaming_data, daemon=True).start()

# -------------------------------
#  Spark Structured Streaming Setup
# -------------------------------

# Define the schema of the streaming data
schema = StructType([
    StructField("patient_id", IntegerType(), True),
    StructField("glucose", IntegerType(), True),
    StructField("systolic_bp", IntegerType(), True),
    StructField("diastolic_bp", IntegerType(), True),
    StructField("heart_rate", IntegerType(), True),
    StructField("timestamp", IntegerType(), True)
])

# Create a streaming DataFrame reading from the simulated directory
streaming_df = spark \
    .readStream \
    .schema(schema) \
    .json(STREAMING_DIR)

# Add a "risk" column using our UDF
streaming_df_with_risk = streaming_df.withColumn("risk", risk_udf(col("glucose"), col("systolic_bp")))

# For this example, we will write the streaming output to a temporary in-memory table.
query = streaming_df_with_risk.writeStream \
    .format("memory") \
    .queryName("vitals_table") \
    .outputMode("append") \
    .start()

# -------------------------------
#  Flask Routes for Dashboard and Analytics
# -------------------------------

@app.route('/')
def dashboard():
    """
    Dashboard route that will:
    - Read the latest batch of streaming data from Spark's in-memory table.
    - Run analytics queries using Spark SQL.
    - Return JSON results (or render them in an HTML template).
    """
    # Wait a moment for data to accumulate
    time.sleep(5)
    
    # Use Spark SQL to query data from the in-memory table
    spark.sql("CACHE TABLE vitals_table")
    
    # Example Analytics 1: Compute average risk across all patients
    avg_risk_df = spark.sql("SELECT AVG(risk) as avg_risk FROM vitals_table")
    avg_risk = avg_risk_df.collect()[0]["avg_risk"]

    # Example Analytics 2: Count records per patient
    count_df = spark.sql("SELECT patient_id, COUNT(*) as count FROM vitals_table GROUP BY patient_id ORDER BY count DESC LIMIT 10")
    counts = count_df.toPandas().to_dict(orient="records")
    
    # Also get a sample batch of raw data (e.g., latest 10 records)
    raw_df = spark.sql("SELECT * FROM vitals_table ORDER BY timestamp DESC LIMIT 10")
    raw_data = raw_df.toPandas().to_dict(orient="records")

    # Return combined analytics as JSON (you could also render an HTML dashboard)
    return jsonify({
        "avg_risk": avg_risk,
        "record_counts": counts,
        "latest_records": raw_data
    })

@app.route('/patient/<int:patient_id>')
def patient_data(patient_id):
    """
    Return analytics for a specific patient.
    """
    df = spark.sql(f"SELECT * FROM vitals_table WHERE patient_id = {patient_id} ORDER BY timestamp DESC LIMIT 10")
    data = df.toPandas().to_dict(orient="records")
    return jsonify(data)

# -------------------------------
#  Main: Run Flask App
# -------------------------------

if __name__ == '__main__':
    # Start Flask on port 8080 (or any port you choose)
    app.run(debug=True, host='0.0.0.0', port=8080)