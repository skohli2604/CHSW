from flask import Flask, jsonify, request
import pymysql

app = Flask(__name__)

# MySQL connection
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="ShreyaRocks",
    database="disease_prediction"
)

# Allowed table names to prevent SQL injection
ALLOWED_TABLES = {"diabetes", "stroke"}

@app.route('/get-data/<table_name>', methods=['GET'])
def get_data(table_name):
    # Validate table name to prevent SQL injection
    if table_name not in ALLOWED_TABLES:
        return jsonify({"error": "Invalid table name"}), 400

    cursor = connection.cursor()
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    return jsonify(rows)

if __name__ == "__main__":
    app.run(debug=True)
