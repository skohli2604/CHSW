from flask import Flask, request, jsonify
import requests
from flask_cors import CORS  # To allow frontend requests

app = Flask(__name__)
CORS(app)  # Prevents CORS issues when calling from clinical.html

API_URL = "https://clinicaltrials.gov/api/v2/studies"

@app.route('/search', methods=['GET'])
def search_trials():
    term = request.args.get('term')
    if not term:
        return jsonify({"error": "Missing search term"}), 400

    headers = {"Accept": "application/json"}  # Required for ClinicalTrials.gov API
    response = requests.get(f"{API_URL}?query.term={term}", headers=headers)

    if response.status_code != 200:
        return jsonify({"error": f"Failed to fetch data: {response.status_code}"}), response.status_code

    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
