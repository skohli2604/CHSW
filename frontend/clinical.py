from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# ClinicalTrials.gov API v2 Base URL
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Clinical Trial Search API!"})

@app.route('/search', methods=['GET'])
def search_trials():
    condition = request.args.get('condition')
    if not condition:
        return jsonify({"error": "Please provide a medical condition."}), 400
    
    params = {
        "query.term": condition,
        "format": "json",
        "pageSize": 10
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        return jsonify({"error": f"Failed to fetch data: {response.status_code} {response.reason}"}), response.status_code
    
    data = response.json()
    trials = []
    
    for study in data.get("studies", []):
        trials.append(study)  # Append the entire study object to include all fields
    
    return jsonify(trials)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)