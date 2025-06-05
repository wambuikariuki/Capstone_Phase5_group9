from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(
    filename="prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Load the XGBoost model
model = joblib.load('churn_model.pkl')

# Manual encoders based on training
complaint_map = {"Solved": 1, "Unsolved": 0}
feedback_map = {
    "Poor Product Quality": 0,
    "Poor Website": 1,
    "Too many ads": 2
}
region_map = {"City": 0, "Town": 1, "Village": 2}
membership_map = {
    "Basic": 0,
    "Silver": 1,
    "Gold": 2,
    "Premium": 3,
    "Platinum": 4
}
referral_map = {"Yes": 1, "No": 0}


def preprocess_input(data):
    try:
        processed = [
            float(data[0]),  # Age
            float(data[1]),  # Avg Time Spent
            float(data[2]),  # Avg Transaction Value
            float(data[3]),  # Avg Login Frequency
            complaint_map.get(data[4], 0),
            feedback_map.get(data[5], 0),
            region_map.get(data[6], 0),
            membership_map.get(data[7], 0),
            referral_map.get(data[8], 0),
            int(data[9])  # Placeholder or dummy value
        ]
        return np.array(processed).reshape(1, -1)
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        return None


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw = request.get_json(force=True).get("features", [])
        processed = preprocess_input(raw)

        if processed is None:
            return jsonify({'error': 'Invalid input format'}), 400

        prediction = model.predict(processed)[0]
        confidence = float(model.predict_proba(processed)[0][int(prediction)])

        logging.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}, Features: {raw}")

        return jsonify({
            'prediction': int(prediction),
            'confidence': confidence
        })
    except Exception as ex:
        logging.error(f"Prediction failed: {ex}")
        return jsonify({'error': 'Could not predict'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
