from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
import os
import logging
import csv
from datetime import datetime

app = Flask(__name__)

# Load model
model = joblib.load("churn_model.pkl")

# Setup logging
LOG_FILE = "prediction_logs.csv"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Prediction", "Confidence", "Features"])

# Mappings matching front-end form exactly
complaint_map = {"Solved": 1, "Unsolved": 0}
feedback_map = {
    "Poor Product Quality": 0,
    "Poor Website": 1,
    "Too many ads": 2
}
region_map = {"City": 0, "Town": 1, "Village": 2}
membership_map = {
    "Basic": 0, "Silver": 1, "Gold": 2, "Premium": 3, "Platinum": 4
}
referral_map = {"Yes": 1, "No": 0}

def preprocess_input(data):
    try:
        processed = [
            float(data[0]),  # age
            float(data[1]),  # avg_time_spent
            float(data[2]),  # avg_transaction_value
            float(data[3]),  # avg_login_freq
            complaint_map.get(data[4], 0),
            feedback_map.get(data[5], 0),
            region_map.get(data[6], 0),
            membership_map.get(data[7], 0),
            referral_map.get(data[8], 0),
            float(data[9])  # dummy value
        ]
        return np.array(processed).reshape(1, -1)
    except Exception as e:
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')



@app.route('/recent-logs')
def recent_logs():
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()[1:]  # skip header
            recent = lines[-5:] if len(lines) >= 5 else lines
            recent_data = [line.strip().split(',') for line in recent]
        return jsonify(recent_data)
    except Exception as ex:
        return jsonify({'error': f'Unable to read logs: {str(ex)}'}), 500, 400

        prediction = int(model.predict(features_array)[0])
        confidence = float(model.predict_proba(features_array)[0][prediction])

        log_row = [datetime.now().isoformat(), prediction, f"{confidence:.4f}", str(raw)]
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_row)

        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as ex:
        return jsonify({'error': f'Prediction failed: {str(ex)}'}), 500

@app.route('/download-logs')
def download_logs():
    return send_file(LOG_FILE, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

