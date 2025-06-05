from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load the XGBoost model
model = joblib.load('churn_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    if hasattr(model, 'predict_proba'):
        confidence = float(model.predict_proba(features)[0][int(prediction)])
    else:
        confidence = None
    return jsonify({
        'prediction': int(prediction),
        'confidence': confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
