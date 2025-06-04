from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

# Load the model
model = joblib.load('churn_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    # Serve the styled HTML file from the "static" folder
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
