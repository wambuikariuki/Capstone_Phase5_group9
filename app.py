from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

model = joblib.load('churn_model.pkl')

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_ui():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
