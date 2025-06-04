from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load('churn_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
