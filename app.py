from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = joblib.load('churn_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')  # Optional landing page

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)

        # DEBUG print to logs
        print("Features received:", features)

        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
