from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

# Load model
model = joblib.load('churn_model.pkl')

app = Flask(__name__)

html_form = '''
<!DOCTYPE html>
<html>
<head>
  <title>Churn Predictor</title>
</head>
<body>
  <h2>Customer Churn Prediction</h2>
  <form id="predictForm">
    <input type="text" id="features" placeholder="Enter 10 comma-separated values" size="60">
    <button type="submit">Predict</button>
  </form>
  <p id="result"></p>

  <script>
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const input = document.getElementById('features').value.split(',').map(Number);

      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: input })
      });

      const data = await response.json();
      document.getElementById('result').innerText = 'Prediction: ' + (data.prediction === 1 ? 'Will Churn' : 'Will Not Churn');
    });
  </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(html_form)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
