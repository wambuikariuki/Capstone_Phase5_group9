from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

# Load the model
model = joblib.load('churn_model.pkl')

app = Flask(__name__)

# HTML with 21 input fields
html_form = '''
<!DOCTYPE html>
<html>
<head>
  <title>Churn Prediction</title>
</head>
<body>
  <h2>Customer Churn Prediction</h2>
  <form id="predictForm">
    {% for i in range(21) %}
      <label>Feature {{ i+1 }}: <input type="number" step="any" name="f{{ i }}"></label><br>
    {% endfor %}
    <button type="submit">Predict</button>
  </form>
  <p id="result"></p>

  <script>
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const inputs = Array.from(document.querySelectorAll('input')).map(i => parseFloat(i.value) || 0);
      
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: inputs })
      });

      const data = await response.json();
      document.getElementById('result').innerText = 
        'Prediction: ' + (data.prediction === 1 ? 'Customer will CHURN' : 'Customer will NOT churn');
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
