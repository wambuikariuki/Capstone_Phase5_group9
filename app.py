from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

model = joblib.load("churn_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get("features")
    if not features or len(features) != 10:
        return jsonify({"error": "Expected 10 features"}), 400

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

   
