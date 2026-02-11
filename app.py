from flask_cors import CORS
from flask import Flask, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)
CORS(app)

app = Flask(__name__)

# Load ML model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "Safe Guard Backend is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Example: expecting features list from frontend
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
