from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import pickle

app = Flask(__name__)
CORS(app)

# ✅ Load model




import joblib
model = joblib.load("model.pkl")





@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get('features')
    prediction = model.predict(np.array(features).reshape(1, -1))
    print("Prediction:", prediction)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
