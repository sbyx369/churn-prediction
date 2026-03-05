# ================================
# CUSTOMER CHURN PREDICTION SYSTEM
# Phase 6 - Flask API
# ================================

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ---- Load model and scaler ----
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---- Health check route ----
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': '✅ Churn Prediction API is running!',
        'usage': 'Send POST request to /predict'
    })

# ---- Prediction route ----
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Scale numerical columns
        df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            'churn_prediction': 'Yes' if prediction == 1 else 'No',
            'churn_probability': round(float(probability) * 100, 2),
            'message': '⚠️ High churn risk!' if prediction == 1 else '✅ Low churn risk'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

import os
port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)
