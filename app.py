"""
ChurnGuard - Flask Application
Run: python app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "churn_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

# ── Load model & metrics at startup ───────────────────────────────────────────
model   = None
metrics = None

def load_artifacts():
    global model, metrics
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Please run:  python train_model.py"
        )
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics file not found at {METRICS_PATH}. "
            "Please run:  python train_model.py"
        )
    model   = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    print("Model and metrics loaded successfully.")

load_artifacts()

# ── Feature lists (must match train_model.py exactly) ─────────────────────────
CATEGORICAL_FEATURES = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ── Valid categorical values ───────────────────────────────────────────────────
VALID_VALUES = {
    "Senior Citizen"   : ["No", "Yes"],
    "Partner"          : ["No", "Yes"],
    "Dependents"       : ["No", "Yes"],
    "Phone Service"    : ["No", "Yes"],
    "Multiple Lines"   : ["No", "Yes", "No phone service"],
    "Internet Service" : ["DSL", "Fiber optic", "No"],
    "Online Security"  : ["No", "Yes", "No internet service"],
    "Online Backup"    : ["No", "Yes", "No internet service"],
    "Device Protection": ["No", "Yes", "No internet service"],
    "Tech Support"     : ["No", "Yes", "No internet service"],
    "Streaming TV"     : ["No", "Yes", "No internet service"],
    "Streaming Movies" : ["No", "Yes", "No internet service"],
    "Contract"         : ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["No", "Yes"],
    "Payment Method"   : [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def get_metrics():
    if metrics is None:
        return jsonify({"error": "Metrics not loaded."}), 500
    return jsonify(metrics)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Run python train_model.py first."}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or empty JSON body."}), 400

    # ── Validate & parse inputs ───────────────────────────────────────────────
    errors = []

    # Categorical
    row = {}
    for feat in CATEGORICAL_FEATURES:
        val = data.get(feat)
        if val is None:
            errors.append(f"Missing field: '{feat}'")
            continue
        val = str(val).strip()
        allowed = VALID_VALUES.get(feat, [])
        if allowed and val not in allowed:
            errors.append(f"Invalid value for '{feat}': '{val}'. Allowed: {allowed}")
        row[feat] = val

    # Numerical
    for feat in NUMERICAL_FEATURES:
        val = data.get(feat)
        if val is None:
            errors.append(f"Missing field: '{feat}'")
            continue
        try:
            num = float(val)
            if num < 0:
                errors.append(f"'{feat}' must be >= 0, got {num}")
            row[feat] = num
        except (ValueError, TypeError):
            errors.append(f"'{feat}' must be a number, got '{val}'")

    if errors:
        return jsonify({"error": "Validation failed.", "details": errors}), 422

    # ── Build DataFrame and predict ───────────────────────────────────────────
    try:
        input_df = pd.DataFrame([row], columns=FEATURES)
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction  = "Likely to Churn" if probability >= 0.5 else "Not Likely to Churn"

        if probability < 0.40:
            risk_level = "LOW"
        elif probability < 0.70:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return jsonify({
            "prediction"       : prediction,
            "churn_probability": round(probability, 4),
            "risk_level"       : risk_level,
        })

    except Exception as e:
        app.logger.error("Prediction error: %s", str(e))
        return jsonify({"error": "Prediction failed. Please check your inputs."}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  ChurnGuard is running!")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
