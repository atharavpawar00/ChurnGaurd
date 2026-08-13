# ChurnGuard — Customer Churn Prediction & Risk Analysis

A machine-learning web application that predicts customer churn probability using Logistic Regression on the IBM Telco Customer Churn dataset.

---

## Project Overview

ChurnGuard takes customer demographic, service, and billing information and returns:
- A **churn probability** (0–100%)
- A **risk level** (LOW / MEDIUM / HIGH)
- A **prediction** (Likely to Churn / Not Likely to Churn)

---

## Dataset

**IBM Telco Customer Churn**  
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

- ~7,043 customer records (7,032 after cleaning)
- Target: `Churn Label` → Yes (1) / No (0)
- Class imbalance: ~26% churn, ~74% no-churn

---

## Machine Learning Approach

### Preprocessing
- `Total Charges` coerced from string to float; ~11 unparseable rows dropped
- Categorical features → **OneHotEncoder** (`handle_unknown="ignore"`)
- Numerical features → **StandardScaler**
- Handled via scikit-learn `ColumnTransformer` inside a `Pipeline`

### Model
- **Logistic Regression** (`class_weight="balanced"`, `max_iter=1000`, `solver="lbfgs"`)
- Stratified 80/20 train/test split
- `random_state=42` for reproducibility

### Features Used (18)
**Categorical:** Senior Citizen, Partner, Dependents, Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, Contract, Paperless Billing, Payment Method

**Numerical:** Tenure Months, Monthly Charges, Total Charges

---

## Metrics

Actual metrics are generated at training time and stored in `model/metrics.json`.  
Typical results:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~80%   |
| Precision | ~65%   |
| Recall    | ~79%   |
| F1 Score  | ~71%   |
| ROC-AUC   | ~85%   |

---

## Project Structure

```
ChurnGuard/
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── requirements.txt
├── README.md
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── model/
│   ├── churn_model.joblib  # Trained pipeline (generated)
│   └── metrics.json        # Model metrics & chart data (generated)
├── templates/
│   └── index.html          # Frontend HTML
└── static/
    ├── style.css
    └── script.js
```

---

## Installation & Setup

```bash
pip install -r requirements.txt
```

---

## Training the Model

```bash
python train_model.py
```

This will:
1. Load and clean the dataset
2. Train a Logistic Regression pipeline
3. Evaluate on the held-out test set
4. Print actual metrics to the console
5. Save `model/churn_model.joblib`
6. Save `model/metrics.json` (metrics + chart data)

---

## Running the Application

```bash
python app.py
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## How Prediction Works

1. User fills in the customer profile form (defaults provided)
2. Frontend sends a POST request to `/predict` with JSON payload
3. Flask loads the trained pipeline and calls `predict_proba()`
4. The actual probability is returned — **no fake values**
5. Risk level is determined by thresholds: LOW < 40%, MEDIUM 40–70%, HIGH ≥ 70%
6. Result is displayed with a visual risk meter and profile summary

---

## API Endpoints

- `GET /` — serves the frontend
- `GET /metrics` — returns metrics.json
- `POST /predict` — accepts customer JSON, returns prediction

### Example `/predict` request

```json
{
  "Senior Citizen": "No",
  "Partner": "No",
  "Dependents": "No",
  "Phone Service": "Yes",
  "Multiple Lines": "No",
  "Internet Service": "Fiber optic",
  "Online Security": "No",
  "Online Backup": "No",
  "Device Protection": "No",
  "Tech Support": "No",
  "Streaming TV": "No",
  "Streaming Movies": "No",
  "Contract": "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check",
  "Tenure Months": 6,
  "Monthly Charges": 79.85,
  "Total Charges": 479.10
}
```

### Example response

```json
{
  "prediction": "Likely to Churn",
  "churn_probability": 0.7341,
  "risk_level": "HIGH"
}
```
