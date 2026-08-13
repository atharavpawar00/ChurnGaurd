"""
ChurnGuard - Model Training Script
Run: python train_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

# ── 2. Clean data ─────────────────────────────────────────────────────────────
print("Cleaning data …")

# TotalCharges comes in as string in this version – coerce to numeric
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

# Drop rows where Total Charges couldn't be parsed (typically ~11 rows)
before = len(df)
df.dropna(subset=["Total Charges"], inplace=True)
print(f"  Dropped {before - len(df)} rows with non-numeric Total Charges.")

# Target: Churn Label → 0/1
df["target"] = (df["Churn Label"] == "Yes").astype(int)
print(f"  Churn distribution:\n{df['target'].value_counts().to_string()}")

# ── 3. Feature selection ──────────────────────────────────────────────────────
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

X = df[FEATURES]
y = df["target"]

print(f"\nFeatures used ({len(FEATURES)}):")
print("  Categorical:", CATEGORICAL_FEATURES)
print("  Numerical  :", NUMERICAL_FEATURES)

# ── 4. Train / test split (stratified) ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ── 5. Preprocessing pipeline ─────────────────────────────────────────────────
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ("num", num_pipeline, NUMERICAL_FEATURES),
])

# ── 6. Full model pipeline ────────────────────────────────────────────────────
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",   # handles class imbalance
        solver="lbfgs"
    ))
])

print("\nTraining Logistic Regression …")
model_pipeline.fit(X_train, y_train)
print("  Training complete.")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating model …")
y_pred  = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_proba)
cm        = confusion_matrix(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

print(f"\n{'='*45}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  Confusion Matrix:")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
print(f"{'='*45}\n")

# ── 8. Dataset-level insight: churn by contract type ─────────────────────────
churn_by_contract = (
    df.groupby("Contract")["target"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "churned", "count": "total"})
    .reset_index()
)
churn_by_contract["churn_rate"] = (
    churn_by_contract["churned"] / churn_by_contract["total"]
).round(4)

# Churn by tenure group
df["tenure_group"] = pd.cut(
    df["Tenure Months"],
    bins=[0, 12, 24, 36, 48, 60, 9999],
    labels=["0-12", "13-24", "25-36", "37-48", "49-60", "60+"]
)
churn_by_tenure = (
    df.groupby("tenure_group", observed=True)["target"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "churned", "count": "total"})
    .reset_index()
)
churn_by_tenure["churn_rate"] = (
    churn_by_tenure["churned"] / churn_by_tenure["total"]
).round(4)

# Overall churn distribution for the dataset
churn_dist = df["target"].value_counts().to_dict()

# ── 9. Save metrics & chart data ──────────────────────────────────────────────
metrics = {
    "accuracy" : round(accuracy, 4),
    "precision": round(precision, 4),
    "recall"   : round(recall, 4),
    "f1"       : round(f1, 4),
    "roc_auc"  : round(roc_auc, 4),
    "confusion_matrix": {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    },
    "roc_curve": {
        "fpr": [round(float(v), 4) for v in fpr[::5]],   # downsample for JSON size
        "tpr": [round(float(v), 4) for v in tpr[::5]],
    },
    "dataset": {
        "total_customers" : int(len(df)),
        "no_churn"        : int(churn_dist.get(0, 0)),
        "churn"           : int(churn_dist.get(1, 0)),
        "train_size"      : int(len(X_train)),
        "test_size"       : int(len(X_test)),
    },
    "churn_by_contract": churn_by_contract.to_dict(orient="records"),
    "churn_by_tenure"  : churn_by_tenure.to_dict(orient="records"),
    "features": {
        "categorical": CATEGORICAL_FEATURES,
        "numerical"  : NUMERICAL_FEATURES,
    }
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved → {METRICS_PATH}")

# ── 10. Save model pipeline ───────────────────────────────────────────────────
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model saved  → {MODEL_PATH}")
print("\nDone! Run `python app.py` to start the web app.")
