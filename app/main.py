"""
Fraud Detection API
--------------------
FastAPI service that scores mobile-money transactions for account-takeover
fraud using a trained Random Forest model, and reports how that model
compares against the legacy static rule (flag TRANSFER/CASH_OUT > $200,000).

Endpoints:
  POST /predict     Score a single transaction
  GET  /model-info   Training metrics, feature importances, static-rule comparison
  GET  /health        Liveness/readiness check
  GET  /docs           Auto-generated Swagger UI
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    TransactionRequest,
    TransactionResponse,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
STATIC_RULE_THRESHOLD = 200_000

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Scores mobile-money transactions for account-takeover fraud risk and "
        "benchmarks the model against a legacy static threshold rule."
    ),
    version="1.0.0",
)

_model = None
_metadata = None


@app.on_event("startup")
def load_artifacts():
    global _model, _metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run scripts/train_model.py first."
        )
    _model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        _metadata = json.load(f)


def _engineer_features(txn: TransactionRequest) -> pd.DataFrame:
    balance_drained_ratio = (txn.oldbalanceOrg - txn.newbalanceOrig) / (txn.oldbalanceOrg + 1.0)
    balance_drained_ratio = min(max(balance_drained_ratio, 0.0), 1.0)
    is_night = 1 if 0 <= txn.hour_of_day <= 5 else 0

    return pd.DataFrame([{
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "newbalanceOrig": txn.newbalanceOrig,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
        "dest_txn_history": txn.dest_txn_history,
        "hour_of_day": txn.hour_of_day,
        "balance_drained_ratio": balance_drained_ratio,
        "is_night": is_night,
        "type": txn.type,
    }])


def _static_rule(txn: TransactionRequest) -> bool:
    return txn.type in ("TRANSFER", "CASH_OUT") and txn.amount > STATIC_RULE_THRESHOLD


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    if _metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    return _metadata


@app.post("/predict", response_model=TransactionResponse, tags=["Scoring"])
def predict(txn: TransactionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = _engineer_features(txn)
    fraud_score = float(_model.predict_proba(features)[0, 1])
    threshold = _metadata.get("decision_threshold", 0.5)
    model_flag = fraud_score >= threshold
    rule_flag = _static_rule(txn)

    return TransactionResponse(
        fraud_score=round(fraud_score, 4),
        model_flag=model_flag,
        static_rule_flag=rule_flag,
        flags_agree=model_flag == rule_flag,
        decision_threshold=threshold,
    )


@app.get("/", tags=["Ops"])
def root():
    return {
        "service": "Fraud Detection API",
        "docs": "/docs",
        "endpoints": ["/predict", "/model-info", "/health"],
    }
