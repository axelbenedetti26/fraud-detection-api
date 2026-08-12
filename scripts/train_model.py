"""
train_model.py
---------------
Trains a Random Forest classifier on the synthetic mobile-money transaction
dataset and benchmarks it against the legacy static rule used as a baseline:

    STATIC RULE: flag as fraud if type in {TRANSFER, CASH_OUT} and amount > $200,000

The static rule is what a lot of real mobile-money platforms still run in
production -- a flat amount threshold on high-risk transaction types. It's
cheap and explainable, but it only catches fraud that happens to be large.
Account-takeover fraud is sized to look like a normal big transfer, so the
static rule mostly misses it.

Outputs:
  - models/fraud_model.pkl        (trained sklearn Pipeline: preprocessing + RF)
  - models/model_metadata.json    (metrics + feature list, consumed by the API's /model-info)
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "models/fraud_model.pkl"
METADATA_PATH = "models/model_metadata.json"
STATIC_RULE_THRESHOLD = 200_000
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "dest_txn_history",
    "hour_of_day",
    "balance_drained_ratio",
    "is_night",
]
CATEGORICAL_FEATURES = ["type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balance_drained_ratio"] = (
        (df["oldbalanceOrg"] - df["newbalanceOrig"]) / (df["oldbalanceOrg"] + 1.0)
    ).clip(0, 1)
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
    return df


def static_rule_flag(df: pd.DataFrame) -> np.ndarray:
    return (
        df["type"].isin(["TRANSFER", "CASH_OUT"]) & (df["amount"] > STATIC_RULE_THRESHOLD)
    ).astype(int).values


def main():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    X = df[ALL_FEATURES]
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_seconds = time.time() - t0

    # Model evaluation at default 0.5 threshold
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    model_recall = recall_score(y_test, y_pred)
    model_precision = precision_score(y_test, y_pred)
    model_auc = roc_auc_score(y_test, y_proba)
    model_avg_precision = average_precision_score(y_test, y_proba)
    model_accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Static rule evaluation: the rule isn't "trained", so we evaluate it over the
    # full labeled dataset rather than just the small held-out test split -- with
    # only ~700 fraud cases total, a test-only slice is too small to reliably show
    # how a flat-threshold rule performs on rare, large-value fraud.
    rule_pred_full = static_rule_flag(df)
    rule_recall = recall_score(y, rule_pred_full)
    rule_precision = precision_score(y, rule_pred_full, zero_division=0)
    r_tn, r_fp, r_fn, r_tp = confusion_matrix(y, rule_pred_full).ravel()

    # Agreement rate is still measured on the held-out test set for an apples-to-apples read
    rule_pred_test = static_rule_flag(df_test)
    agreement_rate = float((y_pred == rule_pred_test).mean())

    # Threshold trade-off curve: recall/precision at a range of decision thresholds,
    # so the frontend can show that 0.5 is a choice, not a magic number.
    threshold_curve = []
    for t in np.arange(0.05, 1.0, 0.05):
        pred_t = (y_proba >= t).astype(int)
        threshold_curve.append(
            {
                "threshold": round(float(t), 2),
                "recall": round(float(recall_score(y_test, pred_t, zero_division=0)), 4),
                "precision": round(float(precision_score(y_test, pred_t, zero_division=0)), 4),
            }
        )

    # Dollar impact: real fraud-dollar totals from the test set (not a hypothetical
    # scenario) -- how much fraud value each approach actually catches vs. misses.
    fraud_mask = (y_test == 1)
    test_amounts = df_test["amount"].values
    total_fraud_usd = float(test_amounts[fraud_mask].sum())
    model_caught_usd = float(test_amounts[fraud_mask & (y_pred == 1)].sum())
    rule_caught_usd = float(test_amounts[fraud_mask & (rule_pred_test == 1)].sum())
    dollar_impact = {
        "test_set_fraud_total_usd": round(total_fraud_usd, 2),
        "model_caught_usd": round(model_caught_usd, 2),
        "model_missed_usd": round(total_fraud_usd - model_caught_usd, 2),
        "rule_caught_usd": round(rule_caught_usd, 2),
        "rule_missed_usd": round(total_fraud_usd - rule_caught_usd, 2),
    }

    print("=== Random Forest (model) ===")
    print(f"  AUC:        {model_auc:.4f}")
    print(f"  Recall:     {model_recall:.4f}")
    print(f"  Precision:  {model_precision:.4f}")
    print(f"  Accuracy:   {model_accuracy:.4f}")
    print(f"  Confusion:  TP={tp} FP={fp} FN={fn} TN={tn}")
    print()
    print("=== Static rule (amount > $200k on TRANSFER/CASH_OUT), full dataset ===")
    print(f"  Recall:     {rule_recall:.4f}")
    print(f"  Precision:  {rule_precision:.4f}")
    print(f"  Confusion:  TP={r_tp} FP={r_fp} FN={r_fn} TN={r_tn}")
    print()
    print(f"Model/rule agreement rate on test set: {agreement_rate:.4f}")

    feature_importances = dict(
        sorted(
            zip(
                NUMERIC_FEATURES
                + list(
                    pipeline.named_steps["preprocess"]
                    .named_transformers_["cat"]
                    .get_feature_names_out(CATEGORICAL_FEATURES)
                ),
                pipeline.named_steps["model"].feature_importances_,
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    metadata = {
        "model_type": "RandomForestClassifier",
        "sklearn_pipeline": True,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "train_seconds": round(train_seconds, 2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "features": ALL_FEATURES,
        "decision_threshold": 0.5,
        "static_rule": {
            "description": f"Flag TRANSFER/CASH_OUT transactions with amount > ${STATIC_RULE_THRESHOLD:,}",
            "evaluated_on": "full_dataset",
            "threshold_usd": STATIC_RULE_THRESHOLD,
            "recall": round(float(rule_recall), 4),
            "precision": round(float(rule_precision), 4),
            "true_positives": int(r_tp),
            "false_positives": int(r_fp),
            "false_negatives": int(r_fn),
            "true_negatives": int(r_tn),
        },
        "model_metrics": {
            "auc": round(float(model_auc), 4),
            "average_precision": round(float(model_avg_precision), 4),
            "recall": round(float(model_recall), 4),
            "precision": round(float(model_precision), 4),
            "accuracy": round(float(model_accuracy), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        },
        "model_vs_rule_agreement_rate": round(agreement_rate, 4),
        "feature_importances": {k: round(float(v), 4) for k, v in feature_importances.items()},
        "threshold_curve": threshold_curve,
        "dollar_impact": dollar_impact,
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
