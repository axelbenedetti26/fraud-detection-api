from typing import Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    type: Literal["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"] = Field(
        ..., description="Mobile money transaction type"
    )
    amount: float = Field(..., gt=0, description="Transaction amount (USD)")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before the transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after the transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Recipient balance before the transaction")
    newbalanceDest: float = Field(..., ge=0, description="Recipient balance after the transaction")
    dest_txn_history: int = Field(
        ..., ge=0, description="Number of prior transactions on the recipient account (mule-account signal)"
    )
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day the transaction occurred (0-23)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "TRANSFER",
                    "amount": 45230.50,
                    "oldbalanceOrg": 46100.00,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 120.00,
                    "newbalanceDest": 45350.50,
                    "dest_txn_history": 1,
                    "hour_of_day": 3,
                }
            ]
        }
    }


class TransactionResponse(BaseModel):
    fraud_score: float = Field(..., description="Model's predicted probability of fraud (0-1)")
    model_flag: bool = Field(..., description="True if fraud_score >= decision threshold (0.5)")
    static_rule_flag: bool = Field(
        ..., description="True if the legacy rule fires: TRANSFER/CASH_OUT with amount > $200,000"
    )
    flags_agree: bool = Field(..., description="True if model_flag and static_rule_flag agree")
    decision_threshold: float = Field(..., description="Probability threshold used for model_flag")

    # Several field names below start with "model_", which Pydantic v2 reserves
    # for its own internals by default. This just silences the harmless warning.
    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    model_type: str
    trained_at: str
    n_train: int
    n_test: int
    fraud_rate_train: float
    fraud_rate_test: float
    features: list[str]
    decision_threshold: float
    static_rule: dict
    model_metrics: dict
    model_vs_rule_agreement_rate: float
    feature_importances: dict

    model_config = {"protected_namespaces": ()}
