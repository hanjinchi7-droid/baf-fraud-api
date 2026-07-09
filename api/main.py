"""Fraud scoring API."""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).with_name("model_bundle.joblib")
BUNDLE = joblib.load(MODEL_PATH)
EXPLAINER = shap.TreeExplainer(BUNDLE["model"])
THRESHOLD = float(BUNDLE["threshold"])


class Application(BaseModel):
    model_config = ConfigDict(extra="forbid")

    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: int = Field(ge=0, le=1)
    housing_status: str
    phone_home_valid: int = Field(ge=0, le=1)
    phone_mobile_valid: int = Field(ge=0, le=1)
    bank_months_count: int
    has_other_cards: int = Field(ge=0, le=1)
    proposed_credit_limit: float
    foreign_request: int = Field(ge=0, le=1)
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int = Field(ge=0, le=1)
    device_distinct_emails_8w: int
    device_fraud_count: int


app = FastAPI(
    title="BAF Agentic Fraud-Triage API",
    version="1.0.0",
    description="Scores an application and returns SHAP risk factors.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "XGBoost cost-sensitive",
        "operating_threshold": THRESHOLD,
    }


def explain_one(transformed: np.ndarray, top_n: int = 5) -> list[dict[str, Any]]:
    values = np.asarray(EXPLAINER.shap_values(transformed))[0]
    names = np.asarray(BUNDLE["transformed_feature_names"])
    feature_values = transformed[0]
    # features pushing risk up; if none, take the biggest by magnitude
    candidate = np.flatnonzero(values > 0)
    if not len(candidate):
        candidate = np.arange(len(values))
    ranked = candidate[np.argsort(np.abs(values[candidate]))[::-1][:top_n]]
    return [
        {
            "feature": str(names[i]),
            "shap_value": round(float(values[i]), 6),
            "transformed_value": round(float(feature_values[i]), 6),
            "direction": "increases_risk" if values[i] > 0 else "decreases_risk",
        }
        for i in ranked
    ]


@app.post("/predict")
def predict(application: Application) -> dict[str, Any]:
    frame = pd.DataFrame([application.model_dump()])[BUNDLE["feature_columns"]]
    transformed = BUNDLE["preprocessor"].transform(frame).astype("float32")
    probability = float(BUNDLE["model"].predict_proba(transformed)[0, 1])
    if probability >= THRESHOLD:
        prediction, level, action = 1, "high", "block_and_escalate"
    elif probability >= THRESHOLD * 0.5:
        prediction, level, action = 0, "medium", "manual_review"
    else:
        prediction, level, action = 0, "low", "allow"
    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "risk_level": level,
        "deterministic_action": action,
        "operating_threshold": round(THRESHOLD, 6),
        "top_risk_factors": explain_one(transformed),
        "explanation_method": "per-instance TreeSHAP",
    }
