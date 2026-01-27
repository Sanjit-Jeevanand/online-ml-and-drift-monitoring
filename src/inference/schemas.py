from typing import Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# Request schema
# ============================================================

class PredictRequest(BaseModel):

    features: Dict[str, Any] = Field(
        ...,
        description="Raw feature values keyed by feature name.",
        example={
            "LIMIT_BAL": 30000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 36,
            "PAY_0": 0,
            "PAY_2": 0,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 3913,
            "BILL_AMT2": 3102,
            "BILL_AMT3": 689,
            "BILL_AMT4": 0,
            "BILL_AMT5": 0,
            "BILL_AMT6": 0,
            "PAY_AMT1": 0,
            "PAY_AMT2": 689,
            "PAY_AMT3": 0,
            "PAY_AMT4": 0,
            "PAY_AMT5": 0,
            "PAY_AMT6": 0
        },
    )


# ============================================================
# Response schema
# ============================================================

class PredictResponse(BaseModel):

    model_name: str = Field(
        ...,
        description="Name of the model used for inference."
    )

    model_version: str = Field(
        ...,
        description="Version of the model used for inference."
    )

    predicted_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of default."
    )