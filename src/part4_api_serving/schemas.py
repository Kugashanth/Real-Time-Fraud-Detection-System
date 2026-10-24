from pydantic import BaseModel, Field


class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction UUID")
    card_id: str = Field(default="card_default", description="Cardholder identifier for Redis velocity lookup")
    Time: float = Field(..., ge=0.0, description="Seconds elapsed since reference epoch")
    Amount: float = Field(..., ge=0.0, description="Transaction monetary value in USD")
    features: list[float] = Field(..., min_length=28, max_length=28, description="PCA anonymized features V1..V28")


class BatchTransactionPayload(BaseModel):
    transactions: list[TransactionPayload] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    manual_review_required: bool
    top_reason_codes: list[str]
    latency_ms: float
