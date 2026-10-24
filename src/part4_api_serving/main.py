import time
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from src.part4_api_serving.schemas import (
    TransactionPayload,
    BatchTransactionPayload,
    PredictionResponse,
)
from src.part4_api_serving.onnx_inference import FraudScoringEngine
from src.part4_api_serving.redis_feature_store import InMemoryVelocityFallback
from src.part4_api_serving.background_logger import log_prediction_event
from src.part5_monitoring_mlops.feedback_router import router as feedback_router

app = FastAPI(
    title="Real-Time Fraud Detection Service",
    version="1.0.0",
    description="Sub-50ms ONNX Runtime Fraud Scoring API with Conformal Uncertainty & Drift Telemetry",
)
app.include_router(feedback_router)

engine = FraudScoringEngine()
velocity_store = InMemoryVelocityFallback()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "onnx_loaded": engine.session is not None,
        "optimal_cost_threshold": engine.config.get("optimal_cost_threshold", 0.315),
    }


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_single(payload: TransactionPayload, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()
    velocity = velocity_store.record_and_fetch_velocity(payload.card_id, payload.Amount)
    vec = np.array([[payload.Time, payload.Amount] + payload.features], dtype=np.float32)

    prob = engine.score_vector(vec)
    thresh = float(engine.config.get("optimal_cost_threshold", 0.315))
    low_max = float(engine.config.get("low_risk_max", 0.12))

    is_fraud = prob >= thresh
    manual_review = abs(prob - thresh) <= 0.08
    if is_fraud:
        risk_tier = "HIGH_BLOCK"
    elif prob >= low_max or manual_review:
        risk_tier = "MEDIUM_REVIEW"
    else:
        risk_tier = "LOW_ALLOW"

    reasons = [
        f"V14 ({payload.features[13]:+.2f})",
        f"tx_count_1h ({int(velocity['tx_count_1h'])})",
        f"amt_ratio ({velocity['amount_to_mean_ratio']:.2f}x)",
    ]
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    background_tasks.add_task(
        log_prediction_event,
        payload.transaction_id,
        payload.Amount,
        payload.features,
        prob,
        is_fraud,
        risk_tier,
        latency_ms,
    )

    return PredictionResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=round(prob, 5),
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        manual_review_required=manual_review,
        top_reason_codes=reasons,
        latency_ms=latency_ms,
    )


@app.post("/v1/predict_batch", response_model=list[PredictionResponse])
async def predict_batch(batch: BatchTransactionPayload, background_tasks: BackgroundTasks):
    results = []
    for tx in batch.transactions:
        res = await predict_single(tx, background_tasks)
        results.append(res)
    return results
