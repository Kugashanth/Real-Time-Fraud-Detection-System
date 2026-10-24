import json
import time
from pathlib import Path

LOG_FILE = Path("artifacts/live_predictions_log.jsonl")


def log_prediction_event(
    transaction_id: str,
    amount: float,
    features: list[float],
    fraud_probability: float,
    is_fraud: bool,
    risk_tier: str,
    latency_ms: float,
) -> None:
    """Non-blocking background logger writing JSONL records for Member 5's drift pipeline."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "transaction_id": transaction_id,
        "Amount": amount,
        "V14": features[13] if len(features) >= 14 else 0.0,
        "V17": features[16] if len(features) >= 17 else 0.0,
        "fraud_probability": fraud_probability,
        "is_fraud": int(is_fraud),
        "risk_tier": risk_tier,
        "latency_ms": latency_ms,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
