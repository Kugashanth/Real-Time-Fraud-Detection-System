import json
import time
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["Monitoring & Feedback (Member 5)"])
FEEDBACK_FILE = Path("data/feedback_store/verified_chargebacks.jsonl")


class ChargebackFeedbackPayload(BaseModel):
    transaction_id: str
    true_class: int = Field(..., ge=0, le=1, description="Verified ground truth: 1=Confirmed Chargeback, 0=Legitimate")
    analyst_notes: str = Field(default="")


@router.post("/v1/feedback")
async def ingest_ground_truth_feedback(feedback: ChargebackFeedbackPayload):
    """Ingests delayed ground-truth chargebacks to compute realized production PR-AUC and trigger retraining."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "received_at": time.time(),
        "transaction_id": feedback.transaction_id,
        "true_class": feedback.true_class,
        "analyst_notes": feedback.analyst_notes,
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "recorded", "transaction_id": feedback.transaction_id}
