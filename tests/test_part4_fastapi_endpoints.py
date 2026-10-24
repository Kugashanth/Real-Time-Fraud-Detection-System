from fastapi.testclient import TestClient
from src.part4_api_serving.main import app

client = TestClient(app)


def test_health_and_realtime_predict_endpoint():
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    tx_payload = {
        "transaction_id": "tx_test_001",
        "card_id": "card_42",
        "Time": 1250.0,
        "Amount": 420.50,
        "features": [-1.2, 0.4, -0.8, 1.1, 0.0, -0.3, 0.2, 0.1, -0.5, -2.1,
                     0.3, -2.8, 0.1, -4.5, 0.0, -1.1, -3.9, 0.2, 0.4, 0.1,
                     0.0, 0.1, -0.1, 0.2, 0.0, 0.1, 0.05, -0.02],
    }
    pred_resp = client.post("/v1/predict", json=tx_payload)
    assert pred_resp.status_code == 200
    body = pred_resp.json()
    assert body["transaction_id"] == "tx_test_001"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_tier"] in ("LOW_ALLOW", "MEDIUM_REVIEW", "HIGH_BLOCK")
    assert len(body["top_reason_codes"]) == 3
