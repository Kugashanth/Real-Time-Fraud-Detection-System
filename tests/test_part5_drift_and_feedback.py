import numpy as np
from fastapi.testclient import TestClient
from src.part4_api_serving.main import app
from src.part5_monitoring_mlops.evidently_drift_job import evaluate_feature_drift
from src.part5_monitoring_mlops.nannyml_cbpe_job import estimate_cbpe_expected_f1
from src.part5_monitoring_mlops.retrain_trigger import should_trigger_retraining

client = TestClient(app)


def test_drift_cbpe_and_feedback_loop():
    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 1.0, 1000)
    cur = rng.normal(0.8, 1.3, 1000)

    drift_res = evaluate_feature_drift(ref, cur)
    assert drift_res["drift_detected"] is True
    assert drift_res["psi"] > 0.25

    probs = np.concatenate([rng.uniform(0.01, 0.1, 900), rng.uniform(0.7, 0.99, 100)])
    est_f1 = estimate_cbpe_expected_f1(probs, threshold=0.315)
    assert 0.5 <= est_f1 <= 1.0

    trigger_res = should_trigger_retraining({"V14": 0.32, "Amount": 0.29}, estimated_pr_auc_drop=0.06)
    assert trigger_res["retrain_triggered"] is True

    fb_resp = client.post("/v1/feedback", json={"transaction_id": "tx_test_001", "true_class": 1, "analyst_notes": "Confirmed stolen card"})
    assert fb_resp.status_code == 200
    assert fb_resp.json()["status"] == "recorded"
