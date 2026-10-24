# Real-Time Fraud Detection System (5-Member End-to-End Repository)

Production-grade Real-Time Credit Card Fraud Detection System comparing **Logistic Regression**, **Random Forest**, and **XGBoost / LightGBM** on highly imbalanced transaction data (~0.17% fraud rate) using **leakage-free SMOTE-ENN**, **Optuna Bayesian tuning**, **Isotonic Probability Calibration**, **Cost-Sensitive Threshold Optimization**, **MAPIE Conformal Prediction**, **SHAP Explainability**, **ONNX Runtime + FastAPI + Redis Serving (<50ms P99)**, and **Evidently AI + NannyML + Prometheus/Grafana Monitoring**.

## 5-Member Team Ownership & Modules
| Member | Package Directory | Core Focus | Key Handoff Artifact |
| :--- | :--- | :--- | :--- |
| **Member 1** | `src/part1_data_engineering/` | Polars ETL, Velocity Features, Pandera Validation, Stratified Split, SMOTE-ENN Pipeline | `data/processed/*.parquet`, `artifacts/preprocessor.joblib` |
| **Member 2** | `src/part2_model_training/` | Logistic Regression, Balanced RF, XGBoost/LightGBM, Optuna Tuning, MLflow Tracking | `artifacts/model_comparison.csv`, MLflow Registry |
| **Member 3** | `src/part3_evaluation_tuning/` | PR-AUC/F2/MCC Evaluation, Isotonic Calibration, Cost Thresholding, MAPIE, SHAP, ONNX Export | `artifacts/champion_model.onnx`, `artifacts/threshold_config.json`, `artifacts/reference_baseline.parquet` |
| **Member 4** | `src/part4_api_serving/` | FastAPI (`/v1/predict`), Async Redis Velocity Store, ONNX Runtime, Docker & Locust Load Testing | Containerized REST API (`<50ms P99`) |
| **Member 5** | `src/part5_monitoring_mlops/` | Prometheus Telemetry, Evidently AI Drift (PSI/K-S), NannyML CBPE, `/v1/feedback` & Auto-Retrain | `observability/grafana_dashboard.json`, Drift Reports |

## Quickstart Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Member 1: Data Engineering & Stratified Split
make data

# 3. Run Member 2: Model Training & Optuna Hyperparameter Search
make train

# 4. Run Member 3: Calibration, Cost Threshold Tuning, SHAP & ONNX Export
make tune

# 5. Run Member 4: Launch FastAPI + Redis + Prometheus + Grafana stack
make serve

# 6. Run Member 5: Execute Evidently Drift & NannyML CBPE Monitoring Job
make monitor

# 7. Run Automated Test Suite (All 5 Parts)
make test
```
