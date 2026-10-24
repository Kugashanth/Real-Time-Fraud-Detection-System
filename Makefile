.PHONY: install data train tune serve monitor test loadtest

install:
	pip install -r requirements.txt

data:
	python3 -m src.part1_data_engineering.split_and_scale

train:
	python3 -m src.part2_model_training.optuna_tuning

tune:
	python3 -m src.part3_evaluation_tuning.threshold_optimizer
	python3 -m src.part3_evaluation_tuning.export_onnx

serve:
	uvicorn src.part4_api_serving.main:app --host 0.0.0.0 --port 8000 --workers 2

monitor:
	python3 -m src.part5_monitoring_mlops.evidently_drift_job

test:
	pytest tests/ -v

loadtest:
	locust -f src/part4_api_serving/locustfile.py --headless -u 200 -r 50 --run-time 1m --host http://localhost:8000
