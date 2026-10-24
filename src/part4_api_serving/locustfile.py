import random
from locust import HttpUser, task, between


class PaymentGatewayUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(10)
    def score_transaction(self):
        payload = {
            "transaction_id": f"tx_{random.randint(100000, 999999)}",
            "card_id": f"card_{random.randint(1, 500)}",
            "Time": random.uniform(100.0, 170000.0),
            "Amount": random.expovariate(1.0 / 75.0),
            "features": [random.gauss(0.0, 1.0) for _ in range(28)],
        }
        self.client.post("/v1/predict", json=payload)
