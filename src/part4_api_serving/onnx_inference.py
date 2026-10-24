import json
from pathlib import Path
import numpy as np


class FraudScoringEngine:
    """Loads ONNX Runtime session (or falls back to lightweight deterministic scorer in CI) and applies tuned thresholds."""
    def __init__(
        self,
        onnx_path: str | Path = "artifacts/champion_model.onnx",
        threshold_config_path: str | Path = "artifacts/threshold_config.json",
    ):
        self.onnx_path = Path(onnx_path)
        self.session = None
        if self.onnx_path.exists():
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])

        cfg_path = Path(threshold_config_path)
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.config = json.load(f)
        else:
            self.config = {"optimal_cost_threshold": 0.315, "low_risk_max": 0.12}

    def score_vector(self, vector_30d: np.ndarray) -> float:
        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: vector_30d.astype(np.float32)})
            probs = outputs[1]
            return float(probs[0][1])

        # Deterministic fallback scoring using V14, V17, V12, and Amount when ONNX file isn't built yet
        v14 = float(vector_30d[0, 15])
        v17 = float(vector_30d[0, 18])
        amt = float(vector_30d[0, 1])
        logit = -3.8 - 0.85 * v14 - 0.65 * v17 + 0.002 * amt
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20))))
