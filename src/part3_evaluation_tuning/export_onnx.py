from pathlib import Path
import numpy as np
from skl2onnx import to_onnx


def export_sklearn_model_to_onnx(model, n_features: int = 30, output_path: str | Path = "artifacts/champion_model.onnx"):
    """Compiles trained Scikit-Learn / XGBoost estimator into an ONNX graph for sub-10ms CPU execution."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_input = np.zeros((1, n_features), dtype=np.float32)
    onnx_model = to_onnx(model, sample_input, options={id(model): {"zipmap": False}})
    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return out
