import math
from contextlib import nullcontext
import torch
import mlflow

def _safe_log_metrics(d: dict, step: int | None = None):
    clean = {}
    for k, v in d.items():
        try:
            if hasattr(v, "item"):
                v = v.item()
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                continue
            clean[k] = v
        except Exception:
            pass
    if clean:
        mlflow.log_metrics(clean, step=step)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    elif device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()
