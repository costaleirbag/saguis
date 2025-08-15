# --- Inference utils para Notebook (PyTorch + timm) ---

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import timm
from timm.data import create_transform

from PIL import Image
from io import BytesIO
import requests
import numpy as np
import matplotlib.pyplot as plt


# ---------- Dispositivo ----------
def resolve_device(pref: Optional[str] = None) -> torch.device:
    if pref:
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- Checkpoint helpers ----------
def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def _extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    # prioridade: ema > model_state > state_dict > model, e casos "direto"
    for key in ["ema", "model_ema", "model_state", "state_dict", "model"]:
        obj = ckpt.get(key)
        if obj is None: 
            continue
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return _strip_module_prefix(obj["state_dict"])
            else:
                return _strip_module_prefix(obj)
    # pode ser um state_dict direto
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        return _strip_module_prefix(ckpt)
    raise RuntimeError("state_dict não encontrado no checkpoint.")


# ---------- Carregar modelo ----------
def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str = "convnext_tiny",
    num_classes: int = 2,
    device: Optional[torch.device] = None,
):
    device = device or resolve_device()
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state_dict = _extract_state_dict(ckpt)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("[Aviso] Chaves faltantes:", missing)
        print("[Aviso] Chaves inesperadas:", unexpected)

    model.to(device).eval()

    class_to_idx = ckpt.get("class_to_idx", None)
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else None

    return model, device, idx_to_class


# ---------- Transform ----------
def build_transform(img_size: int = 224):
    # coerente com inferência (sem aug; normalização ImageNet)
    return create_transform(input_size=img_size, is_training=False)


# ---------- IO de imagem ----------
def load_image_from_path(img_path: str | Path) -> Image.Image:
    return Image.open(img_path).convert("RGB")

def load_image_from_url(url: str, timeout: int = 15) -> Image.Image:
    # precisa de internet habilitada no ambiente do notebook
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


# ---------- Predição ----------
@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    img: Image.Image,
    transform,
    device: torch.device,
    idx_to_class: Optional[Dict[int, str]] = None
) -> tuple[str, int, np.ndarray]:
    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_label = idx_to_class.get(pred_idx, str(pred_idx)) if idx_to_class else str(pred_idx)
    return pred_label, pred_idx, probs


# ---------- Helpers de notebook (plot + retorno) ----------
def infer_and_show_from_path(
    model, device, idx_to_class, img_path: str | Path, img_size: int = 224
):
    transform = build_transform(img_size)
    img = load_image_from_path(img_path)
    pred_label, pred_idx, probs = predict_image(model, img, transform, device, idx_to_class)

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    title = f"Pred: {pred_label} (idx={pred_idx}) | p={probs[pred_idx]:.3f}"
    plt.title(title)
    plt.show()

    return pred_label, pred_idx, probs

def infer_and_show_from_url(
    model, device, idx_to_class, url: str, img_size: int = 224
):
    transform = build_transform(img_size)
    img = load_image_from_url(url)
    pred_label, pred_idx, probs = predict_image(model, img, transform, device, idx_to_class)

    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    title = f"Pred: {pred_label} (idx={pred_idx}) | p={probs[pred_idx]:.3f}"
    plt.title(title)
    plt.show()

    return pred_label, pred_idx, probs
