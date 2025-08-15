#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import json
import torch
import pandas as pd
import numpy as np

from .datasets import build_transforms, Args as DataArgs, PathImageFolder
from .models import build_model

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def _forward_features_generic(model, x):
    """
    Tenta extrair o penúltimo nível de representação de modelos timm.
    - Se existir model.forward_features(x), usa e faz global avg pool se vier [N,C,H,W].
    - Caso contrário, cai para 'semifeature': pega logits (menos ideal).
    """
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        if feats.ndim == 4:  # [N, C, H, W] -> GAP
            feats = feats.mean(dim=(2, 3))
        return feats
    else:
        # fallback: usa logits (não ideal), ainda assim funciona para o XGB
        return model(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str, help="Root com pastas de classe (ou train/val)")
    ap.add_argument("--model", default="convnext_tiny", type=str)
    ap.add_argument("--ckpt", required=True, type=str, help="Checkpoint (.pth) — use o best (EMA)")
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--split", default="all", choices=["all","train","val"], help="Qual split extrair")
    ap.add_argument("--out", default="embeddings.parquet", type=str)
    args = ap.parse_args()

    device = get_device()
    print("Device:", device)

    # Carrega checkpoint (o seu best salva apenas o estado do EMA)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)
    model_name = ckpt.get("model_name", args.model)
    print("Loaded ckpt with classes:", class_to_idx)

    model = build_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Transforms de VAL (determinístico)
    _, val_tf = build_transforms(args.img_size, aug_backend="alb", aug_preset="light")
    # Use o mesmo backend/preset do seu val; pode ajustar se quiser:
    # _, val_tf = build_transforms(args.img_size, aug_backend="alb", aug_preset="none")

    # Descobre root real do split
    root = Path(args.data_dir)
    if (root / "train").exists() or (root / "val").exists():
        if args.split == "train":
            data_root = root / "train"
        elif args.split == "val":
            data_root = root / "val"
        else:
            # all = concatenar train + val
            pass
    else:
        data_root = root

    def build_loader(dirpath):
        ds = PathImageFolder(dirpath, transform=val_tf)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=False
        )
        return ds, dl

    records = []

    def run_one(dirpath, split_name):
        ds, dl = build_loader(dirpath)
        print(f"[{split_name}] {len(ds)} imagens em {dirpath}")
        for (imgs, ys, paths) in dl:
            imgs = imgs.to(device)
            feats = _forward_features_generic(model, imgs)  # [N, D]
            feats = feats.detach().cpu().numpy()
            ys = ys.numpy()
            for p, y, f in zip(paths, ys, feats):
                base = Path(p).name
                stem = Path(p).stem
                rec = {
                    "path": str(p),
                    "basename": base,
                    "stem": stem,
                    "y": int(y),
                    "split": split_name,
                }
                for i, val in enumerate(f.tolist()):
                    rec[f"f_{i}"] = val
                records.append(rec)

    if args.split == "all":
        if (root / "train").exists():
            run_one(root / "train", "train")
        if (root / "val").exists():
            run_one(root / "val", "val")
        if not (root / "train").exists() and not (root / "val").exists():
            run_one(root, "all")
    else:
        run_one(data_root, args.split)

    df = pd.DataFrame.from_records(records)
    # salva também o mapeamento de classes para referência
    df.attrs["class_to_idx"] = class_to_idx
    df.attrs["model_name"] = model_name
    df.to_parquet(args.out, index=False)
    meta_path = Path(args.out).with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "model_name": model_name}, f, ensure_ascii=False, indent=2)
    print("Saved:", args.out, "and", meta_path)

if __name__ == "__main__":
    main()
