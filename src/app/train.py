#!/usr/bin/env python3
"""
Fine-tune (binário: hibrido vs nao_hibrido) com timm, rodando em MPS no Mac M1 quando disponível.

Usa:
- datasets.build_dataloaders  (split automático se não houver val/)
- models.build_model / build_optimizer / build_scheduler
- Mixup + LabelSmoothing/SoftTargetCE
- Cosine LR + warmup
- Salva best_model.pth + class_to_idx.json

Exemplo:
  poetry run saguis-train \
    --data_dir data/raw \
    --model convnext_tiny \
    --epochs 5 --batch_size 16 \
    --img_size 224 --mixup 0.1 \
    --lr 5e-4 --weight_decay 0.05 \
    --freeze_epochs 2 \
    --out_dir outputs/aurita_test
"""

import os
import json
import math
import argparse
from pathlib import Path
import mlflow
from contextlib import nullcontext



from zmq import device

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2

from .datasets import Args as DataArgs, build_dataloaders
from .models import build_model, build_optimizer, build_scheduler

from collections import Counter

def count_samples(ds):
    try:
        samples = ds.samples
    except AttributeError:
        base, idxs = ds.dataset, ds.indices  # Subset
        samples = [base.samples[i] for i in idxs]
    return Counter([y for _, y in samples])

def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    elif device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    else:
        return nullcontext()

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader: DataLoader, optimizer, device, loss_fn, mixup_fn=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            imgs, targets = mixup_fn(imgs, targets)

        with autocast_ctx(device):
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if mixup_fn is None:
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / max(1, len(loader))
    acc = (correct / total * 100.0) if total > 0 else float("nan")
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader: DataLoader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        with autocast_ctx(device):
            outputs = model(imgs)
            loss = ce(outputs, targets)
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(1, len(loader))
    acc = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--model", default="convnext_tiny", type=str)
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--warmup_epochs", default=3, type=int)
    p.add_argument("--freeze_epochs", default=2, type=int)
    p.add_argument("--mixup", default=0.2, type=float)
    p.add_argument("--cutmix", default=0.0, type=float)
    p.add_argument("--smoothing", default=0.1, type=float)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--out_dir", default="outputs", type=str)
    args = p.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # Data
    dl_args = DataArgs(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_dl, val_dl, class_to_idx = build_dataloaders(dl_args)
    num_classes = len(class_to_idx)
    print(f"Classes: {class_to_idx} (num_classes={num_classes})")

    idx_to_class = {v:k for k,v in class_to_idx.items()}
    tr_counts = count_samples(train_dl.dataset)
    va_counts = count_samples(val_dl.dataset)
    print("Train counts:", {idx_to_class[i]: tr_counts.get(i,0) for i in idx_to_class})
    print("Val counts:",   {idx_to_class[i]: va_counts.get(i,0) for i in idx_to_class})


    # Model
    model = build_model(args.model, num_classes=num_classes).to(device)

    # Congela backbone inicialmente (head-only)
    head_names = ("head", "fc", "classifier", "cls")
    base_params = []
    for n, p in model.named_parameters():
        if not any(h in n for h in head_names):
            base_params.append(p)
    for p in base_params:
        p.requires_grad_(False)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        num_steps_per_epoch=len(train_dl),
    )

    # Loss/Mixup
    mixup_active = (args.mixup > 0.0) or (args.cutmix > 0.0)
    mixup_fn = Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        label_smoothing=args.smoothing,
        num_classes=num_classes,
    ) if mixup_active else None
    loss_fn = SoftTargetCrossEntropy() if mixup_active else LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    ema = ModelEmaV2(model, decay=0.9999)

    best_acc = 0.0
    best_path = Path(args.out_dir) / "best_model.pth"

    # ===== MLflow =====
    mlflow.set_experiment("saguis")
    run_name = f"{args.model}-{Path(args.data_dir).name}"

    with mlflow.start_run(run_name=run_name):
        # params & tags
        mlflow.set_tag("device", device.type)
        mlflow.set_tag("num_classes", len(class_to_idx))
        mlflow.log_params({
            "model": args.model,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "freeze_epochs": args.freeze_epochs,
            "mixup": args.mixup,
            "cutmix": args.cutmix,
            "smoothing": args.smoothing,
            "num_workers": args.num_workers,
            "data_dir": args.data_dir,
        })

        # loop de treino
        for epoch in range(args.epochs):
            if epoch == args.freeze_epochs:
                for p in base_params:
                    p.requires_grad_(True)
                print("Unfroze backbone for full fine-tuning.")

            print(f"\nEpoch {epoch+1}/{args.epochs}")
            tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, device, loss_fn, mixup_fn=mixup_fn)
            scheduler.step()
            va_loss, va_acc = evaluate(model, val_dl, device)

            ema.update(model)
            ema_va_loss, ema_va_acc = evaluate(ema.module, val_dl, device)

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "ema_val_acc": ema_va_acc,
                "lr": current_lr,
            }, step=epoch + 1)

            print(
                f"Train loss {tr_loss:.4f} | Train acc {tr_acc:.2f}% | "
                f"Val loss {va_loss:.4f} | Val acc {va_acc:.2f}% | "
                f"EMA Val acc {ema_va_acc:.2f}% | LR {current_lr:.6f}"
            )

            monitor_acc = ema_va_acc
            monitor_model = ema.module

            if monitor_acc > best_acc:
                best_acc = monitor_acc
                torch.save(
                    {
                        "model_state": monitor_model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "model_name": args.model,
                        "img_size": args.img_size,
                    },
                    best_path,
                )
                with open(Path(args.out_dir) / "class_to_idx.json", "w") as f:
                    json.dump(class_to_idx, f, indent=2)
                print(f"Saved new best: {best_path} (Val acc = {best_acc:.2f}%).")

        # artefatos
        mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(Path(args.out_dir) / "class_to_idx.json"))

    print("Done.")
    print(f"Best Val Acc: {best_acc:.2f}% | Checkpoint: {best_path}")



if __name__ == "__main__":
    main()
