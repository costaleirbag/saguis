#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
import mlflow

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .datasets import Args as DataArgs, build_dataloaders
from .models import build_model, build_optimizer, build_scheduler


# -------------------- utils --------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_ctx(device):
    # MPS não suporta autocast no PyTorch estável → desliga
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    elif device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    else:
        return nullcontext()


# -------------------- train / eval --------------------

def train_one_epoch(model, loader: DataLoader, optimizer, device, loss_fn, mixup_fn=None, ema: ModelEmaV2 | None = None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            imgs, targets = mixup_fn(imgs, targets)  # targets vira one-hot soft (N x C)

        with autocast_ctx(device):
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        running_loss += loss.item()

        # accuracy mesmo com mixup: usa argmax dos rótulos suaves
        preds = outputs.argmax(dim=1)
        if targets.ndim == 2:
            hard = targets.argmax(dim=1)
        else:
            hard = targets
        correct += (preds == hard).sum().item()
        total   += hard.size(0)

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


@torch.no_grad()
def predict_all_logits(model, loader, device):
    model.eval()
    ys, logits = [], []
    for x, y in loader:
        x = x.to(device)
        with autocast_ctx(device):
            logit = model(x)           # [N, C]
        ys.extend(y.cpu().tolist())
        logits.append(logit.cpu())
    if len(logits) == 0:
        return [], []
    logits = torch.cat(logits, dim=0)
    return ys, logits


# -------------------- main --------------------

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
    p.add_argument("--freeze_epochs", default=2, type=int, help="épocas de head-only antes de descongelar backbone")
    p.add_argument("--mixup", default=0.2, type=float)
    p.add_argument("--cutmix", default=0.0, type=float)
    p.add_argument("--smoothing", default=0.1, type=float)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--out_dir", default="outputs", type=str)
    p.add_argument("--resume", type=str, default="", help="checkpoint para retomar (ex.: outputs/.../last.pth)")
    p.add_argument("--auto_resume", action="store_true", help="se existir out_dir/last.pth, retoma automaticamente")
    p.add_argument("--pos_class", type=str, default="nao_hibrido",
                   help="nome da classe positiva para AUROC/threshold (ex.: 'hibrido' ou 'nao_hibrido')")
    args = p.parse_args()

    device = get_device()
    torch.set_float32_matmul_precision("high")  # melhora throughput no M1
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

    # índice da classe positiva para métricas probabilísticas
    if args.pos_class not in class_to_idx:
        # fallback: usa a maior classe (índice 1) — mas melhor avisar
        print(f"[warn] --pos_class '{args.pos_class}' não está em {list(class_to_idx.keys())}. "
              f"Usando a última classe por padrão.")
        pos_idx = max(class_to_idx.values())
    else:
        pos_idx = class_to_idx[args.pos_class]

    # Model
    model = build_model(args.model, num_classes=num_classes).to(device)

    # Congela backbone inicialmente (head-only)
    head_names = ("head", "fc", "classifier", "cls")
    base_params = [p for n, p in model.named_parameters() if not any(h in n for h in head_names)]
    for p in base_params:
        p.requires_grad_(False)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs, num_steps_per_epoch=len(train_dl)
    )

    # Loss / Mixup
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
    last_path = Path(args.out_dir) / "last.pth"

    # -------- Resume (opcional) --------
    start_epoch = 0
    ckpt_path = Path(args.resume) if args.resume else None
    if args.auto_resume and last_path.exists() and not ckpt_path:
        ckpt_path = last_path
    if ckpt_path and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "ema_state" in ckpt:
            ema.module.load_state_dict(ckpt["ema_state"])
        best_acc = ckpt.get("best_acc", 0.0)
        start_epoch = ckpt.get("epoch", -1) + 1
        # Em caso de retomar depois do ponto de unfreeze
        if start_epoch >= args.freeze_epochs:
            for p in base_params:
                p.requires_grad_(True)
    else:
        print("Fresh run (no resume).")

    # ===== MLflow =====
    mlflow.set_experiment("saguis")
    run_name = f"{args.model}-{Path(args.data_dir).name}"
    with mlflow.start_run(run_name=run_name):
        # params & tags
        mlflow.set_tag("device", device.type)
        mlflow.set_tag("num_classes", num_classes)
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
            "pos_class": args.pos_class,
        })
        if ckpt_path:
            mlflow.set_tag("resume_from", str(ckpt_path))

        try:
            # -------- training loop --------
            for epoch in range(start_epoch, args.epochs):
                if epoch == args.freeze_epochs:
                    for p in base_params:
                        p.requires_grad_(True)
                    print("Unfroze backbone for full fine-tuning.")

                print(f"\nEpoch {epoch+1}/{args.epochs}")
                tr_loss, tr_acc = train_one_epoch(
                    model, train_dl, optimizer, device, loss_fn, mixup_fn=mixup_fn, ema=ema
                )
                scheduler.step()

                va_loss, va_acc = evaluate(model, val_dl, device)
                ema_va_loss, ema_va_acc = evaluate(ema.module, val_dl, device)

                # métricas macro (EMA)
                y_true, logits = predict_all_logits(ema.module, val_dl, device)
                if len(logits) > 0:
                    probs = torch.softmax(logits, dim=1)[:, pos_idx].numpy().tolist()
                    y_pred = [1 if p >= 0.5 else 0 for p in probs]
                    f1_macro = f1_score(y_true, y_pred, average="macro")
                    try:
                        auroc = roc_auc_score(y_true, probs)
                    except ValueError:
                        auroc = float("nan")

                    # threshold sweep (salvar como métrica por época)
                    ths = np.linspace(0.05, 0.95, 19)
                    best_f1, best_th = 0.0, 0.5
                    for th in ths:
                        yp = [1 if p >= th else 0 for p in probs]
                        f1 = f1_score(y_true, yp, average="macro")
                        if f1 > best_f1:
                            best_f1, best_th = f1, th
                    mlflow.log_metric("best_threshold", float(best_th), step=epoch + 1)
                    mlflow.log_metrics({"ema_f1_macro": f1_macro, "ema_auroc": auroc}, step=epoch + 1)
                else:
                    best_th = 0.5
                    f1_macro = float("nan")
                    auroc = float("nan")

                current_lr = optimizer.param_groups[0]["lr"]
                mlflow.log_metrics({
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "ema_val_loss": ema_va_loss,
                    "ema_val_acc": ema_va_acc,
                    "lr": current_lr,
                }, step=epoch + 1)

                print(
                    f"Train loss {tr_loss:.4f} | Train acc {tr_acc:.2f}% | "
                    f"Val loss {va_loss:.4f} | Val acc {va_acc:.2f}% | "
                    f"EMA Val loss {ema_va_loss:.4f} | EMA Val acc {ema_va_acc:.2f}% | "
                    f"F1(macro) {f1_macro:.3f} | AUROC {auroc:.3f} | LR {current_lr:.6f} | "
                    f"best_th {best_th:.2f}"
                )

                # ----- save last checkpoint (sempre) -----
                torch.save({
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "ema_state": ema.module.state_dict(),
                    "class_to_idx": class_to_idx,
                    "model_name": args.model,
                    "img_size": args.img_size,
                }, last_path)

                # ----- save best (ema) -----
                monitor_acc = ema_va_acc
                monitor_model = ema.module
                if monitor_acc > best_acc:
                    best_acc = monitor_acc
                    torch.save({
                        "model_state": monitor_model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "model_name": args.model,
                        "img_size": args.img_size,
                    }, best_path)
                    with open(Path(args.out_dir) / "class_to_idx.json", "w") as f:
                        json.dump(class_to_idx, f, indent=2)
                    print(f"Saved new best: {best_path} (Val acc = {best_acc:.2f}%).")

        except KeyboardInterrupt:
            print("\n[interrupt] salvando last.pth antes de sair…")
            torch.save({
                "epoch": epoch if 'epoch' in locals() else -1,
                "best_acc": best_acc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_state": ema.module.state_dict(),
                "class_to_idx": class_to_idx,
                "model_name": args.model,
                "img_size": args.img_size,
            }, last_path)
            raise

        # artefatos
        mlflow.log_artifact(str(best_path))
        if (Path(args.out_dir) / "class_to_idx.json").exists():
            mlflow.log_artifact(str(Path(args.out_dir) / "class_to_idx.json"))
        if last_path.exists():
            mlflow.log_artifact(str(last_path))

    print("Done.")
    print(f"Best Val Acc: {best_acc:.2f}% | Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
