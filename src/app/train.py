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

def compute_probs_and_metrics(logits, y_true, pos_idx):
    """Retorna (probs_pos, f1_macro, auroc, best_th, flipped) escolhendo a melhor orientação."""
    if len(logits) == 0:
        return [], float("nan"), float("nan"), 0.5, False

    probs_pos = torch.softmax(logits, dim=1)[:, pos_idx].numpy().tolist()
    # Métricas na orientação direta
    y_pred = [1 if p >= 0.5 else 0 for p in probs_pos]
    f1_dir = f1_score(y_true, y_pred, average="macro")
    try:
        auroc_dir = roc_auc_score(y_true, probs_pos)
    except ValueError:
        auroc_dir = float("nan")

    # Métricas na orientação invertida (caso classe positiva tenha sido escolhida “ao contrário”)
    probs_neg = (1.0 - np.array(probs_pos)).tolist()
    y_pred_neg = [1 if p >= 0.5 else 0 for p in probs_neg]
    f1_inv = f1_score(y_true, y_pred_neg, average="macro")
    try:
        auroc_inv = roc_auc_score(y_true, probs_neg)
    except ValueError:
        auroc_inv = float("nan")

    # Escolhe a melhor orientação pelo AUROC (mais estável)
    flipped = False
    f1_best, auroc_best, probs_best = f1_dir, auroc_dir, probs_pos
    if (not np.isnan(auroc_inv)) and (np.isnan(auroc_dir) or auroc_inv > auroc_dir):
        f1_best, auroc_best, probs_best = f1_inv, auroc_inv, probs_neg
        flipped = True

    # Threshold sweep para F1 macro
    ths = np.linspace(0.05, 0.95, 19)
    best_f1, best_th = 0.0, 0.5
    for th in ths:
        yp = [1 if p >= th else 0 for p in probs_best]
        f1 = f1_score(y_true, yp, average="macro")
        if f1 > best_f1:
            best_f1, best_th = f1, th

    return probs_best, float(f1_best), float(auroc_best), float(best_th), flipped


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
    p.add_argument("--ema_decay", type=float, default=0.99,
                   help="EMA decay (use menor p/ datasets e treinos pequenos; ex.: 0.99)")
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

    # EMA com decay configurável
    ema = ModelEmaV2(model, decay=args.ema_decay)

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
            "ema_decay": args.ema_decay,
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

                # Avaliações
                va_loss, va_acc = evaluate(model, val_dl, device)
                ema_va_loss, ema_va_acc = evaluate(ema.module, val_dl, device)

                # ----- Métricas probabilísticas (BASE e EMA), com correção de orientação -----
                y_true_base, logits_base = predict_all_logits(model, val_dl, device)
                probs_b, f1_b, auroc_b, best_th_b, flipped_b = compute_probs_and_metrics(
                    logits_base, y_true_base, pos_idx
                )

                y_true_ema, logits_ema = predict_all_logits(ema.module, val_dl, device)
                probs_e, f1_e, auroc_e, best_th_e, flipped_e = compute_probs_and_metrics(
                    logits_ema, y_true_ema, pos_idx
                )

                # Gap médio entre parâmetros (modelo vs EMA) — sanidade do EMA
                with torch.no_grad():
                    diffs = []
                    for p_w, q_w in zip(model.parameters(), ema.module.parameters()):
                        diffs.append((p_w.detach().float() - q_w.detach().float()).abs().mean().item())
                    ema_gap = float(np.mean(diffs)) if len(diffs) else 0.0

                current_lr = optimizer.param_groups[0]["lr"]

                # Logs no MLflow
                mlflow.log_metrics({
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "ema_val_loss": ema_va_loss,
                    "ema_val_acc": ema_va_acc,
                    "base_f1_macro": f1_b,
                    "base_auroc": auroc_b,
                    "base_best_threshold": best_th_b,
                    "ema_f1_macro": f1_e,
                    "ema_auroc": auroc_e,
                    "ema_best_threshold": best_th_e,
                    "ema_param_gap_l1_mean": ema_gap,
                    "lr": current_lr,
                }, step=epoch + 1)

                mlflow.set_tags({
                    "base_orientation_flipped": str(flipped_b),
                    "ema_orientation_flipped": str(flipped_e),
                })

                # Print humano
                print(
                    f"Train loss {tr_loss:.4f} | Train acc {tr_acc:.2f}% | "
                    f"Val loss {va_loss:.4f} | Val acc {va_acc:.2f}% | "
                    f"EMA Val loss {ema_va_loss:.4f} | EMA Val acc {ema_va_acc:.2f}% | "
                    f"[BASE] F1 {f1_b:.3f} AUROC {auroc_b:.3f} th {best_th_b:.2f} flip={flipped_b} | "
                    f"[EMA]  F1 {f1_e:.3f} AUROC {auroc_e:.3f} th {best_th_e:.2f} flip={flipped_e} | "
                    f"EMA_gap {ema_gap:.3e} | LR {current_lr:.6f}"
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
