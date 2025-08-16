import os, json
from pathlib import Path
import numpy as np
import torch
import mlflow
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
from sklearn.metrics import (
    precision_recall_fscore_support, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix
)

from .args import build_parser  # opcional se quiser reusar
from .logging_utils import _safe_log_metrics, get_device
from .engine import train_one_epoch, evaluate, predict_all_logits
from .metrics import compute_probs_and_metrics, pr_auc_macro_from_probs
from .datasets import Args as DataArgs, build_dataloaders, build_kfold_dataloaders, build_dataloaders_fusion
from .models import build_model, build_optimizer, build_scheduler

def run_training(args):
    device = get_device()
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    dl_args = DataArgs(
        data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, aug_backend=args.aug_backend, aug_preset=args.aug_preset
    )

    fusion_active = bool(args.fusion_tab_csv_glob)
    if args.cv_splits and args.cv_splits > 1:
        train_dl, val_dl, class_to_idx, train_counts, val_counts, meta = build_kfold_dataloaders(
            dl_args, n_splits=args.cv_splits, fold=args.cv_fold, use_weighted_sampler=True
        )
        tab_dim = None
        print(f"[CV] Using fold {meta['fold']+1}/{meta['n_splits']} | "
              f"train={meta['train_size']} val={meta['val_size']} | root={meta['root_used']}")
        if fusion_active:
            print("[WARN] CV+fusão não implementado aqui (use treino simples ou crie variante kfold_fusion).")
    else:
        if fusion_active:
            train_dl, val_dl, class_to_idx, train_counts, val_counts, tab_cols = build_dataloaders_fusion(
                dl_args, csv_glob=args.fusion_tab_csv_glob, use_weighted_sampler=True
            )
            tab_dim = len(tab_cols)
        else:
            train_dl, val_dl, class_to_idx, train_counts, val_counts = build_dataloaders(dl_args)
            tab_dim = None

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    train_counts_named = {idx_to_class[i]: int(c) for i, c in train_counts.items()}
    val_counts_named   = {idx_to_class[i]: int(c) for i, c in val_counts.items()}

    print(f"Classes: {class_to_idx} (num_classes={num_classes})")
    print("Train counts:", train_counts_named, f"(total={sum(train_counts.values())})")
    print("Val counts:  ", val_counts_named,   f"(total={sum(val_counts.values())})")
    print(f"Augmentations: backend={args.aug_backend}, preset={args.aug_preset}")
    print(f"Steps/epoch (train): {len(train_dl)} | (val): {len(val_dl)}")
    if len(train_dl) == 0:
        raise RuntimeError("len(train_dl) == 0.")

    pos_idx = class_to_idx[args.pos_class] if args.pos_class in class_to_idx else max(class_to_idx.values())
    val_total = int(sum(val_counts.values()))
    val_pos_prev = float(val_counts.get(pos_idx, 0) / max(1, val_total))
    print(f"[Info] Val prevalence (pos='{args.pos_class}') ≈ {val_pos_prev:.3f}")

    model = build_model(args.model, num_classes=num_classes, tab_dim=tab_dim).to(device)

    head_names = ("head", "fc", "classifier", "cls")
    base_params = [p for n, p in model.named_parameters() if not any(h in n for h in head_names)]
    for p in base_params:
        p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable}/{total}")
    if trainable == 0:
        raise RuntimeError("Nenhum parâmetro com requires_grad=True.")

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                                num_steps_per_epoch=len(train_dl))

    if fusion_active:
        mixup_fn = None
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing if args.smoothing > 0 else 0.0)
    else:
        mixup_active = (args.mixup > 0.0) or (args.cutmix > 0.0)
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            label_smoothing=args.smoothing, num_classes=num_classes,
        ) if mixup_active else None
        loss_fn = SoftTargetCrossEntropy() if mixup_active else LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    ema = ModelEmaV2(model, decay=args.ema_decay)

    monitor_name = "bal_acc_ema"
    best_monitor = -1.0
    best_path = Path(args.out_dir) / "best_model.pth"
    last_path = Path(args.out_dir) / "last.pth"

    start_epoch = 0
    ckpt_path = Path(args.resume) if args.resume else None
    if args.auto_resume and last_path.exists() and not ckpt_path:
        ckpt_path = last_path
    if ckpt_path and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state"])
        if "ema_state" in ckpt:       ema.module.load_state_dict(ckpt["ema_state"])
        best_monitor = ckpt.get("best_acc", -1.0)
        start_epoch = ckpt.get("epoch", -1) + 1
        if start_epoch >= args.freeze_epochs:
            for p in base_params: p.requires_grad_(True)
    else:
        print("Fresh run (no resume).")

    mlflow.set_experiment("saguis")
    run_name = f"{args.model}-{Path(args.data_dir).name}" + ("-fusion" if fusion_active else "")
    with mlflow.start_run(run_name=run_name):
        print("[MLflow] tracking_uri:", mlflow.get_tracking_uri())
        print("[MLflow] run_id:", mlflow.active_run().info.run_id)

        mlflow.set_tag("device", device.type)
        mlflow.set_tag("num_classes", num_classes)
        if args.cv_splits and args.cv_splits > 1:
            mlflow.set_tags({
                "cv_splits": str(args.cv_splits),
                "cv_fold": str(args.cv_fold),
                "cv_group": f"{args.model}-{Path(args.data_dir).name}-cv{args.cv_splits}",
            })
        if fusion_active:
            mlflow.set_tag("fusion", "tabular")
            mlflow.log_param("fusion_tab_csv_glob", args.fusion_tab_csv_glob)

        mlflow.log_params({
            "model": args.model,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "freeze_epochs": args.freeze_epochs,
            "mixup": args.mixup if not fusion_active else 0.0,
            "cutmix": args.cutmix if not fusion_active else 0.0,
            "smoothing": args.smoothing,
            "num_workers": args.num_workers,
            "data_dir": args.data_dir,
            "pos_class": args.pos_class,
            "ema_decay": args.ema_decay,
            "aug_backend": args.aug_backend,
            "aug_preset": args.aug_preset,
            "train_total": int(sum(train_counts.values())),
            "val_total": int(sum(val_counts.values())),
            "train_counts_json": json.dumps(train_counts_named, ensure_ascii=False),
            "val_counts_json": json.dumps(val_counts_named, ensure_ascii=False),
            "monitor_metric": monitor_name,
        })
        if ckpt_path:
            mlflow.set_tag("resume_from", str(ckpt_path))

        try:
            for epoch in range(start_epoch, args.epochs):
                if epoch == args.freeze_epochs:
                    for p in base_params: p.requires_grad_(True)
                    print("Unfroze backbone for full fine-tuning.")

                print(f"\nEpoch {epoch+1}/{args.epochs}")
                tr_loss, tr_acc = train_one_epoch(
                    model, train_dl, optimizer, device, loss_fn,
                    mixup_fn=mixup_fn, ema=ema, scheduler=scheduler
                )

                va_loss, va_acc = evaluate(model, val_dl, device)
                ema_va_loss, ema_va_acc = evaluate(ema.module, val_dl, device)

                y_true_base, logits_base = predict_all_logits(model, val_dl, device)
                y_true_ema,  logits_ema  = predict_all_logits(ema.module, val_dl, device)

                _, f1_b, auroc_b, best_th_b, flipped_b = compute_probs_and_metrics(
                    logits_base, y_true_base, pos_idx
                )
                _, f1_e, auroc_e, best_th_e, flipped_e = compute_probs_and_metrics(
                    logits_ema, y_true_ema, pos_idx
                )

                probs_base = torch.softmax(logits_base, dim=1).numpy() if len(y_true_base) else np.empty((0, num_classes))
                preds_base = probs_base.argmax(axis=1) if probs_base.size else np.array([])
                pred_pos_rate_base = float((preds_base == pos_idx).mean()) if preds_base.size else float("nan")
                bal_acc_base = balanced_accuracy_score(y_true_base, preds_base) if preds_base.size else float("nan")
                mcc_base = matthews_corrcoef(y_true_base, preds_base) if preds_base.size else float("nan")
                pr_auc_base = pr_auc_macro_from_probs(y_true_base, probs_base) if probs_base.size else float("nan")
                if preds_base.size:
                    prec_b, rec_b, f1_b_pc, _ = precision_recall_fscore_support(
                        y_true_base, preds_base, labels=list(range(num_classes)), zero_division=0
                    )
                else:
                    prec_b = rec_b = f1_b_pc = np.array([np.nan]*num_classes)

                probs_ema = torch.softmax(logits_ema, dim=1).numpy() if len(y_true_ema) else np.empty((0, num_classes))
                preds_ema = probs_ema.argmax(axis=1) if probs_ema.size else np.array([])
                pred_pos_rate_ema = float((preds_ema == pos_idx).mean()) if preds_ema.size else float("nan")
                bal_acc_ema = balanced_accuracy_score(y_true_ema, preds_ema) if preds_ema.size else float("nan")
                mcc_ema = matthews_corrcoef(y_true_ema, preds_ema) if preds_ema.size else float("nan")
                pr_auc_ema = pr_auc_macro_from_probs(y_true_ema, probs_ema) if probs_ema.size else float("nan")

                if preds_ema.size:
                    prec_e, rec_e, f1_e_pc, _ = precision_recall_fscore_support(
                        y_true_ema, preds_ema, labels=list(range(num_classes)), zero_division=0
                    )
                    cm = confusion_matrix(y_true_ema, preds_ema, labels=list(range(num_classes)))
                else:
                    prec_e = rec_e = f1_e_pc = np.array([np.nan]*num_classes)
                    cm = np.zeros((num_classes, num_classes), dtype=int)

                with torch.no_grad():
                    diffs = []
                    for p_w, q_w in zip(model.parameters(), ema.module.parameters()):
                        diffs.append((p_w.detach().float() - q_w.detach().float()).abs().mean().item())
                    ema_gap = float(np.mean(diffs)) if len(diffs) else 0.0

                try:
                    current_lr = float(scheduler.get_last_lr()[0])
                except Exception:
                    current_lr = float(optimizer.param_groups[0]["lr"])

                log_payload = {
                    "epoch": epoch + 1,
                    "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": va_loss,   "val_acc": va_acc,
                    "ema_val_loss": ema_va_loss, "ema_val_acc": ema_va_acc,
                    "base_f1_macro": f1_b, "base_auroc": auroc_b, "base_best_threshold": best_th_b,
                    "ema_f1_macro": f1_e,  "ema_auroc": auroc_e,  "ema_best_threshold": best_th_e,
                    "ema_param_gap_l1_mean": ema_gap, "lr": current_lr,
                    "bal_acc_base": bal_acc_base, "mcc_base": mcc_base,
                    "pred_pos_rate_base": pred_pos_rate_base, "pr_auc_base_macro": pr_auc_base,
                    "bal_acc_ema": bal_acc_ema, "mcc_ema": mcc_ema,
                    "pred_pos_rate_ema": pred_pos_rate_ema, "pr_auc_ema_macro": pr_auc_ema,
                }
                for c in range(num_classes):
                    log_payload[f"prec_base_c{c}"] = float(prec_b[c])
                    log_payload[f"rec_base_c{c}"]  = float(rec_b[c])
                    log_payload[f"f1_base_c{c}"]   = float(f1_b_pc[c])
                    log_payload[f"prec_ema_c{c}"] = float(prec_e[c])
                    log_payload[f"rec_ema_c{c}"]  = float(rec_e[c])
                    log_payload[f"f1_ema_c{c}"]   = float(f1_e_pc[c])

                for i in range(num_classes):
                    for j in range(num_classes):
                        log_payload[f"cm_ema_{i}{j}"] = int(cm[i, j])

                _safe_log_metrics(log_payload, step=epoch + 1)

                mlflow.set_tags({
                    "base_orientation_flipped": str(bool(auroc_b != auroc_b)),  # simples placeholder
                    "ema_orientation_flipped":  str(bool(auroc_e != auroc_e)),
                })

                print(
                    f"Train loss {tr_loss:.4f} | Train acc {tr_acc:.2f}% | "
                    f"Val loss {va_loss:.4f} | Val acc {va_acc:.2f}% | "
                    f"EMA Val loss {ema_va_loss:.4f} | EMA Val acc {ema_va_acc:.2f}% | "
                    f"[BASE] F1 {f1_b:.3f} AUROC {auroc_b:.3f} th {best_th_b:.2f} | "
                    f"[EMA]  F1 {f1_e:.3f} AUROC {auroc_e:.3f} th {best_th_e:.2f} | "
                    f"bal_acc_ema {bal_acc_ema:.3f} | mcc_ema {mcc_ema:.3f} | "
                    f"PR-AUC(macro)_ema {pr_auc_ema:.3f} | EMA_gap {ema_gap:.3e} | LR {current_lr:.6f}"
                )

                torch.save({
                    "epoch": epoch,
                    "best_acc": best_monitor,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "ema_state": ema.module.state_dict(),
                    "class_to_idx": class_to_idx,
                    "model_name": args.model,
                    "img_size": args.img_size,
                    "tab_dim": (tab_dim if fusion_active else None),
                }, last_path)

                monitor_value = bal_acc_ema if not np.isnan(bal_acc_ema) else -1.0
                if monitor_value > best_monitor:
                    best_monitor = monitor_value
                    torch.save({
                        "model_state": ema.module.state_dict(),
                        "class_to_idx": class_to_idx,
                        "model_name": args.model,
                        "img_size": args.img_size,
                        "tab_dim": (tab_dim if fusion_active else None),
                    }, best_path)
                    with open(Path(args.out_dir) / "class_to_idx.json", "w") as f:
                        json.dump(class_to_idx, f, indent=2)
                    print(f"Saved new best ({monitor_name}={best_monitor:.3f}): {best_path}")

        except KeyboardInterrupt:
            print("\n[interrupt] salvando last.pth antes de sair…")
            torch.save({
                "epoch": epoch if 'epoch' in locals() else -1,
                "best_acc": best_monitor,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "ema_state": ema.module.state_dict(),
                "class_to_idx": class_to_idx,
                "model_name": args.model,
                "img_size": args.img_size,
                "tab_dim": (tab_dim if fusion_active else None),
            }, last_path)
            raise

        mlflow.log_artifact(str(best_path))
        if (Path(args.out_dir) / "class_to_idx.json").exists():
            mlflow.log_artifact(str(Path(args.out_dir) / "class_to_idx.json"))
        if last_path.exists():
            mlflow.log_artifact(str(last_path))

    print("Done.")
    print(f"Best ({monitor_name}) = {best_monitor:.3f} | Checkpoint: {best_path}")
