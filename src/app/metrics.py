import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

def pr_auc_macro_from_probs(y_true, probs_2d):
    if len(y_true) == 0:
        return float("nan")
    y_true = np.array(y_true)
    C = probs_2d.shape[1]
    ap = []
    for c in range(C):
        y_c = (y_true == c).astype(int)
        try:
            ap.append(average_precision_score(y_c, probs_2d[:, c]))
        except ValueError:
            ap.append(np.nan)
    return float(np.nanmean(ap))

def compute_probs_and_metrics(logits, y_true, pos_idx):
    if len(logits) == 0:
        return [], float("nan"), float("nan"), 0.5, False

    with torch.no_grad():
        probs_pos = torch.softmax(logits, dim=1)[:, pos_idx].cpu().numpy().tolist()

    y_pred = [1 if p >= 0.5 else 0 for p in probs_pos]
    f1_dir = f1_score(y_true, y_pred, average="macro")
    try:
        auroc_dir = roc_auc_score(y_true, probs_pos)
    except ValueError:
        auroc_dir = float("nan")

    probs_neg = (1.0 - np.array(probs_pos)).tolist()
    y_pred_neg = [1 if p >= 0.5 else 0 for p in probs_neg]
    f1_inv = f1_score(y_true, y_pred_neg, average="macro")
    try:
        auroc_inv = roc_auc_score(y_true, probs_neg)
    except ValueError:
        auroc_inv = float("nan")

    flipped = False
    f1_best, auroc_best, probs_best = f1_dir, auroc_dir, probs_pos
    if (not np.isnan(auroc_inv)) and (np.isnan(auroc_dir) or auroc_inv > auroc_dir):
        f1_best, auroc_best, probs_best = f1_inv, auroc_inv, probs_neg
        flipped = True

    ths = np.linspace(0.05, 0.95, 19)
    best_f1, best_th = 0.0, 0.5
    for th in ths:
        yp = [1 if p >= th else 0 for p in probs_best]
        f1 = f1_score(y_true, yp, average="macro")
        if f1 > best_f1:
            best_f1, best_th = f1, th

    return probs_best, float(f1_best), float(auroc_best), float(best_th), flipped
