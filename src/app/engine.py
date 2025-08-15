import torch
import torch.nn as nn
import numpy as np
from .logging_utils import autocast_ctx

def _unpack_batch(batch, device):
    # Suporta (imgs, targets) ou (imgs, tabs, targets)
    if len(batch) == 3:
        imgs, tabs, targets = batch
        return imgs.to(device), tabs.to(device), targets.to(device)
    imgs, targets = batch
    return imgs.to(device), None, targets.to(device)

def forward_model(model, imgs, tabs, device):
    with autocast_ctx(device):
        if tabs is None:
            return model(imgs)
        return model(imgs, tabs)

def train_one_epoch(model, loader, optimizer, device, loss_fn,
                    mixup_fn=None, ema=None, scheduler=None, max_grad_norm=1.0):
    model.train()
    running_loss, correct, total, steps = 0.0, 0, 0, 0
    for batch in loader:
        imgs, tabs, targets = _unpack_batch(batch, device)
        if mixup_fn is not None and tabs is None:  # evite mixup com fusão
            imgs, targets = mixup_fn(imgs, targets)

        outputs = forward_model(model, imgs, tabs, device)
        loss = loss_fn(outputs, targets)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        running_loss += loss.item()
        steps += 1
        hard = targets.argmax(dim=1) if targets.ndim == 2 else targets
        preds = outputs.argmax(dim=1)
        correct += (preds == hard).sum().item()
        total += hard.size(0)

    return running_loss / max(1, steps), (correct / total * 100.0 if total else float("nan"))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    running_loss, correct, total, steps = 0.0, 0, 0, 0
    for batch in loader:
        imgs, tabs, targets = _unpack_batch(batch, device)
        outputs = forward_model(model, imgs, tabs, device)
        loss = ce(outputs, targets)
        running_loss += loss.item(); steps += 1
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / max(1, steps), (correct / total * 100.0 if total else 0.0)

@torch.no_grad()
def predict_all_logits(model, loader, device):
    model.eval()
    ys, logits = [], []
    for batch in loader:
        imgs, tabs, y = _unpack_batch(batch, device)
        logit = forward_model(model, imgs, tabs, device)
        ys.extend(y.cpu().tolist())
        logits.append(logit.detach().cpu())
    if len(logits) == 0:
        return [], []
    return ys, torch.cat(logits, dim=0)
