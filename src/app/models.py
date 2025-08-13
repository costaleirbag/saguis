import timm
import torch.nn as nn
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR

def build_model(model_name: str, num_classes: int = 2):
    # num_classes should be 2: hybrid and non-hybrid
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def build_scheduler(optimizer, epochs: int, warmup_epochs: int, num_steps_per_epoch: int):
    total_steps = epochs * num_steps_per_epoch
    warmup_steps = max(1, int(warmup_epochs * num_steps_per_epoch))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
