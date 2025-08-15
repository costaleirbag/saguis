import timm
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR

class FusionNet(nn.Module):
    """
    Extrai features visuais (GAP) + projeta tabular por MLP e concatena.
    """
    def __init__(self, backbone_name: str, num_classes: int, tab_dim: int, img_feat_dim: int | None = None):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)  # sem head
        # tenta inferir dimensão do feature da CNN
        self.img_feat_dim = img_feat_dim or getattr(self.backbone, "num_features", None)
        if self.img_feat_dim is None:
            # fallback: faz um forward fake depois (ver forward)
            self.img_feat_dim = -1

        # MLP para tabular
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        # cabeça de classificação após concat
        # se não sabemos img_feat_dim ainda, criamos preguiçosamente no forward
        self.cls = None
        self.num_classes = num_classes

    def _ensure_cls(self, d_img):
        if self.cls is None:
            self.cls = nn.Sequential(
                nn.Linear(d_img + 128, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, self.num_classes),
            ).to(next(self.parameters()).device)

    def forward(self, x_img, x_tab):
        feats = self.backbone.forward_features(x_img) if hasattr(self.backbone, "forward_features") else self.backbone(x_img)
        if feats.ndim == 4:
            feats = feats.mean(dim=(2,3))
        d_img = feats.shape[1]
        if self.img_feat_dim < 0:
            self._ensure_cls(d_img)
        z_tab = self.tab_mlp(x_tab)
        z = torch.cat([feats, z_tab], dim=1)
        if self.cls is None:  # se não criada
            self._ensure_cls(d_img)
        logits = self.cls(z)
        return logits


def build_model(model_name: str, num_classes: int, tab_dim: int | None = None):
    if tab_dim is None:
        # caminho antigo (só imagem)
        import timm
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    else:
        return FusionNet(model_name, num_classes=num_classes, tab_dim=tab_dim)

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
