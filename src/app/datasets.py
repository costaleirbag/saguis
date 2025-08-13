import os
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from dataclasses import dataclass

@dataclass
class Args:
    data_dir: str
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),

        # <= Mude a ordem: ToTensor ANTES do ColorJitter
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),

        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf

def build_dataloaders(args: Args):
    data_root = Path(args.data_dir)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    train_tf, val_tf = build_transforms(args.img_size)
    if train_dir.exists():
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        if val_dir.exists():
            val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        else:
            val_size = max(1, int(0.15 * len(train_ds)))
            train_size = len(train_ds) - val_size
            train_base, val_base = random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            val_ds = datasets.ImageFolder(train_dir, transform=val_tf)
            val_ds.samples = [train_ds.samples[i] for i in val_base.indices]
            train_ds.samples = [train_ds.samples[i] for i in train_base.indices]
    else:
        full_ds = datasets.ImageFolder(data_root, transform=train_tf)
        val_size = max(1, int(0.15 * len(full_ds)))
        train_size = len(full_ds) - val_size
        train_base, val_base = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        train_ds = full_ds
        val_ds = datasets.ImageFolder(data_root, transform=val_tf)
        val_ds.samples = [full_ds.samples[i] for i in val_base.indices]
        train_ds.samples = [full_ds.samples[i] for i in train_base.indices]
    class_to_idx = train_ds.class_to_idx
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=max(8, args.batch_size // 2), shuffle=False, num_workers=args.num_workers, pin_memory=False)
    # Now expects two classes: 'hybrid' and 'non-hybrid'
    return train_dl, val_dl, class_to_idx
