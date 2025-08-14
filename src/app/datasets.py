import os
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
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

def _counts_from_dataset(ds):
    # suporta ImageFolder ou Subset(ImageFolder)
    try:
        samples = ds.samples
    except AttributeError:
        base, idxs = ds.dataset, ds.indices
        samples = [base.samples[i] for i in idxs]
    counts = {}
    for _, y in samples:
        counts[y] = counts.get(y, 0) + 1
    return counts, samples

def build_dataloaders(args: Args, use_weighted_sampler: bool = True):
    data_root = Path(args.data_dir)
    train_dir, val_dir = data_root / "train", data_root / "val"
    train_tf, val_tf = build_transforms(args.img_size)

    if train_dir.exists():
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        if val_dir.exists():
            val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        else:
            # split de dentro do train
            full = datasets.ImageFolder(train_dir, transform=train_tf)
            gen = torch.Generator().manual_seed(42)
            val_size = max(1, int(0.1 * len(full)))
            train_size = len(full) - val_size
            tr_idx, va_idx = random_split(range(len(full)), [train_size, val_size], generator=gen)
            train_ds = full
            val_ds = datasets.ImageFolder(train_dir, transform=val_tf)
            val_ds.samples = [full.samples[i] for i in va_idx.indices]
            train_ds.samples = [full.samples[i] for i in tr_idx.indices]
    else:
        full = datasets.ImageFolder(data_root, transform=train_tf)
        gen = torch.Generator().manual_seed(42)
        val_size = max(1, int(0.1 * len(full)))
        train_size = len(full) - val_size
        tr, va = random_split(full, [train_size, val_size], generator=gen)
        train_ds = full
        val_ds = datasets.ImageFolder(data_root, transform=val_tf)
        val_ds.samples = [full.samples[i] for i in va.indices]
        train_ds.samples = [full.samples[i] for i in tr.indices]

    class_to_idx = train_ds.class_to_idx

    # --- Sampler ponderado (inverso da frequência) ---
    train_sampler = None
    if use_weighted_sampler:
        counts, samples = _counts_from_dataset(train_ds)
        # ordem: índice da classe
        num_classes = len(class_to_idx)
        class_freq = torch.zeros(num_classes, dtype=torch.float)
        for y, cnt in counts.items():
            class_freq[y] = cnt
        class_weights = 1.0 / (class_freq + 1e-8)
        # peso por amostra
        sample_weights = torch.tensor([class_weights[y] for _, y in samples], dtype=torch.float)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        shuffle_train = False
    else:
        shuffle_train = True

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=shuffle_train, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=max(8, args.batch_size // 2),
        shuffle=False, num_workers=args.num_workers, pin_memory=False
    )
    return train_dl, val_dl, class_to_idx
