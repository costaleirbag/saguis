import os
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

import pandas as pd
from glob import glob
from datetime import datetime

@dataclass
class Args:
    data_dir: str
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    aug_backend: str = "alb"       # "alb" ou "torchvision"
    aug_preset: str = "strong"     # "none" | "light" | "strong"


class PathImageFolder(datasets.ImageFolder):
    """Retorna (img, target, path)."""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class TabularLookup:
    """
    Lê CSVs preparados e cria vetores por 'basename' (se existir) ou pelo basename do image_url.
    Mantém cols_order para uso consistente em treino/inferência.
    """
    def __init__(self, csv_glob: str):
        paths = sorted(glob(csv_glob))
        if not paths:
            raise FileNotFoundError(f"Nenhum CSV em {csv_glob}")
        dfs = []
        for p in paths:
            df = pd.read_csv(p)
            if "image_url" not in df.columns:
                raise ValueError(f"{p} precisa da coluna 'image_url'")
            # define basename preferindo coluna pronta; senão extrai do image_url
            if "basename" in df.columns:
                base = df["basename"].astype(str)
                base = base.where(base.str.len() > 0, df["image_url"].astype(str).apply(lambda s: s.split("/")[-1].split("?")[0]))
            else:
                base = df["image_url"].astype(str).apply(lambda s: s.split("/")[-1].split("?")[0])
            # normaliza extensão
            base = base.apply(lambda s: s if any(s.lower().endswith(ext) for ext in (".jpg",".jpeg",".png")) else f"{s}.jpg")
            df["basename"] = base
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)
        self._engineer()

    def _engineer(self):
        # Datas (opcional)
        def parse_date(s):
            try:
                return datetime.strptime(str(s), "%d/%m/%Y")
            except Exception:
                return pd.NaT

        if "observed_on" in self.df.columns:
            dts = self.df["observed_on"].apply(parse_date)
        else:
            dts = pd.Series(pd.NaT, index=self.df.index)

        self.df["year"] = dts.dt.year
        self.df["month"] = dts.dt.month
        self.df["dayofyear"] = dts.dt.dayofyear

        two_pi = 2*np.pi
        self.df["month_sin"] = np.sin(two_pi*(self.df["month"].fillna(0)/12))
        self.df["month_cos"] = np.cos(two_pi*(self.df["month"].fillna(0)/12))
        self.df["doy_sin"]   = np.sin(two_pi*(self.df["dayofyear"].fillna(0)/366))
        self.df["doy_cos"]   = np.cos(two_pi*(self.df["dayofyear"].fillna(0)/366))

        # One-hot do estado (se existir)
        if "place_state_name" in self.df.columns:
            states = pd.get_dummies(self.df["place_state_name"].astype(str), prefix="state")
        else:
            states = pd.DataFrame(index=self.df.index)  # vazio

        self.df = pd.concat([self.df, states], axis=1)

        # colunas que vão pro vetor tabular (ordem fixa)
        base_cols = [
            "latitude", "longitude", "year", "month", "dayofyear",
            "month_sin", "month_cos", "doy_sin", "doy_cos"
        ]
        # garante colunas ausentes como 0
        for c in base_cols:
            if c not in self.df.columns:
                self.df[c] = 0.0

        state_cols = [c for c in self.df.columns if c.startswith("state_")]
        self.cols_order = base_cols + state_cols

        # index por basename (hash)
        dedup = self.df.dropna(subset=["basename"]).drop_duplicates(subset=["basename"]).set_index("basename")
        # seleção e cast
        cols_present = [c for c in self.cols_order if c in dedup.columns]
        # se algo estiver faltando, completa com 0
        for c in self.cols_order:
            if c not in dedup.columns:
                dedup[c] = 0.0
        self.vecs_by_basename = dedup[self.cols_order].astype(float)

    def vector_for_basename(self, basename: str):
        try:
            return self.vecs_by_basename.loc[basename].to_numpy(dtype=np.float32)
        except KeyError:
            return None


def build_dataloaders_fusion(args: Args, csv_glob: str, use_weighted_sampler: bool = True):
    """
    Constrói DataLoaders (treino/val) para FUSÃO imagem+tabular.
    O casamento é feito por 'basename' (<hash>.jpg) obtido do log de download.
    """
    data_root = Path(args.data_dir)
    train_dir, val_dir = data_root / "train", data_root / "val"
    train_tf, val_tf = build_transforms(args.img_size, args.aug_backend, args.aug_preset)

    # carrega datasets com path
    if train_dir.exists():
        base_train = PathImageFolder(train_dir, transform=train_tf)
        class_to_idx = base_train.class_to_idx
        if val_dir.exists():
            base_val = PathImageFolder(val_dir, transform=val_tf)
        else:
            full = PathImageFolder(train_dir, transform=train_tf)
            gen = torch.Generator().manual_seed(42)
            val_size = max(1, int(0.1 * len(full)))
            train_size = len(full) - val_size
            tr_subset, va_subset = random_split(full, [train_size, val_size], generator=gen)
            base_train = PathImageFolder(train_dir, transform=train_tf)
            base_val   = PathImageFolder(train_dir, transform=val_tf)
            base_train.samples = [full.samples[i] for i in tr_subset.indices]
            base_val.samples   = [full.samples[i] for i in va_subset.indices]
    else:
        full = PathImageFolder(data_root, transform=train_tf)
        gen = torch.Generator().manual_seed(42)
        val_size = max(1, int(0.1 * len(full)))
        train_size = len(full) - val_size
        tr_subset, va_subset = random_split(full, [train_size, val_size], generator=gen)
        base_train = PathImageFolder(data_root, transform=train_tf)
        base_val   = PathImageFolder(data_root, transform=val_tf)
        base_train.samples = [full.samples[i] for i in tr_subset.indices]
        base_val.samples   = [full.samples[i] for i in va_subset.indices]
        class_to_idx = base_train.class_to_idx

    # tabular lookup (usa data/_download_ok.csv por padrão)
    tab = TabularLookup(csv_glob, download_ok_csv=Path(args.data_dir).parent / "_download_ok.csv")

    # filtra amostras sem vetor tabular (por basename)
    def filter_with_tab(base_ds: PathImageFolder):
        kept = []
        for p, y in base_ds.samples:
            base = Path(p).name  # <hash>.jpg
            v = tab.vector_for_basename(base)
            if v is not None and np.isfinite(v).all():
                kept.append((p, y))
        new_ds = PathImageFolder(base_ds.root, transform=base_ds.transform)
        new_ds.samples = kept
        new_ds.class_to_idx = base_ds.class_to_idx
        return new_ds

    train_ds = filter_with_tab(base_train)
    val_ds   = filter_with_tab(base_val)

    # contagens e sampler ponderado (como já faz)
    train_counts, train_samples = _counts_from_dataset(train_ds)
    val_counts,   _             = _counts_from_dataset(val_ds)
    num_classes = len(class_to_idx)

    train_sampler = None
    if use_weighted_sampler:
        class_freq = torch.zeros(num_classes, dtype=torch.float)
        for y, cnt in train_counts.items():
            class_freq[y] = cnt
        class_weights = 1.0 / (class_freq + 1e-8)
        sample_weights = torch.tensor([class_weights[y] for _, y in train_samples], dtype=torch.float)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False
    else:
        shuffle_train = True

    def collate_with_tab(batch):
        imgs, ys, paths = zip(*batch)
        bases = [Path(p).name for p in paths]  # <hash>.jpg
        tab_vecs = []
        for base in bases:
            v = tab.vector_for_basename(base)
            if v is None:
                v = np.zeros(len(tab.cols_order), dtype=np.float32)
            tab_vecs.append(v)
        tabs = torch.tensor(np.stack(tab_vecs, axis=0), dtype=torch.float32)
        imgs = torch.stack(imgs, dim=0)
        ys   = torch.tensor(ys, dtype=torch.long)
        return imgs, tabs, ys

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_with_tab,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(8, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_with_tab,
    )
    return train_dl, val_dl, class_to_idx, train_counts, val_counts, tab.cols_order  # <- ordem das colunas tabulares


class AlbumentationsTransform:
    """
    Wrapper para aplicar A.Compose em imagens PIL (ImageFolder) e devolver tensors PyTorch.
    """
    def __init__(self, augment: A.Compose):
        self.augment = augment

    def __call__(self, img: Image.Image):
        # Garante RGB (evita RGBA/grayscale com 1 ou 4 canais)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_np = np.array(img)  # (H, W, C), uint8
        out = self.augment(image=img_np)
        return out["image"]


def build_transforms(img_size: int, aug_backend: str = "alb", aug_preset: str = "strong"):
    """
    Transforms para treino/val com dois backends: Albumentations ou Torchvision.
    Presets: none | light | strong
    """
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    if aug_backend == "alb":
        # ---- Albumentations ----
        if aug_preset == "none":
            train_aug = A.Compose([
                A.SmallestMaxSize(max_size=int(img_size * 1.15)),
                A.CenterCrop(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        elif aug_preset == "light":
            train_aug = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10, val_shift_limit=10, p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:  # strong
            train_aug = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    scale=(0.9, 1.1),
                    rotate=(-5, 5),
                    shear=0,
                    p=0.25
                ),
                A.Perspective(scale=(0.02, 0.05), p=0.15),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=12, val_shift_limit=12, p=0.25),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.20),
                A.OneOf([
                    A.GaussNoise(std_range=(5/255, 25/255), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.20),
                A.ImageCompression(quality_range=(60, 95), p=0.25),
                A.GridDropout(ratio=0.5, random_offset=True, p=0.20),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

        val_aug = A.Compose([
            A.SmallestMaxSize(max_size=int(img_size * 1.15)),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

        train_tf = AlbumentationsTransform(train_aug)
        val_tf   = AlbumentationsTransform(val_aug)

    else:
        # ---- Torchvision ----
        from torchvision import transforms as T
        normalize = T.Normalize(mean=mean, std=std)

        if aug_preset == "none":
            train_tf = T.Compose([
                T.Resize(int(img_size * 1.15)),
                T.CenterCrop(img_size),
                T.ToTensor(),
                normalize,
            ])
        elif aug_preset == "light":
            train_tf = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                T.ToTensor(),
                normalize,
            ])
        else:  # strong
            train_tf = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.RandomRotation(5)], p=0.2),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.15),
                T.ToTensor(),
                normalize,
                T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
            ])

        val_tf = T.Compose([
            T.Resize(int(img_size * 1.15)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            normalize,
        ])

    return train_tf, val_tf


def _counts_from_dataset(ds):
    # Suporta ImageFolder ou Subset(ImageFolder)
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
    train_tf, val_tf = build_transforms(args.img_size, args.aug_backend, args.aug_preset)

    # Construção de datasets (com split se necessário)
    if train_dir.exists():
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        class_to_idx = train_ds.class_to_idx

        if val_dir.exists():
            val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        else:
            # Split 90/10 a partir do train
            full = datasets.ImageFolder(train_dir, transform=train_tf)
            gen = torch.Generator().manual_seed(42)
            val_size = max(1, int(0.1 * len(full)))
            train_size = len(full) - val_size
            tr_subset, va_subset = random_split(full, [train_size, val_size], generator=gen)

            train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
            val_ds   = datasets.ImageFolder(train_dir, transform=val_tf)
            train_ds.samples = [full.samples[i] for i in tr_subset.indices]
            val_ds.samples   = [full.samples[i] for i in va_subset.indices]
            class_to_idx = train_ds.class_to_idx
    else:
        # Estrutura sem pastas train/val: split 90/10 no diretório raiz
        full = datasets.ImageFolder(data_root, transform=train_tf)
        gen = torch.Generator().manual_seed(42)
        val_size = max(1, int(0.1 * len(full)))
        train_size = len(full) - val_size
        tr_subset, va_subset = random_split(full, [train_size, val_size], generator=gen)

        train_ds = datasets.ImageFolder(data_root, transform=train_tf)
        val_ds   = datasets.ImageFolder(data_root, transform=val_tf)
        train_ds.samples = [full.samples[i] for i in tr_subset.indices]
        val_ds.samples   = [full.samples[i] for i in va_subset.indices]
        class_to_idx = train_ds.class_to_idx

    # --- Sampler ponderado (inverso da frequência) ---
    train_sampler = None
    if use_weighted_sampler:
        counts, samples = _counts_from_dataset(train_ds)
        num_classes = len(class_to_idx)
        class_freq = torch.zeros(num_classes, dtype=torch.float)
        for y, cnt in counts.items():
            class_freq[y] = cnt
        class_weights = 1.0 / (class_freq + 1e-8)
        sample_weights = torch.tensor([class_weights[y] for _, y in samples], dtype=torch.float)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False
    else:
        shuffle_train = True

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(8, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    train_counts, _ = _counts_from_dataset(train_ds)
    val_counts, _   = _counts_from_dataset(val_ds)

    return train_dl, val_dl, class_to_idx, train_counts, val_counts


# --- K-FOLD ESTRATIFICADO ---
from sklearn.model_selection import StratifiedKFold

def _subset_imagefolder(base_ds: datasets.ImageFolder, indices, transform):
    ds = datasets.ImageFolder(base_ds.root, transform=transform)
    ds.class_to_idx = base_ds.class_to_idx
    ds.samples = [base_ds.samples[i] for i in indices]
    ds.targets = [s[1] for s in ds.samples]
    return ds

def build_kfold_dataloaders(
    args: Args,
    n_splits: int = 5,
    fold: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    use_weighted_sampler: bool = True,
):
    data_root = Path(args.data_dir)
    train_root = data_root / "train"
    root = train_root if train_root.exists() else data_root

    train_tf, val_tf = build_transforms(args.img_size, args.aug_backend, args.aug_preset)

    base = datasets.ImageFolder(root, transform=train_tf)
    class_to_idx = base.class_to_idx
    samples = base.samples
    if len(samples) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em {root}.")

    y = [s[1] for s in samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y))
    if not (0 <= fold < n_splits):
        raise ValueError(f"fold={fold} inválido (0..{n_splits-1}).")
    train_idx, val_idx = folds[fold]

    train_ds = _subset_imagefolder(base, train_idx, transform=train_tf)
    val_ds   = _subset_imagefolder(base, val_idx,   transform=val_tf)

    train_sampler = None
    if use_weighted_sampler:
        counts, subsamples = _counts_from_dataset(train_ds)
        num_classes = len(class_to_idx)
        class_freq = torch.zeros(num_classes, dtype=torch.float)
        for cls_idx, cnt in counts.items():
            class_freq[cls_idx] = cnt
        class_weights = 1.0 / (class_freq + 1e-8)
        sample_weights = torch.tensor([class_weights[y] for _, y in train_ds.samples], dtype=torch.float)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle_train = False
    else:
        shuffle_train = True

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=max(8, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    train_counts, _ = _counts_from_dataset(train_ds)
    val_counts, _   = _counts_from_dataset(val_ds)

    meta = {
        "n_splits": n_splits,
        "fold": fold,
        "train_size": len(train_ds.samples),
        "val_size": len(val_ds.samples),
        "root_used": str(root),
    }
    return train_dl, val_dl, class_to_idx, train_counts, val_counts, meta
