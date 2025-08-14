#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
from typing import Dict, List

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def list_images(p: Path) -> List[Path]:
    return [x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS and x.is_file()]

def stratified_move(raw_dir: Path, data_root: Path, train_ratio: float, seed: int, clear: bool):
    assert 0.5 <= train_ratio < 1.0
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    # opcional: limpar splits antigos
    if clear:
        for split in ("train","val"):
            for c in classes:
                d = data_root / split / c
                if d.exists():
                    shutil.rmtree(d)

    for c in classes:
        files = list_images(raw_dir / c)
        # “embaralhar” determinístico por hash + seed
        files_sorted = sorted(files, key=lambda p: hashlib.sha1((p.name+str(seed)).encode()).hexdigest())
        n = len(files_sorted); k = int(round(n*train_ratio))
        train_files, val_files = files_sorted[:k], files_sorted[k:]

        for split, bunch in (("train",train_files), ("val",val_files)):
            out = data_root / split / c
            out.mkdir(parents=True, exist_ok=True)
            for src in bunch:
                dst = out / src.name
                if not dst.exists():
                    src.replace(dst)

        print(f"{c}: total={n}  train={len(train_files)}  val={len(val_files)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data", help="raiz que contém raw/, train/, val/")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clear_existing", action="store_true", help="apaga splits antigos antes de criar novos")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    raw_dir = data_root / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"Não achei {raw_dir}. Consolidar primeiro com o downloader.")

    stratified_move(raw_dir, data_root, args.train_ratio, args.seed, args.clear_existing)
    print("✅ Split criado em train/ e val/")

if __name__ == "__main__":
    main()
