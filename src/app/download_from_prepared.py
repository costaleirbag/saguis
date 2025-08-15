#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, hashlib, io, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/119.0 Safari/537.36")

SAVE_EXT = ".jpg"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def map_tipo_binary(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("h", "hybrido", "híbrido", "hybrid"): return "hibrido"
    if s in ("n-h", "não-hibrido", "nao-hibrido", "non-hybrid", "nh"): return "nao_hibrido"
    if "hybr" in s or "hí" in s or "hibr" in s: return "hibrido"
    return "nao_hibrido"

def hashed_filename_from_url(url: str, ext: str = SAVE_EXT, nhex: int = 16) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:nhex]
    return f"{h}{ext}"

def fetch_image(url: str, timeout: float = 15.0, max_retries: int = 2) -> Optional[bytes]:
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(0.8 * (attempt + 1))
    return None

def is_image_ok(content: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(content)) as im:
            im.verify()
        return True
    except Exception:
        return False

def save_jpeg(content: bytes, out_path: Path, quality: int = 92) -> None:
    with Image.open(io.BytesIO(content)) as im:
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        im.save(out_path, format="JPEG", quality=quality, optimize=True)

def stratified_split_move(files_by_class: Dict[str, List[Path]], raw_root: Path, train_ratio: float):
    """Move de data/raw/<c>/.. para data/{train,val}/<c>/.. (split estratificado por classe)."""
    assert 0.5 <= train_ratio < 1.0
    moved_any = False
    import hashlib as _hh
    for klass, files in files_by_class.items():
        if not files: 
            continue
        files_sorted = sorted(files, key=lambda p: _hh.sha1(p.name.encode()).hexdigest())
        n = len(files_sorted)
        k = int(round(n * train_ratio))
        train_files, val_files = files_sorted[:k], files_sorted[k:]
        for split, split_files in (("train", train_files), ("val", val_files)):
            out_dir = raw_root.parent / split / klass
            ensure_dir(out_dir)
            for src in split_files:
                dst = out_dir / src.name
                if src.exists() and not dst.exists():
                    src.replace(dst)
                    moved_any = True
    if moved_any:
        print(f"split feito em {raw_root.parent}/train e {raw_root.parent}/val")
    else:
        print("nada para mover no split (verifique se os arquivos existem).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True,
                    help='Ex.: "data/csv_prepared/*_prepared_mapped.csv"')
    ap.add_argument("--out-dir", default="data/raw",
                    help="Raiz onde salvar: data/raw/<classe>/<hash>.jpg")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", type=float, default=0.0,
                    help="0.0 = não dividir; ex.: 0.85 para criar data/train e data/val")
    args = ap.parse_args()

    # Resolve glob (compatível com poetry run)
    if any(ch in args.input_glob for ch in "*?[]"):
        csvs = sorted(Path().glob(args.input_glob))
    else:
        csvs = [Path(args.input_glob)]
    if not csvs:
        print(f"nenhum arquivo casa com {args.input_glob}")
        return

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    # Coleta (url, classe, filename_hash)
    rows: List[Tuple[str, str, str]] = []
    for p in csvs:
        df = pd.read_csv(p)
        # checagens mínimas
        need = {"image_url", "tipo"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} precisa das colunas {missing}")
        tipo_bin = df["tipo"].apply(map_tipo_binary).astype(str)
        for url, lab in zip(df["image_url"].astype(str), tipo_bin):
            fname = hashed_filename_from_url(url)  # sempre <hash>.jpg
            rows.append((url, lab, fname))

    # Baixa concorrente
    ok_files_by_class: Dict[str, List[Path]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2info = {}
        for url, lab, fname in rows:
            out_path = out_root / lab / fname
            ensure_dir(out_path.parent)
            if out_path.exists():
                ok_files_by_class.setdefault(lab, []).append(out_path)
                continue
            fut2info[ex.submit(fetch_image, url)] = (url, lab, out_path)

        for fut in tqdm(as_completed(fut2info), total=len(fut2info), desc="baixando"):
            url, lab, out_path = fut2info[fut]
            content = fut.result()
            if not content or not is_image_ok(content):
                continue
            try:
                save_jpeg(content, out_path)
                ok_files_by_class.setdefault(lab, []).append(out_path)
            except Exception:
                if out_path.exists():
                    try: out_path.unlink()
                    except Exception: 
                        pass

    # Split opcional
    if args.split and 0.5 <= args.split < 1.0:
        stratified_split_move(ok_files_by_class, out_root, args.split)
    else:
        print(f"imagens em {out_root}")

if __name__ == "__main__":
    main()
