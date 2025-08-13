#!/usr/bin/env python3
"""
Baixa imagens a partir de CSV(s) com colunas: url, classe,
organizando por ORIGEM (aurita/penicillata/jacchus) e rótulo binário (hibrido/nao_hibrido).

Estrutura gerada:
- data/raw_sources/<source>/<hibrido|nao_hibrido>/<hash>.jpg
- (opcional) consolidação única em data/raw/<hibrido|nao_hibrido>/<hash>.jpg
  via hardlink (padrão), symlink ou cópia.

Também grava:
- data/_download_ok.csv (por execução)
- data/_download_errors.csv
- data/_index.csv (índice acumulado de tudo que foi baixado)

Uso (um CSV por vez):
  poetry run saguis-download \
    --csv data/csv/raw_aurita.csv \
    --source aurita \
    --out_sources data/raw_sources \
    --consolidate_dir data/raw \
    --workers 8 \
    --split_after_download 0.85 \
    --link_mode hard

Você pode chamar 3x, um para cada CSV de origem.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
from PIL import Image
from tqdm import tqdm

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/119.0 Safari/537.36")

SAVE_EXT = ".jpg"

# ---------------- utils ---------------- #

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = s.replace('"', '').replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]+", "", s)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_csv_rows(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return rows
        header_like = False
        if len(first) >= 2:
            h0, h1 = first[0].strip().lower(), first[1].strip().lower()
            header_like = ("url" in h0 and "class" in h1) or (h0 == "url" and h1 in ("classe", "class"))
        if not header_like:
            if len(first) >= 2 and first[0].strip():
                rows.append((first[0].strip(), first[1].strip()))
        for r in reader:
            if len(r) < 2: continue
            url, klass = r[0].strip(), r[1].strip()
            if url: rows.append((url, klass))
    return rows

def map_to_binary(label: str, hybrid_regex: Optional[str]) -> str:
    s = label.strip().lower()

    # Mapeamento direto dos códigos do CSV
    if s in ("h", "hybrido", "híbrido", "hybrid"):
        return "hibrido"
    if s in ("n-h", "não-hibrido", "nao-hibrido", "non-hybrid"):
        return "nao_hibrido"

    # fallback: regex ou heurística
    if hybrid_regex and re.search(hybrid_regex, s, flags=re.IGNORECASE):
        return "hibrido"
    if any(k in s for k in ("hybrid", "híbrido", "hibrido", "hìbrido", "hýbrido")):
        return "hibrido"
    return "nao_hibrido"

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

def hash_name(*parts: str) -> str:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{h}{SAVE_EXT}"

def hardlink_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists(): return
    ensure_dir(dst.parent)
    if mode == "hard":
        os.link(src, dst)
    elif mode == "soft":
        dst.symlink_to(src)
    else:
        # copy
        from shutil import copy2
        copy2(src, dst)

# ---------------- split ---------------- #

def stratified_move(files_by_class: Dict[str, List[Path]], data_root: Path, train_ratio: float):
    assert 0.5 <= train_ratio < 1.0
    for klass, files in files_by_class.items():
        if not files: continue
        files_sorted = sorted(files, key=lambda p: hashlib.sha1(p.name.encode()).hexdigest())
        n = len(files_sorted)
        k = int(round(n * train_ratio))
        train_files, val_files = files_sorted[:k], files_sorted[k:]
        for split, split_files in (("train", train_files), ("val", val_files)):
            out_dir = data_root / split / klass
            ensure_dir(out_dir)
            for src in split_files:
                dst = out_dir / src.name
                if not dst.exists():
                    src.replace(dst)

# ---------------- main ---------------- #

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--csv", required=True, help="CSV com colunas: url,classe")
    pa.add_argument("--source", default=None, help="tag da origem (ex.: aurita). Se vazio, usa o stem do CSV.")
    pa.add_argument("--out_sources", default="data/raw_sources", help="raiz das pastas por origem")
    pa.add_argument("--consolidate_dir", default="data/raw", help="raiz do dataset consolidado binário")
    pa.add_argument("--link_mode", choices=["hard", "soft", "copy"], default="hard",
                    help="como consolidar em data/raw: hardlink (padrão), symlink (soft) ou copiar")
    pa.add_argument("--workers", type=int, default=8)
    pa.add_argument("--per_class_limit", type=int, default=0)
    pa.add_argument("--dry-run", action="store_true")
    pa.add_argument("--hybrid_regex", type=str, default=None)
    pa.add_argument("--split_after_download", type=float, default=0.0,
                    help="se >0, move de consolidate_dir/raw -> train/val (ex.: 0.85)")
    args = pa.parse_args()

    csv_path = Path(args.csv)
    source = slugify(args.source or csv_path.stem.replace("raw_", ""))

    out_sources = Path(args.out_sources)
    consolidate_dir = Path(args.consolidate_dir)
    ensure_dir(out_sources)
    ensure_dir(consolidate_dir)

    rows = read_csv_rows(csv_path)
    if not rows:
        print("CSV vazio ou inválido.")
        sys.exit(1)

    # mapeia para binário
    bin_rows: List[Tuple[str, str, str]] = []  # (url, original_label, bin_label)
    for url, klass in rows:
        bin_label = map_to_binary(klass, args.hybrid_regex)
        bin_rows.append((url, klass, bin_label))

    # aplica limite por classe binária (se houver)
    if args.per_class_limit > 0:
        kept: Dict[str, int] = {}
        limited: List[Tuple[str, str, str]] = []
        for url, orig, binlab in bin_rows:
            if kept.get(binlab, 0) < args.per_class_limit:
                limited.append((url, orig, binlab))
                kept[binlab] = kept.get(binlab, 0) + 1
        bin_rows = limited

    classes = sorted({b for _, _, b in bin_rows})
    if args.dry_run:
        print(f"[dry-run] source={source} | {len(bin_rows)} imagens | classes={classes}")
        sys.exit(0)

    # garante pastas de origem
    for c in classes:
        ensure_dir(out_sources / source / c)

    ok_rows: List[Tuple[str, str, str, str, str]] = []  # url, source, orig_label, bin_label, path_source
    err_rows: List[Tuple[str, str, str]] = []           # url, source, orig_label

    # download concorrente
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for url, orig_label, binlab in bin_rows:
            name = hash_name(url, source, binlab)
            out_path = out_sources / source / binlab / name
            if out_path.exists():
                ok_rows.append((url, source, orig_label, binlab, str(out_path)))
                continue
            futures[ex.submit(fetch_image, url)] = (url, orig_label, binlab, out_path)

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Baixando [{source}]"):
            url, orig_label, binlab, out_path = futures[fut]
            content = fut.result()
            if not content or not is_image_ok(content):
                err_rows.append((url, source, orig_label))
                continue
            try:
                save_jpeg(content, out_path)
                ok_rows.append((url, source, orig_label, binlab, str(out_path)))
            except Exception:
                if out_path.exists():
                    try: out_path.unlink()
                    except Exception: pass
                err_rows.append((url, source, orig_label))

    # relatórios desta execução
    ok_csv = out_sources.parent / "_download_ok.csv"
    err_csv = out_sources.parent / "_download_errors.csv"
    with ok_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(["url","source","original_label","binary_label","path_source"])
        for r in ok_rows: w.writerow(r)
    with err_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(["url","source","original_label"])
        for r in err_rows: w.writerow(r)

    print(f"✅ {len(ok_rows)} baixadas | ❌ {len(err_rows)} erros | source={source}")

    # consolidação (hardlink/symlink/cópia) em data/raw
    index_rows: List[Tuple[str,str,str,str,str]] = []  # source, original_label, binary_label, path_source, path_consolidated
    for url, src, orig, binlab, psrc in ok_rows:
        name = Path(psrc).name
        pdst = consolidate_dir / binlab / name
        hardlink_or_copy(Path(psrc), pdst, mode=args.link_mode)
        index_rows.append((src, orig, binlab, psrc, str(pdst)))

    # índice acumulado
    index_csv = consolidate_dir.parent / "_index.csv"
    with index_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["source","original_label","binary_label","path_source","path_consolidated"])
        for r in index_rows: w.writerow(r)

    # split opcional no dataset consolidado
    if args.split_after_download and 0.5 <= args.split_after_download < 1.0:
        files_by_class: Dict[str, List[Path]] = {}
        for _, _, binlab, _, pdst in index_rows:
            files_by_class.setdefault(binlab, []).append(Path(pdst))
        stratified_move(files_by_class, consolidate_dir.parent, args.split_after_download)
        print(f"📁 Split criado em {consolidate_dir.parent}/train e {consolidate_dir.parent}/val")

if __name__ == "__main__":
    main()
