#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepara *_location_fixed.csv e mapeia URLs para arquivos locais via data/_download_ok.csv.

Pipeline:
1) Lê um glob de CSVs de features (ex.: data/csv/*_location_fixed.csv).
2) Valida/normaliza colunas básicas e tipos, corrige datas e lat/lon.
3) Cria features temporais (ano/mês/doy e sen/cos opcionais).
4) (Opcional) cria clusters geográficos KMeans em lat/lon.
5) Lê data/_download_ok.csv para mapear url -> basename (<hash>.jpg) e reporta taxa de match.
6) (Opcional) descarta linhas sem match (--drop-unmatched).
7) Salva em data/csv_prepared/<nome>_prepared_mapped.csv com a coluna 'basename'
   (usada no treino com fusão imagem+tabular).
"""

from __future__ import annotations
import argparse
from glob import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


REQUIRED_COLS = [
    "observed_on",
    "latitude",
    "longitude",
    "place_state_name",
    "image_url",
    "tipo",
]
OPTIONAL_COLS = ["place_county_name"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", required=True, help='ex.: "data/csv/*_location_fixed.csv"')
    ap.add_argument("--output-dir", default="data/csv_prepared")
    ap.add_argument("--download-ok", default="data/_download_ok.csv",
                    help="CSV com colunas [url, path_source] (ou path_consolidated).")
    ap.add_argument("--geo-clusters", type=int, default=0,
                    help="K de clusters geográficos em lat/lon (0 = desliga).")
    ap.add_argument("--keep-county", action="store_true",
                    help="Mantém place_county_name no CSV final (default: remove).")
    ap.add_argument("--no-cyc", action="store_true",
                    help="Não salva month_sin/month_cos/doy_sin/doy_cos (default: salva).")
    ap.add_argument("--drop-unmatched", action="store_true",
                    help="Descarta linhas sem mapeamento url->basename (recomendado p/ treino).")
    return ap.parse_args()


def _coerce_date_ddmmyyyy(s) -> pd.Timestamp | pd.NaT:
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _clean_latlon(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[~df["latitude"].between(-90, 90), "latitude"] = np.nan
    df.loc[~df["longitude"].between(-180, 180), "longitude"] = np.nan
    return df.dropna(subset=["latitude", "longitude"])


def _engineer_time(df: pd.DataFrame, add_cyclic: bool = True) -> pd.DataFrame:
    dt = df["observed_on"].apply(_coerce_date_ddmmyyyy)
    df["observed_on"] = dt.dt.strftime("%d/%m/%Y")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    if add_cyclic:
        two_pi = 2 * np.pi
        df["month_sin"] = np.sin(two_pi * (df["month"].fillna(0) / 12))
        df["month_cos"] = np.cos(two_pi * (df["month"].fillna(0) / 12))
        df["doy_sin"]   = np.sin(two_pi * (df["dayofyear"].fillna(0) / 366))
        df["doy_cos"]   = np.cos(two_pi * (df["dayofyear"].fillna(0) / 366))
    return df


def _geo_cluster(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or not SKLEARN_OK or len(df) < k:
        return df
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    coords = df[["latitude", "longitude"]].to_numpy()
    df["geo_cluster"] = km.fit_predict(coords).astype(int)
    return df


def _ensure_required_cols(df: pd.DataFrame, path: Path):
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"{path.name}: faltam colunas obrigatórias: {miss}")


def _map_tipo_binary(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("h", "hybrido", "híbrido", "hybrid"): return "hibrido"
    if s in ("n-h", "não-hibrido", "nao-hibrido", "non-hybrid", "nh"): return "nao_hibrido"
    if "hybr" in s or "hí" in s or "hibr" in s: return "hibrido"
    return "nao_hibrido"


def _read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Lê CSV com autodetecção de separador; faz fallback para ',' e ';'.
    Evita ParserError em arquivos com delimitadores inconsistentes.
    """
    # 1) tentar autodetecção
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        pass
    # 2) tentar vírgula
    try:
        return pd.read_csv(path, sep=",", engine="python", encoding="utf-8")
    except Exception:
        pass
    # 3) tentar ponto-e-vírgula
    try:
        return pd.read_csv(path, sep=";", engine="python", encoding="utf-8")
    except Exception as e:
        raise e


def _load_download_ok(path: Path) -> dict[str, str]:
    """
    Lê data/_download_ok.csv e devolve dict url->basename(<hash>.jpg).
    Aceita colunas: url, path_source (ou path_consolidated).
    Tolerante a separador (',' ou ';').
    """
    if not path.exists():
        print(f"[warn] {path} não encontrado; match poderá ser baixo.")
        return {}
    ok = _read_csv_safely(path)

    # Normaliza nomes de colunas (tira espaços/maiusc)
    ok.columns = [str(c).strip().lower() for c in ok.columns]

    # nomes aceitos
    url_col = "url"
    path_col = None
    for candidate in ("path_source", "path_consolidated"):
        if candidate in ok.columns:
            path_col = candidate
            break
    if url_col not in ok.columns or path_col is None:
        print(f"[warn] {path} sem colunas esperadas ['url','path_source'|'path_consolidated']; "
              f"colunas lidas: {list(ok.columns)}")
        return {}

    url2base = {}
    for u, p in zip(ok[url_col].astype(str), ok[path_col].astype(str)):
        if pd.notna(u) and pd.notna(p) and len(str(p).strip()) > 0:
            url2base[str(u)] = Path(str(p)).name
    print(f"[info] download_ok: {len(url2base)} URLs mapeadas para basenames.")
    return url2base


def process_file(in_path: Path, out_dir: Path, url2base: dict[str, str],
                 keep_county: bool, geo_k: int, add_cyc: bool, drop_unmatched: bool) -> Path:
    df = _read_csv_safely(in_path)
    _ensure_required_cols(df, in_path)

    # tipagem básica
    df["place_state_name"] = df["place_state_name"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["image_url"] = df["image_url"].astype(str).str.strip()
    if "place_county_name" in df.columns:
        df["place_county_name"] = df["place_county_name"].astype(str).str.strip()

    # limpeza
    n0 = len(df)
    df = _clean_latlon(df)
    df = _engineer_time(df, add_cyclic=add_cyc)
    df = _geo_cluster(df, geo_k)
    df["binary_label"] = df["tipo"].apply(_map_tipo_binary)

    # mapeia url -> basename(hash).jpg
    df["basename"] = df["url"].map(url2base)
    matched = df["basename"].notna().sum()
    match_rate = (matched / len(df)) if len(df) else 0.0

    print(f"✔ {in_path.name}: {len(df)}/{n0} após limpeza | match url→basename: {matched}/{len(df)} ({match_rate:.1%})")

    # opcional: filtra fora sem match
    if drop_unmatched:
        df = df[df["basename"].notna()].copy()

    # colunas de saída
    keep_cols = [
        "observed_on", "latitude", "longitude",
        "place_state_name",
        "url", "image_url", "basename",
        "tipo", "binary_label",
        "year", "month", "dayofyear",
    ]
    if add_cyc:
        keep_cols += ["month_sin", "month_cos", "doy_sin", "doy_cos"]
    if "geo_cluster" in df.columns:
        keep_cols += ["geo_cluster"]
    if keep_county and "place_county_name" in df.columns:
        keep_cols.insert(4, "place_county_name")

    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}_prepared_mapped.csv"
    df_out.to_csv(out_path, index=False)
    return out_path, matched, len(df)


def main():
    args = parse_args()
    files = sorted(glob(args.input_glob))
    if not files:
        print(f"nenhum arquivo casa com {args.input_glob}")
        return

    out_dir = Path(args.output_dir)
    url2base = _load_download_ok(Path(args.download_ok))

    total_rows, total_matched = 0, 0
    outputs = []
    for f in files:
        out_path, matched, rows = process_file(
            Path(f), out_dir, url2base,
            keep_county=args.keep_county,
            geo_k=args.geo_clusters,
            add_cyc=(not args.no_cyc),
            drop_unmatched=args.drop_unmatched,
        )
        outputs.append(out_path)
        total_rows += rows
        total_matched += matched

    if total_rows > 0:
        print(f"\nResumo global: match url→basename = {total_matched}/{total_rows} ({total_matched/total_rows:.1%})")

    print("\nArquivos gerados:")
    for p in outputs:
        print(" -", p)


if __name__ == "__main__":
    main()
