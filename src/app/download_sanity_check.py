#!/usr/bin/env python3
# sanity_check_dataset.py
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd

CLASSES = ("hibrido", "nao_hibrido")

def sha1_name(url: str, n=16) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:n] + ".jpg"

def list_files(root: Path) -> dict[str, set[str]]:
    out = {c: set() for c in CLASSES}
    for c in CLASSES:
        d = root / c
        if d.exists():
            out[c] = {p.name for p in d.glob("*") if p.is_file()}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-glob", required=True, help='ex.: "data/csv_prepared/*_prepared_mapped.csv"')
    ap.add_argument("--data-root", default="data", help="raiz com raw/train/val")
    ap.add_argument("--save-details", action="store_true", help="salva CSVs de faltantes/sobrando")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    csv_paths = sorted(Path().glob(args.csv_glob))
    if not csv_paths:
        print(f"Nenhum CSV em {args.csv_glob}")
        return

    # 1) junta URLs e classes dos CSVs
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=["image_url", "tipo"])
        df["tipo"] = df["tipo"].astype(str).str.lower()
        df["tipo"] = df["tipo"].map(lambda s: "hibrido" if ("hybr" in s or s in {"h","hybrid","híbrido","hybrido"}) else "nao_hibrido")
        rows.append(df)
    full = pd.concat(rows, ignore_index=True).dropna(subset=["image_url", "tipo"])
    full["fname"] = full["image_url"].astype(str).map(sha1_name)

    # 2) conjunto esperado por classe
    expected = {c: set(full.loc[full["tipo"] == c, "fname"].tolist()) for c in CLASSES}

    # 3) varre splits no disco
    splits = ["raw", "train", "val"]
    present = {sp: list_files(data_root / sp) for sp in splits}

    # 4) resumo COMBINADO e por split (sem "missing" por split)
    present_all = {c: set() for c in CLASSES}
    for sp in splits:
        for c in CLASSES:
            present_all[c] |= present[sp][c]

    print("\n=== COMBINADO (raw ∪ train ∪ val) ===")
    for c in CLASSES:
        have = present_all[c]
        exp  = expected[c]
        ok   = len(have & exp)
        miss = len(exp - have)          # faltas reais
        extra= len(have - exp)          # sobras reais
        print(f"{c:12s} | ok:{ok:5d}  faltando:{miss:5d}  extra:{extra:5d}  (no disco:{len(have):5d})")

    print("\n=== DISTRIBUIÇÃO POR SPLIT (apenas contagem; 'missing' aqui não é significativo) ===")
    for sp in splits:
        print(f"[{sp}]")
        for c in CLASSES:
            have = present[sp][c]
            print(f"  {c:12s} | no disco:{len(have):5d}")

    # 5) (opcional) salvar DETALHES de faltas/sobras (COMBINADO) e extras por split
    if args.save_details:
        outdir = data_root / "_sanity_reports"
        outdir.mkdir(parents=True, exist_ok=True)

        # COMBINADO: faltas reais e sobras reais
        for c in CLASSES:
            have = present_all[c]
            exp  = expected[c]
            miss_list  = sorted(exp - have)   # arquivos esperados que não existem em NENHUM split
            extra_list = sorted(have - exp)   # arquivos no disco que não aparecem nos CSVs
            pd.DataFrame({"fname": miss_list}).to_csv(outdir / f"missing_combined_{c}.csv", index=False)
            pd.DataFrame({"fname": extra_list}).to_csv(outdir / f"extra_combined_{c}.csv", index=False)

        # Por split: salvar SOMENTE extras (útil para limpar arquivos órfãos em um split)
        for sp in splits:
            for c in CLASSES:
                have = present[sp][c]
                exp  = expected[c]
                extra_list = sorted(have - exp)
                pd.DataFrame({"fname": extra_list}).to_csv(outdir / f"extra_{sp}_{c}.csv", index=False)

        # mapa url->fname para auditoria
        full[["image_url","tipo","fname"]].drop_duplicates().to_csv(outdir / "url_to_fname.csv", index=False)
        print(f"\nRelatórios detalhados em: {outdir}")

if __name__ == "__main__":
    main()
