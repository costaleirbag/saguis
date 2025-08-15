# train_tipo_geo.py
# -*- coding: utf-8 -*-
"""
Classificador de Tipo (H vs N-H) usando data, latitude, longitude e estado.

Requisitos:
  - pandas, numpy, scikit-learn, matplotlib
  - xgboost (opcional; se não houver, usa RandomForest)

Execução:
  python train_tipo_geo.py \
    --aurita data/csv/aurita_location_fixed.csv \
    --penicillata data/csv/penicillata_location_fixed.csv \
    --jacchus data/csv/jacchus_location_fixed.csv \
    --test_size 0.2 \
    --seed 42
"""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(paths):
    """Lê e concatena CSVs esperados com colunas:
    observed_on,latitude,longitude,place_county_name,place_state_name,url,image_url,Tipo
    """
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")
        df = pd.read_csv(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Converter data (dd/mm/yyyy)
    df["observed_on"] = pd.to_datetime(df["observed_on"], dayfirst=True, errors="coerce")

    # Remover linhas sem data ou coordenadas essenciais
    df = df.dropna(subset=["observed_on", "latitude", "longitude", "place_state_name", "Tipo"])

    # Extrair componentes temporais
    df["month"] = df["observed_on"].dt.month
    df["dayofyear"] = df["observed_on"].dt.dayofyear
    df["year"] = df["observed_on"].dt.year

    # Codificação cíclica de mês e dia-do-ano
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    # Aproximação simples: 365 dias (ignora bissexto — suficiente aqui)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)

    # Mapeia alvo para 0/1
    # N-H -> 0 | H -> 1
    df["target"] = df["Tipo"].map({"N-H": 0, "H": 1})

    # Checagens
    if df["target"].isna().any():
        vals = df.loc[df["target"].isna(), "Tipo"].unique()
        raise ValueError(f"Valores inesperados em 'Tipo': {vals}. Esperado apenas 'H' ou 'N-H'.")

    return df


def get_model(random_state: int = 42):
    # Tenta XGBoost
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
        )
        model_name = "XGBoost"
        return model_name, model
    except Exception:
        pass

    # Fallback: RandomForest
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    model_name = "RandomForest"
    return model_name, model


def build_pipeline(model):
    # Colunas
    categorical = ["place_state_name"]
    numeric_passthrough = ["latitude", "longitude", "year", "month_sin", "month_cos", "day_sin", "day_cos"]

    # Pré-processador
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", StandardScaler(), numeric_passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
    return pipe, categorical + numeric_passthrough


def evaluate(pipe, X_train, y_train, X_test, y_test, model_name: str):
    # Cross-val ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=None)
    print(f"\n[{model_name}] ROC-AUC CV (5 folds): mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit final
    pipe.fit(X_train, y_train)

    # Avaliação holdout
    y_pred = pipe.predict(X_test)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    else:
        # Alguns modelos podem não ter predict_proba
        # Usa decisão bruta e aplica ranking para AUC aproximado
        if hasattr(pipe.named_steps["clf"], "decision_function"):
            y_raw = pipe.decision_function(X_test)
            # Normaliza para [0,1]
            y_prob = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)
        else:
            # Como último recurso, usa a própria predição binária
            y_prob = y_pred.astype(float)

    print("\n==== Classification report (holdout) ====")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (holdout):")
    print(cm)

    try:
        roc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        print(f"ROC-AUC (holdout): {roc:.3f}")
        print(f"Average Precision (PR AUC) (holdout): {ap:.3f}")
    except Exception:
        pass

    return pipe


def plot_feature_importance(pipe, feature_input_names):
    """Extrai nomes de features expandidos e plota importâncias se o modelo suportar."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        # Fallback simples
        feature_names = np.array(feature_input_names)

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "get_booster"):  # xgboost
        try:
            # Algumas versões fornecem por ganho/peso; usamos 'gain' se disponível
            booster = clf.get_booster()
            score_dict = booster.get_score(importance_type="gain")
            # Precisamos alinhar com feature_names; em XGB as chaves vêm como f0, f1, ...
            # Então tentamos mapear pelo índice
            importances = np.zeros(len(feature_names))
            for k, v in score_dict.items():
                if k.startswith("f"):
                    idx = int(k[1:])
                    if idx < len(importances):
                        importances[idx] = v
        except Exception:
            importances = None

    if importances is None:
        print("\n[Info] O modelo não disponibiliza importâncias de features de forma direta.")
        return

    order = np.argsort(importances)[::-1]
    top_k = min(30, len(order))

    plt.figure(figsize=(10, 7))
    plt.bar(range(top_k), importances[order][:top_k])
    plt.xticks(range(top_k), feature_names[order][:top_k], rotation=90)
    plt.title("Importância de Features (Top 30)")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aurita", required=True, help="Caminho para aurita_location_fixed.csv")
    parser.add_argument("--penicillata", required=True, help="Caminho para penicillata_location_fixed.csv")
    parser.add_argument("--jacchus", required=True, help="Caminho para jacchus_location_fixed.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporção do conjunto de teste")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória")
    args = parser.parse_args()

    # 1) Carregar e preparar
    df_raw = load_data([args.aurita, args.penicillata, args.jacchus])
    df = feature_engineering(df_raw)

    # 2) Selecionar colunas e split
    features = ["latitude", "longitude", "place_state_name", "year", "month_sin", "month_cos", "day_sin", "day_cos"]
    X = df[features].copy()
    y = df["target"].astype(int).copy()

    # Checagem de classes
    cls_counts = y.value_counts().sort_index()
    print("Distribuição do alvo (0=N-H, 1=H):")
    print(cls_counts.to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # 3) Modelo + pipeline
    model_name, model = get_model(random_state=args.seed)
    pipe, feature_input_names = build_pipeline(model)

    # 4) Treino e avaliação
    fitted_pipe = evaluate(pipe, X_train, y_train, X_test, y_test, model_name=model_name)

    # 5) Importância de features
    plot_feature_importance(fitted_pipe, feature_input_names)

    print("\nConcluído com sucesso.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.", file=sys.stderr)
        sys.exit(130)
