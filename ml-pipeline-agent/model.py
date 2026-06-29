"""
Lógica de ML — funções puras chamadas pelos nós do agente.
"""

import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

COLS_FEATURES = [
    "area_m2", "funcionarios",
    "ticket_medio", "n_clientes", "custo_pct", "margem",
    "crescimento_3m", "crescimento_6m",
    "ticket_3m_avg", "clientes_3m_avg",
    "sazonalidade_mes", "dia_semana_pico",
    "share_eletronicos", "share_vestuario", "share_alimentos",
    "share_moveis", "share_esportes", "share_livros",
]

COLS_CATEGORICAS = ["regiao", "porte", "categoria_principal"]

MODELOS_BASE = {
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1),
    "LinearRegression": LinearRegression(),
}


def carregar_dados(caminho_parquet: str) -> pd.DataFrame:
    return pd.read_parquet(caminho_parquet)


def preparar_features(df: pd.DataFrame, target_col: str = "target", loja_id: Optional[int] = None) -> tuple:
    """Prepara X e y, com encoding de categóricas e opção de filtrar loja."""
    if loja_id is not None:
        df = df[df["loja_id"] == loja_id]

    df = df.copy()

    for col in COLS_CATEGORICAS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)

    feature_cols = COLS_FEATURES + [
        c for c in df.columns
        if c.startswith("regiao_") or c.startswith("porte_") or c.startswith("categoria_principal_")
    ]

    X = df[feature_cols].values
    y = df[target_col].values

    return X, y, feature_cols


def split_dados(X: np.ndarray, y: np.ndarray) -> dict:
    """Split treino/teste 80/20."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "n_train": len(X_train), "n_test": len(X_test),
    }


def treinar_modelo(nome: str, X_train: np.ndarray, y_train: np.ndarray):
    """Treina um modelo pelo nome."""
    modelo = MODELOS_BASE[nome]
    modelo.fit(X_train, y_train)
    return modelo


def avaliar_modelo(modelo, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Retorna métricas de avaliação."""
    y_pred = modelo.predict(X_test)
    return {
        "r2": round(r2_score(y_test, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "n_amostras": len(y_test),
    }


def selecionar_melhor(resultados: dict) -> tuple:
    """Seleciona o melhor modelo por R²."""
    melhor_nome = max(resultados, key=lambda k: resultados[k]["r2"])
    return melhor_nome, resultados[melhor_nome]


def salvar_modelo(modelo, caminho: str):
    with open(caminho, "wb") as f:
        pickle.dump(modelo, f)


def carregar_modelo(caminho: str):
    with open(caminho, "rb") as f:
        return pickle.load(f)
