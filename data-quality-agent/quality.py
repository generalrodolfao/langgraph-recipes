"""
Agentes de qualidade — um por dimensão.
Cada função retorna um dict com: status, score (0-20), problemas, detalhes.
"""

from datetime import date
from typing import Any, Dict

import numpy as np
import pandas as pd


def _serializar(obj: Any) -> Any:
    """Converte numpy types para Python nativo (msgpack-safe)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _serializar(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializar(v) for v in obj]
    return obj


def auditar_completude(df: pd.DataFrame) -> Dict[str, Any]:
    """Audita % de nulos por coluna."""
    total = len(df)
    nulos = df.isnull().sum()
    pct_nulos = (nulos / total * 100).round(2)

    problematicas = pct_nulos[pct_nulos > 5]
    colunas_afetadas = len(problematicas)

    score = max(0, 20 - colunas_afetadas * 5)

    return _serializar({
        "dimensao": "completude",
        "status": "ok" if colunas_afetadas == 0 else ("warning" if colunas_afetadas <= 2 else "critico"),
        "score": score,
        "colunas_com_nulos": nulos[nulos > 0].to_dict(),
        "colunas_criticas": problematicas.to_dict(),
        "detalhes": f"{colunas_afetadas} colunas com >5% nulos" if colunas_afetadas > 0 else "Todas as colunas OK",
    })


def auditar_frescor(df: pd.DataFrame, coluna_data: str = "data_atualizacao", limite_dias: int = 730) -> Dict[str, Any]:
    """Audita frescor: % de registros sem atualização recente."""
    hoje = pd.Timestamp(date.today())
    datas = pd.to_datetime(df[coluna_data])
    desatualizados = datas < (hoje - pd.Timedelta(days=limite_dias))
    pct = round(desatualizados.sum() / len(df) * 100, 2)

    if pct <= 5:
        status, score = "ok", 20
    elif pct <= 15:
        status, score = "warning", 15
    else:
        status, score = "critico", 10

    return _serializar({
        "dimensao": "frescor",
        "status": status,
        "score": score,
        "registros_desatualizados": int(desatualizados.sum()),
        "pct_desatualizado": pct,
        "data_mais_antiga": str(datas.min().date()) if len(datas) > 0 else None,
        "detalhes": f"{pct}% dos registros sem atualização há >{limite_dias // 365} anos",
    })


def auditar_unicidade(df: pd.DataFrame, colunas_chave: list = None) -> Dict[str, Any]:
    """Audita duplicatas por colunas-chave."""
    if colunas_chave is None:
        colunas_chave = ["cpf"]

    problemas = []
    for col in colunas_chave:
        if col in df.columns:
            dups = df[col].duplicated().sum()
            if dups > 0:
                problemas.append({"coluna": col, "duplicatas": int(dups)})

    total_dups = sum(p["duplicatas"] for p in problemas)

    if total_dups == 0:
        status, score = "ok", 20
    elif total_dups <= 2:
        status, score = "warning", 15
    else:
        status, score = "critico", 5

    return _serializar({
        "dimensao": "unicidade",
        "status": status,
        "score": score,
        "problemas": problemas,
        "total_duplicatas": total_dups,
        "detalhes": f"{total_dups} duplicatas encontradas" if total_dups > 0 else "Sem duplicatas",
    })


def auditar_consistencia(df: pd.DataFrame) -> Dict[str, Any]:
    """Audita regras de negócio: idade, email, datas."""
    violacoes = []

    # Idade negativa
    if "idade" in df.columns:
        idade_neg = (df["idade"] < 0).sum()
        if idade_neg > 0:
            violacoes.append({"regra": "idade >= 0", "violacoes": int(idade_neg)})

    # Email sem @
    if "email" in df.columns:
        mask = df["email"].notna()
        emails_invalidos = mask & ~df.loc[mask, "email"].str.contains("@")
        inv = emails_invalidos.sum()
        if inv > 0:
            violacoes.append({"regra": "email contém @", "violacoes": int(inv)})

    # Data de nascimento < data de cadastro
    if "data_nascimento" in df.columns and "data_cadastro" in df.columns:
        nasc = pd.to_datetime(df["data_nascimento"])
        cad = pd.to_datetime(df["data_cadastro"])
        datas_inv = (nasc > cad).sum()
        if datas_inv > 0:
            violacoes.append({"regra": "data_nascimento <= data_cadastro", "violacoes": int(datas_inv)})

    total = sum(v["violacoes"] for v in violacoes)

    if total == 0:
        status, score = "ok", 20
    elif total <= 5:
        status, score = "warning", 12
    else:
        status, score = "critico", 5

    return _serializar({
        "dimensao": "consistencia",
        "status": status,
        "score": score,
        "violacoes": violacoes,
        "total_violacoes": total,
        "detalhes": f"{total} violações em {len(violacoes)} regras" if total > 0 else "Todas as regras OK",
    })


def auditar_acuracia(df: pd.DataFrame, colunas_numericas: list = None) -> Dict[str, Any]:
    """Audita outliers por z-score (|z| > 3)."""
    if colunas_numericas is None:
        colunas_numericas = ["renda", "compras_12m"]

    outliers = []
    for col in colunas_numericas:
        if col not in df.columns or df[col].dtype not in ["int64", "float64"]:
            continue
        serie = df[col].dropna()
        if len(serie) == 0:
            continue
        z = np.abs((serie - serie.mean()) / serie.std())
        n_out = (z > 3).sum()
        if n_out > 0:
            outliers.append({"coluna": col, "outliers": int(n_out), "pct": round(n_out / len(serie) * 100, 2)})

    total = sum(o["outliers"] for o in outliers)

    if total == 0:
        status, score = "ok", 20
    elif total <= 5:
        status, score = "warning", 12
    else:
        status, score = "critico", 5

    return _serializar({
        "dimensao": "acuracia",
        "status": status,
        "score": score,
        "outliers": outliers,
        "total_outliers": total,
        "detalhes": f"{total} outliers em {len(outliers)} colunas" if total > 0 else "Sem outliers",
    })


def gerar_diagnostico(resultados: list, threshold: float = 0.0) -> Dict[str, Any]:
    """Consolida resultados dos 5 agentes em diagnóstico unificado."""
    scores = [r["score"] for r in resultados]
    score_total = sum(scores)

    criticos = [r for r in resultados if r["status"] == "critico"]
    warnings = [r for r in resultados if r["status"] == "warning"]

    if len(criticos) >= 2:
        severidade = "ALTA"
    elif len(criticos) >= 1 or len(warnings) >= 3:
        severidade = "MÉDIA"
    elif len(warnings) >= 1:
        severidade = "BAIXA"
    else:
        severidade = "OK"

    alertas = []
    for r in criticos:
        alertas.append({"nivel": "CRITICO", "dimensao": r["dimensao"], "mensagem": r["detalhes"]})
    for r in warnings:
        alertas.append({"nivel": "WARNING", "dimensao": r["dimensao"], "mensagem": r["detalhes"]})

    return _serializar({
        "score_qualidade": score_total,
        "score_maximo": 100,
        "severidade": severidade,
        "dimensoes_ok": sum(1 for r in resultados if r["status"] == "ok"),
        "dimensoes_warning": len(warnings),
        "dimensoes_criticas": len(criticos),
        "alertas": alertas,
        "resultados_por_dimensao": {r["dimensao"]: r for r in resultados},
    })
