"""
Data Quality Agent — Squad de agentes LangGraph para auditoria de qualidade de dados.

5 agentes especialistas executam em paralelo, convergem em diagnóstico unificado.

Uso:
    python agent.py
    python agent.py --input data/clientes.parquet --threshold 0.05
"""

import argparse
import os
import sys
from datetime import datetime
from typing import TypedDict

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from quality import (
    auditar_acuracia,
    auditar_completude,
    auditar_consistencia,
    auditar_frescor,
    auditar_unicidade,
    gerar_diagnostico,
)


class EstadoAgent(TypedDict):
    input_path: str
    threshold: float
    n_linhas: int
    n_colunas: int
    resultado_completude: dict
    resultado_frescor: dict
    resultado_unicidade: dict
    resultado_consistencia: dict
    resultado_acuracia: dict
    diagnostico: dict
    status: str


_runtime: dict = {}


def no_load(state: EstadoAgent) -> EstadoAgent:
    print("[load] Carregando dados...")
    df = pd.read_parquet(state["input_path"])
    _runtime["df"] = df

    nulos = df.isnull().sum().sum()
    print(f"[load] {len(df):,} linhas, {len(df.columns)} colunas, {nulos} nulos detectados")

    return {**state, "n_linhas": len(df), "n_colunas": len(df.columns), "status": "load_ok"}


def auditor_completude(state: EstadoAgent) -> dict:
    print("[completude] Auditando nulos...")
    df = _runtime["df"]
    resultado = auditar_completude(df)
    pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    print(f"[completude] {resultado['detalhes']} ({pct:.1f}% células nulas)")

    return {"resultado_completude": resultado}


def auditor_frescor(state: EstadoAgent) -> dict:
    print("[frescor] Auditando atualização...")
    df = _runtime["df"]
    resultado = auditar_frescor(df)
    print(f"[frescor] {resultado['detalhes']}")

    return {"resultado_frescor": resultado}


def auditor_unicidade(state: EstadoAgent) -> dict:
    print("[unicidade] Auditando duplicatas...")
    df = _runtime["df"]
    resultado = auditar_unicidade(df)
    print(f"[unicidade] {resultado['detalhes']}")

    return {"resultado_unicidade": resultado}


def auditor_consistencia(state: EstadoAgent) -> dict:
    print("[consistencia] Auditando regras de negócio...")
    df = _runtime["df"]
    resultado = auditar_consistencia(df)
    print(f"[consistencia] {resultado['detalhes']}")

    return {"resultado_consistencia": resultado}


def auditor_acuracia(state: EstadoAgent) -> dict:
    print("[acuracia] Auditando outliers...")
    df = _runtime["df"]
    resultado = auditar_acuracia(df)
    print(f"[acuracia] {resultado['detalhes']}")

    return {"resultado_acuracia": resultado}


def no_diagnostico(state: EstadoAgent) -> EstadoAgent:
    print("[diagnostico] Consolidando resultados...")
    resultados = [
        state["resultado_completude"],
        state["resultado_frescor"],
        state["resultado_unicidade"],
        state["resultado_consistencia"],
        state["resultado_acuracia"],
    ]

    diagnostico = gerar_diagnostico(resultados, state["threshold"])
    print(f"[diagnostico] Score: {diagnostico['score_qualidade']}/100 — Severidade: {diagnostico['severidade']}")

    return {**state, "diagnostico": diagnostico, "status": "done"}


def construir_grafo() -> StateGraph:
    builder = StateGraph(EstadoAgent)

    builder.add_node("load", no_load)
    builder.add_node("completude", auditor_completude)
    builder.add_node("frescor", auditor_frescor)
    builder.add_node("unicidade", auditor_unicidade)
    builder.add_node("consistencia", auditor_consistencia)
    builder.add_node("acuracia", auditor_acuracia)
    builder.add_node("diagnostico", no_diagnostico)

    builder.set_entry_point("load")

    # Fan-out: load → todos os auditores em paralelo
    builder.add_edge("load", "completude")
    builder.add_edge("load", "frescor")
    builder.add_edge("load", "unicidade")
    builder.add_edge("load", "consistencia")
    builder.add_edge("load", "acuracia")

    # Fan-in: todos os auditores → diagnostico
    builder.add_edge("completude", "diagnostico")
    builder.add_edge("frescor", "diagnostico")
    builder.add_edge("unicidade", "diagnostico")
    builder.add_edge("consistencia", "diagnostico")
    builder.add_edge("acuracia", "diagnostico")

    builder.add_edge("diagnostico", END)

    return builder


def executar_pipeline(input_path: str, threshold: float = 0.0):
    if not os.path.exists(input_path):
        print(f"Erro: arquivo '{input_path}' não encontrado. Execute: python data/generate.py")
        sys.exit(1)

    print(f"=== Data Quality Audit ===")
    print(f"Input:  {input_path}")
    print(f"Início: {datetime.now().strftime('%H:%M:%S')}")
    print()

    inicio = datetime.now()

    builder = construir_grafo()
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    estado_inicial: EstadoAgent = {
        "input_path": input_path,
        "threshold": threshold,
        "n_linhas": 0,
        "n_colunas": 0,
        "resultado_completude": {},
        "resultado_frescor": {},
        "resultado_unicidade": {},
        "resultado_consistencia": {},
        "resultado_acuracia": {},
        "diagnostico": {},
        "status": "start",
    }

    config = {"configurable": {"thread_id": f"dq-audit-{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
    resultado = graph.invoke(estado_inicial, config)

    duracao = (datetime.now() - inicio).total_seconds()
    d = resultado["diagnostico"]

    print()
    print(f"=== Diagnóstico ===")
    print(f"Score qualidade: {d['score_qualidade']}/{d['score_maximo']}")
    print(f"Severidade: {d['severidade']}")
    print(f"Dimensões OK: {d['dimensoes_ok']} | Warning: {d['dimensoes_warning']} | Críticas: {d['dimensoes_criticas']}")
    print(f"Alertas: {len(d['alertas'])} ({sum(1 for a in d['alertas'] if a['nivel'] == 'CRITICO')} críticos, {sum(1 for a in d['alertas'] if a['nivel'] == 'WARNING')} warnings)")
    print(f"Tempo total: {duracao:.1f}s")

    if d["alertas"]:
        print()
        for a in d["alertas"]:
            emoji = "❌" if a["nivel"] == "CRITICO" else "⚠"
            print(f"  {emoji} [{a['nivel']:8s}] {a['dimensao']:15s} — {a['mensagem']}")

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Quality Agent — LangGraph Squad")
    parser.add_argument("--input", default="data/clientes.parquet")
    parser.add_argument("--threshold", type=float, default=0.0, help="Tolerância para alertas (0-1)")
    args = parser.parse_args()

    executar_pipeline(args.input, args.threshold)
