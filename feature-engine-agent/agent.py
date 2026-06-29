"""
Feature Engine Agent — Pipeline de feature engineering com LangGraph.

Fluxo:
  ingest → validate → transform → feature_calc → save

Cada nó é um step atômico com checkpoint para retomada de falha.

Uso:
    python agent.py
    python agent.py --input data/vendas.parquet --output features.duckdb
"""

import argparse
import os
import sys
from datetime import date, datetime
from typing import TypedDict

import duckdb
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from features import (
    calcular_crescimento_mom,
    calcular_frequencia_cliente,
    calcular_recencia_cliente,
    calcular_sazonalidade_dia,
    calcular_share_categoria,
    calcular_tendencia_7d,
    calcular_ticket_medio,
    calcular_top_produtos,
    carregar_dados,
    limpar_dados,
    salvar_features,
    validar_schema,
)


class EstadoAgent(TypedDict):
    """Estado compartilhado entre nós do grafo."""

    input_path: str
    output_path: str
    df: pd.DataFrame | None
    con: duckdb.DuckDBPyConnection | None
    validacao: dict
    features: dict
    erros: list[str]
    status: str


def no_ingest(state: EstadoAgent) -> EstadoAgent:
    """Carrega dados do parquet para DataFrame e registra no DuckDB."""
    print("[ingest] Carregando dados...")
    df = carregar_dados(state["input_path"])

    con = duckdb.connect(":memory:")
    con.register("vendas", df)
    con.execute("CREATE TEMP TABLE vendas AS SELECT * FROM vendas")

    print(f"[ingest] {len(df):,} linhas carregadas")

    return {
        **state,
        "df": df,
        "con": con,
        "status": "ingest_ok",
    }


def no_validate(state: EstadoAgent) -> EstadoAgent:
    """Valida schema e qualidade dos dados."""
    print("[validate] Validando schema...")
    resultado = validar_schema(state["df"])

    if resultado["valido"]:
        print("[validate] Schema OK")
        return {**state, "validacao": resultado, "status": "validate_ok"}
    else:
        erros = resultado["problemas"]
        print(f"[validate] {len(erros)} problemas encontrados:")
        for e in erros:
            print(f"  - {e}")
        return {**state, "validacao": resultado, "erros": erros, "status": "validate_fail"}


def no_transform(state: EstadoAgent) -> EstadoAgent:
    """Limpa e transforma dados."""
    print("[transform] Limpando dados...")
    df_limpo = limpar_dados(state["df"])
    removidas = len(state["df"]) - len(df_limpo)

    con = duckdb.connect(":memory:")
    con.register("vendas", df_limpo)
    con.execute("CREATE TEMP TABLE vendas AS SELECT FROM vendas")

    print(f"[transform] {removidas} linhas removidas, {len(df_limpo):,} restantes")

    return {**state, "df": df_limpo, "con": con, "status": "transform_ok"}


def no_feature_calc(state: EstadoAgent) -> EstadoAgent:
    """Calcula todas as features."""
    print("[features] Calculando features...")
    con = state["con"]
    data_ref = date.today()

    features = {}

    print("  - ticket_medio...")
    features["ticket_medio"] = calcular_ticket_medio(con)

    print("  - frequencia_cliente...")
    features["frequencia_cliente"] = calcular_frequencia_cliente(con)

    print("  - share_categoria...")
    features["share_categoria"] = calcular_share_categoria(con)

    print("  - crescimento_mom...")
    features["crescimento_mom"] = calcular_crescimento_mom(con)

    print("  - recencia_cliente...")
    features["recencia_cliente"] = calcular_recencia_cliente(con, data_ref)

    print("  - tendencia_7d...")
    features["tendencia_7d"] = calcular_tendencia_7d(con)

    print("  - sazonalidade_dia...")
    features["sazonalidade_dia"] = calcular_sazonalidade_dia(con)

    print("  - top_produtos...")
    features["top_produtos"] = calcular_top_produtos(con)

    total_linhas = sum(len(df) for df in features.values())
    print(f"[features] {len(features)} tabelas geradas, {total_linhas:,} linhas totais")

    return {**state, "features": features, "status": "features_ok"}


def no_save(state: EstadoAgent) -> EstadoAgent:
    """Salva features em DuckDB."""
    print(f"[save] Salvando features em {state['output_path']}...")
    salvar_features(state["con"], state["features"], state["output_path"])

    tamanho = os.path.getsize(state["output_path"])
    print(f"[save] Arquivo salvo: {tamanho / 1024 / 1024:.1f} MB")

    return {**state, "status": "done"}


def rota_apos_validate(state: EstadoAgent) -> str:
    """Roteamento condicional: se validação falhou, termina com erro."""
    if state["status"] == "validate_fail":
        return END
    return "transform"


def rota_apos_erro(state: EstadoAgent) -> str:
    if state["erros"]:
        return END
    return "transform"


def construir_grafo() -> StateGraph:
    """Constrói o grafo de agentes com checkpoint."""
    builder = StateGraph(EstadoAgent)

    builder.add_node("ingest", no_ingest)
    builder.add_node("validate", no_validate)
    builder.add_node("transform", no_transform)
    builder.add_node("feature_calc", no_feature_calc)
    builder.add_node("save", no_save)

    builder.set_entry_point("ingest")
    builder.add_edge("ingest", "validate")
    builder.add_conditional_edges("validate", rota_apos_validate)
    builder.add_edge("transform", "feature_calc")
    builder.add_edge("feature_calc", "save")
    builder.add_edge("save", END)

    return builder


def executar_pipeline(input_path: str, output_path: str):
    """Executa o pipeline completo de feature engineering."""
    if not os.path.exists(input_path):
        print(f"Erro: arquivo de entrada '{input_path}' não encontrado.")
        print("Execute primeiro: python data/generate.py")
        sys.exit(1)

    print(f"=== Feature Engine Agent ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Início: {datetime.now().strftime('%H:%M:%S')}")
    print()

    inicio = datetime.now()

    builder = construir_grafo()
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    estado_inicial: EstadoAgent = {
        "input_path": input_path,
        "output_path": output_path,
        "df": None,
        "con": None,
        "validacao": {},
        "features": {},
        "erros": [],
        "status": "start",
    }

    config = {"configurable": {"thread_id": f"feature-engine-{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
    resultado = graph.invoke(estado_inicial, config)

    duracao = (datetime.now() - inicio).total_seconds()

    print()
    print(f"=== Pipeline concluído ===")
    print(f"Status: {resultado['status']}")
    print(f"Features geradas: {len(resultado['features'])} tabelas")
    print(f"Tempo total: {duracao:.1f}s")

    if resultado["erros"]:
        print(f"Erros: {len(resultado['erros'])}")
        for e in resultado["erros"]:
            print(f"  - {e}")

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engine Agent — LangGraph pipeline")
    parser.add_argument("--input", default="data/vendas.parquet", help="Parquet de entrada")
    parser.add_argument("--output", default="features.duckdb", help="DuckDB de saída")
    args = parser.parse_args()

    executar_pipeline(args.input, args.output)
