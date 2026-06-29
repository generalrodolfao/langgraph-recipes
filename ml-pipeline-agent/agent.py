"""
ML Pipeline Agent — Pipeline de machine learning end-to-end com LangGraph.

Fluxo:
  load → split → train → evaluate → select → deploy

Cada nó é atômico com checkpoint. Se falhar no train, retoma do split.

Uso:
    python agent.py
    python agent.py --target loja_id 42
"""

import argparse
import os
import sys
from datetime import datetime
from typing import TypedDict

import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from model import (
    COLS_FEATURES,
    avaliar_modelo,
    carregar_dados,
    preparar_features,
    salvar_modelo,
    selecionar_melhor,
    split_dados,
    treinar_modelo,
)


class EstadoAgent(TypedDict):
    """Estado serializável compartilhado entre nós do grafo."""

    input_path: str
    output_path: str
    loja_id: int | None
    n_lojas: int
    n_features: int
    n_train: int
    n_test: int
    resultados: dict
    melhor_modelo: str
    melhor_r2: float
    feature_names: list[str]
    status: str


# Estado não-serializável mantido em memória
_runtime: dict = {}


def no_load(state: EstadoAgent) -> EstadoAgent:
    print("[load] Carregando dados...")
    df = carregar_dados(state["input_path"])
    X, y, feature_names = preparar_features(df, loja_id=state["loja_id"])

    _runtime["X"] = X
    _runtime["y"] = y

    print(f"[load] {df['loja_id'].nunique()} lojas, {len(df['mes'].unique())} meses, {len(feature_names)} features")

    return {
        **state,
        "n_lojas": df["loja_id"].nunique(),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "status": "load_ok",
    }


def no_split(state: EstadoAgent) -> EstadoAgent:
    print("[split] Separando treino/teste 80/20...")
    dados = split_dados(_runtime["X"], _runtime["y"])
    _runtime.update(dados)

    print(f"[split] Treino: {dados['n_train']:,} lojas, Teste: {dados['n_test']:,} lojas")

    return {**state, "n_train": dados["n_train"], "n_test": dados["n_test"], "status": "split_ok"}


def no_train(state: EstadoAgent) -> EstadoAgent:
    print("[train] Treinando 3 modelos...")
    modelos = {}
    X_train, y_train = _runtime["X_train"], _runtime["y_train"]

    for nome in ["RandomForest", "LinearRegression"]:
        print(f"  - {nome}...")
        modelos[nome] = treinar_modelo(nome, X_train, y_train)

    _runtime["modelos"] = modelos
    return {**state, "status": "train_ok"}


def no_evaluate(state: EstadoAgent) -> EstadoAgent:
    print("[evaluate] Avaliando modelos...")
    X_test, y_test = _runtime["X_test"], _runtime["y_test"]
    modelos = _runtime["modelos"]
    resultados = {}

    for nome, modelo in modelos.items():
        metricas = avaliar_modelo(modelo, X_test, y_test)
        resultados[nome] = metricas
        print(f"  {nome:20s} R²={metricas['r2']:.4f}  RMSE={metricas['rmse']:,.0f}  MAE={metricas['mae']:,.0f}")

    return {**state, "resultados": resultados, "status": "eval_ok"}


def no_select(state: EstadoAgent) -> EstadoAgent:
    print("[select] Selecionando melhor modelo...")
    melhor_nome, melhor_metricas = selecionar_melhor(state["resultados"])
    melhor_r2 = melhor_metricas["r2"]

    _runtime["modelo_final"] = _runtime["modelos"][melhor_nome]

    print(f"[select] Melhor: {melhor_nome} (R²={melhor_r2:.4f})")

    return {**state, "melhor_modelo": melhor_nome, "melhor_r2": melhor_r2, "status": "select_ok"}


def no_deploy(state: EstadoAgent) -> EstadoAgent:
    print(f"[deploy] Salvando modelo em {state['output_path']}...")
    salvar_modelo(_runtime["modelo_final"], state["output_path"])

    tamanho = os.path.getsize(state["output_path"])
    print(f"[deploy] Modelo salvo: {tamanho / 1024 / 1024:.1f} MB")

    return {**state, "status": "done"}


def construir_grafo() -> StateGraph:
    builder = StateGraph(EstadoAgent)

    builder.add_node("load", no_load)
    builder.add_node("split", no_split)
    builder.add_node("train", no_train)
    builder.add_node("evaluate", no_evaluate)
    builder.add_node("select", no_select)
    builder.add_node("deploy", no_deploy)

    builder.set_entry_point("load")
    builder.add_edge("load", "split")
    builder.add_edge("split", "train")
    builder.add_edge("train", "evaluate")
    builder.add_edge("evaluate", "select")
    builder.add_edge("select", "deploy")
    builder.add_edge("deploy", END)

    return builder


def executar_pipeline(input_path: str, output_path: str, loja_id: int | None = None):
    if not os.path.exists(input_path):
        print(f"Erro: arquivo '{input_path}' não encontrado. Execute: python data/generate.py")
        sys.exit(1)

    print(f"=== ML Pipeline Agent ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    if loja_id:
        print(f"Loja:   {loja_id}")
    print(f"Início: {datetime.now().strftime('%H:%M:%S')}")
    print()

    inicio = datetime.now()

    builder = construir_grafo()
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    estado_inicial: EstadoAgent = {
        "input_path": input_path,
        "output_path": output_path,
        "loja_id": loja_id,
        "n_lojas": 0,
        "n_features": 0,
        "n_train": 0,
        "n_test": 0,
        "resultados": {},
        "melhor_modelo": "",
        "melhor_r2": 0.0,
        "feature_names": [],
        "status": "start",
    }

    config = {"configurable": {"thread_id": f"ml-pipeline-{datetime.now().strftime('%Y%m%d%H%M%S')}"}}
    resultado = graph.invoke(estado_inicial, config)

    duracao = (datetime.now() - inicio).total_seconds()

    print()
    print(f"=== Pipeline concluído ===")
    print(f"Status: {resultado['status']}")
    print(f"Modelo: {resultado['melhor_modelo']} (R²={resultado['melhor_r2']:.4f})")
    print(f"Nº features: {resultado['n_features']}")
    print(f"Treino/Teste: {resultado['n_train']}/{resultado['n_test']}")
    print(f"Tempo total: {duracao:.1f}s")

    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Pipeline Agent — LangGraph")
    parser.add_argument("--input", default="data/vendas_ml.parquet")
    parser.add_argument("--output", default="model.pkl")
    parser.add_argument("--target", type=int, default=None, help="Filtrar por loja_id específica")
    args = parser.parse_args()

    executar_pipeline(args.input, args.output, args.target)
