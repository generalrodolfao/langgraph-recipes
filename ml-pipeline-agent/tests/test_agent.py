"""Testes para ML Pipeline Agent."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import construir_grafo, EstadoAgent
from model import (
    avaliar_modelo,
    carregar_dados,
    preparar_features,
    selecionar_melhor,
    split_dados,
    treinar_modelo,
)


@pytest.fixture
def dados_teste():
    np.random.seed(42)
    data = []
    for loja in range(1, 11):
        for mes in range(1, 13):
            data.append({
                "loja_id": loja,
                "regiao": "Sudeste",
                "porte": "Médio",
                "categoria_principal": "Eletrônicos",
                "area_m2": 200,
                "funcionarios": 10,
                "mes": mes,
                "ano": 2025,
                "receita": 100_000 + loja * 5_000 + mes * 3_000 + np.random.normal(0, 5_000),
                "ticket_medio": 300 + np.random.normal(0, 10),
                "n_clientes": 300 + np.random.randint(-20, 20),
                "custo_pct": 0.6 + np.random.normal(0, 0.02),
                "margem": 40_000 + np.random.normal(0, 5_000),
                "share_eletronicos": 0.3, "share_vestuario": 0.2, "share_alimentos": 0.2,
                "share_moveis": 0.15, "share_esportes": 0.08, "share_livros": 0.07,
                "crescimento_3m": np.random.normal(2, 1),
                "crescimento_6m": np.random.normal(5, 2),
                "ticket_3m_avg": 300 + np.random.normal(0, 5),
                "clientes_3m_avg": 300,
                "sazonalidade_mes": 1.0 + np.random.normal(0, 0.02),
                "dia_semana_pico": np.random.randint(0, 7),
                "target": 110_000 + loja * 4_000 + np.random.normal(0, 5_000),
            })
    return pd.DataFrame(data)


class TestPreparacao:
    def test_features_shape(self, dados_teste):
        X, y, cols = preparar_features(dados_teste)
        assert X.shape[0] == len(dados_teste)
        assert len(y) == len(dados_teste)
        assert len(cols) > 15

    def test_filtro_loja(self, dados_teste):
        X, y, _ = preparar_features(dados_teste, loja_id=5)
        assert len(X) == 12
        assert len(y) == 12


class TestSplit:
    def test_proporcoes(self):
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        dados = split_dados(X, y)
        assert dados["n_train"] == 80
        assert dados["n_test"] == 20


class TestTreino:
    def test_treina_rf(self):
        X = np.random.randn(50, 5)
        y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.normal(0, 0.1, 50)
        modelo = treinar_modelo("RandomForest", X, y)
        pred = modelo.predict(X[:5])
        assert len(pred) == 5
        assert modelo.n_estimators == 100


class TestAvaliacao:
    def test_metricas(self):
        X = np.random.randn(50, 5)
        y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.normal(0, 0.01, 50)
        modelo = treinar_modelo("LinearRegression", X, y)
        metricas = avaliar_modelo(modelo, X, y)
        assert metricas["r2"] > 0.99
        assert metricas["rmse"] > 0
        assert metricas["mae"] > 0


class TestSelecao:
    def test_melhor_modelo(self):
        resultados = {
            "A": {"r2": 0.85, "rmse": 100},
            "B": {"r2": 0.92, "rmse": 80},
            "C": {"r2": 0.78, "rmse": 120},
        }
        nome, metricas = selecionar_melhor(resultados)
        assert nome == "B"
        assert metricas["r2"] == 0.92


class TestGrafo:
    def test_construcao(self):
        builder = construir_grafo()
        assert builder is not None

    def test_compilavel(self):
        builder = construir_grafo()
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
        assert graph is not None


class TestPipelineEndToEnd:
    def test_pipeline_completo(self):
        from agent import executar_pipeline

        output_path = os.path.join(tempfile.gettempdir(), "test_ml_model.pkl")

        try:
            resultado = executar_pipeline("data/vendas_ml.parquet", output_path, loja_id=None)
            assert resultado["status"] == "done"
            assert resultado["melhor_r2"] > 0.5
            assert resultado["n_lojas"] == 200
            assert resultado["n_features"] > 15
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
