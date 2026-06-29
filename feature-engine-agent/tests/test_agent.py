"""Testes para Feature Engine Agent."""

import os
import sys
import tempfile

import pandas as pd
import pytest
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import construir_grafo, EstadoAgent
from features import (
    calcular_ticket_medio,
    carregar_dados,
    limpar_dados,
    validar_schema,
)


@pytest.fixture
def dados_teste():
    return pd.DataFrame({
        "data_venda": pd.to_datetime(["2025-01-15", "2025-01-16", "2025-01-17"] * 3),
        "loja_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "produto_id": [10, 20, 30] * 3,
        "categoria": ["Eletrônicos", "Vestuário", "Alimentos"] * 3,
        "quantidade": [2, 5, 10, 1, 3, 7, 4, 2, 8],
        "valor_unitario": [100.0, 50.0, 25.0, 200.0, 80.0, 40.0, 150.0, 60.0, 30.0],
        "valor_total": [200.0, 250.0, 250.0, 200.0, 240.0, 280.0, 600.0, 120.0, 240.0],
        "custo": [150.0, 100.0, 200.0, 80.0, 160.0, 180.0, 450.0, 80.0, 180.0],
        "cliente_id": [1, 2, 3, 1, 2, 3, 4, 5, 6],
        "canal_venda": ["online", "fisico", "app"] * 3,
        "regiao": ["Sudeste", "Sul", "Nordeste"] * 3,
    })


@pytest.fixture
def dados_invalidos():
    return pd.DataFrame({
        "data_venda": pd.to_datetime(["2025-01-15", None, "2025-01-17"]),
        "loja_id": [1, 2, 3],
        "produto_id": [10, -1, None],
        "categoria": ["Eletrônicos", None, "Alimentos"],
        "quantidade": [2, -5, 0],
        "valor_unitario": [100.0, -50.0, 0.0],
        "valor_total": [200.0, 250.0, 0.0],
        "custo": [150.0, 100.0, 200.0],
        "cliente_id": [1, 2, 3],
        "canal_venda": ["online", "fisico", "app"],
        "regiao": ["Sudeste", "Sul", "Nordeste"],
    })


class TestValidacao:
    def test_schema_valido(self, dados_teste):
        resultado = validar_schema(dados_teste)
        assert resultado["valido"]

    def test_schema_invalido(self, dados_invalidos):
        resultado = validar_schema(dados_invalidos)
        assert not resultado["valido"]


class TestLimpeza:
    def test_remove_negativos(self, dados_invalidos):
        df_limpo = limpar_dados(dados_invalidos)
        assert len(df_limpo) < len(dados_invalidos)
        assert all(df_limpo["quantidade"] > 0)
        assert all(df_limpo["valor_unitario"] > 0)

    def test_linhas_validas_mantidas(self, dados_teste):
        df_limpo = limpar_dados(dados_teste)
        assert len(df_limpo) == len(dados_teste)


class TestFeatures:
    def test_ticket_medio(self, dados_teste):
        import duckdb
        con = duckdb.connect(":memory:")
        con.register("vendas", dados_teste)
        resultado = calcular_ticket_medio(con)
        assert len(resultado) > 0
        assert "ticket_medio" in resultado.columns
        assert "loja_id" in resultado.columns

    def test_ticket_medio_valores(self, dados_teste):
        import duckdb
        con = duckdb.connect(":memory:")
        con.register("vendas", dados_teste)
        resultado = calcular_ticket_medio(con)
        loja1 = resultado[resultado["loja_id"] == 1]
        assert len(loja1) == 3


class TestGrafo:
    def test_construcao_grafo(self):
        builder = construir_grafo()
        assert builder is not None

    def test_grafo_compilavel(self):
        builder = construir_grafo()
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)
        assert graph is not None


class TestPipelineEndToEnd:
    def test_pipeline_completo(self):
        from agent import executar_pipeline

        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            output_path = tmp.name

        try:
            resultado = executar_pipeline("data/vendas.parquet", output_path)
            assert resultado["status"] == "done"
            assert len(resultado["features"]) == 8

            import duckdb
            con = duckdb.connect(output_path)
            tabelas = con.execute("SELECT name FROM sqlite_master WHERE type='table'").df()
            assert len(tabelas) == 8
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
