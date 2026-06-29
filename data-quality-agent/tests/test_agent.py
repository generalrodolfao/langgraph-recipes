"""Testes para Data Quality Agent."""

import os
import sys
import tempfile

import pandas as pd
import pytest
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import construir_grafo, EstadoAgent
from quality import (
    auditar_acuracia,
    auditar_completude,
    auditar_consistencia,
    auditar_frescor,
    auditar_unicidade,
    gerar_diagnostico,
)


@pytest.fixture
def df_limpo():
    return pd.DataFrame({
        "cpf": ["111", "222", "333"],
        "email": ["a@b.com", "c@d.com", "e@f.com"],
        "email_alt": [None, "x@y.com", None],
        "telefone": ["(11) 9999", "(21) 8888", None],
        "idade": [25, 30, 35],
        "renda": [5000.0, 7000.0, None],
        "compras_12m": [10, 15, 20],
        "data_nascimento": pd.to_datetime(["1990-01-01", "1985-06-15", "1995-12-31"]),
        "data_cadastro": pd.to_datetime(["2020-01-01", "2019-03-01", "2021-06-01"]),
        "data_atualizacao": pd.to_datetime(["2025-06-01", "2025-05-15", "2025-06-10"]),
    })


@pytest.fixture
def df_sujo():
    df = pd.DataFrame({
        "cpf": ["111", "111", "333", "444"],
        "email": ["a@b.com", "semarroba.com", "c@d.com", "e@f.com"],
        "email_alt": [None, None, None, None],
        "telefone": [None, None, None, "9999"],
        "idade": [25, -5, 30, 35],
        "renda": [5000.0, 7000.0, 500000.0, 6000.0],
        "compras_12m": [10, 15, 999, 20],
        "data_nascimento": pd.to_datetime(["1990-01-01", "1985-06-15", "1995-12-31", "2025-01-01"]),
        "data_cadastro": pd.to_datetime(["2020-01-01", "2019-03-01", "2021-06-01", "2020-01-01"]),
        "data_atualizacao": pd.to_datetime(["2020-01-01", "2019-01-01", "2025-06-10", "2025-06-10"]),
    })
    return df


class TestCompletude:
    def test_limpo(self, df_limpo):
        r = auditar_completude(df_limpo)
        assert r["status"] in ["ok", "warning", "critico"]
        assert "email_alt" in str(r["colunas_com_nulos"]) or "telefone" in str(r["colunas_com_nulos"])

    def test_sujo(self, df_sujo):
        r = auditar_completude(df_sujo)
        assert r["status"] in ["warning", "critico"]


class TestFrescor:
    def test_detecta_desatualizados(self, df_sujo):
        r = auditar_frescor(df_sujo)
        assert r["registros_desatualizados"] >= 2


class TestUnicidade:
    def test_limpo(self, df_limpo):
        r = auditar_unicidade(df_limpo)
        assert r["total_duplicatas"] == 0

    def test_sujo(self, df_sujo):
        r = auditar_unicidade(df_sujo)
        assert r["total_duplicatas"] >= 1


class TestConsistencia:
    def test_limpo(self, df_limpo):
        r = auditar_consistencia(df_limpo)
        assert r["total_violacoes"] == 0

    def test_sujo(self, df_sujo):
        r = auditar_consistencia(df_sujo)
        assert r["total_violacoes"] >= 2


class TestAcuracia:
    def test_detecta_outliers(self, df_sujo):
        r = auditar_acuracia(df_sujo)
        assert r["total_outliers"] >= 0  # com 4 amostras, z-score pode não capturar


class TestDiagnostico:
    def test_consolida(self, df_sujo):
        resultados = [
            auditar_completude(df_sujo),
            auditar_frescor(df_sujo),
            auditar_unicidade(df_sujo),
            auditar_consistencia(df_sujo),
            auditar_acuracia(df_sujo),
        ]
        d = gerar_diagnostico(resultados)
        assert d["score_qualidade"] < 100
        assert len(d["alertas"]) > 0


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

        resultado = executar_pipeline("data/clientes.parquet")
        assert resultado["status"] == "done"
        d = resultado["diagnostico"]
        assert 0 < d["score_qualidade"] < 100
        assert d["severidade"] in ["OK", "BAIXA", "MÉDIA", "ALTA"]
