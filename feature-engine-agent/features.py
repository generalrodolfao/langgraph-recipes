"""
Lógica de feature engineering — funções puras chamadas pelos nós do agente.
"""

from datetime import date, timedelta
from typing import Optional

import duckdb
import pandas as pd


def carregar_dados(caminho_parquet: str) -> pd.DataFrame:
    """Carrega dados de vendas do parquet."""
    df = pd.read_parquet(caminho_parquet)
    df["data_venda"] = pd.to_datetime(df["data_venda"])
    return df


def validar_schema(df: pd.DataFrame) -> dict:
    """Valida schema e retorna diagnóstico."""
    colunas_esperadas = {
        "data_venda",
        "loja_id",
        "produto_id",
        "categoria",
        "quantidade",
        "valor_unitario",
        "valor_total",
        "custo",
        "cliente_id",
        "canal_venda",
        "regiao",
    }

    problemas = []
    nulos = df.isnull().sum()
    nulos_cols = nulos[nulos > 0]

    for col in colunas_esperadas:
        if col not in df.columns:
            problemas.append(f"coluna ausente: {col}")

    if len(nulos_cols) > 0:
        for col, qtde in nulos_cols.items():
            problemas.append(f"nulos em {col}: {qtde}")

    return {
        "valido": len(problemas) == 0,
        "problemas": problemas,
        "linhas": len(df),
        "colunas": len(df.columns),
        "nulos_por_coluna": {str(k): int(v) for k, v in nulos_cols.items()},
    }


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers básicos e corrige tipos."""
    df = df.copy()
    df = df[df["quantidade"] > 0]
    df = df[df["valor_unitario"] > 0]
    df = df[df["valor_total"] > 0]
    df["data_venda"] = pd.to_datetime(df["data_venda"]).dt.date
    return df


def calcular_ticket_medio(con: duckdb.DuckDBPyConnection, data_ref: Optional[date] = None) -> pd.DataFrame:
    """Ticket médio por loja x dia."""
    filtro = f"WHERE data_venda <= '{data_ref}'" if data_ref else ""
    return con.sql(f"""
        SELECT
            data_venda,
            loja_id,
            SUM(valor_total) AS receita_total,
            COUNT(*) AS n_vendas,
            ROUND(SUM(valor_total) / COUNT(*), 2) AS ticket_medio,
            COUNT(DISTINCT cliente_id) AS clientes_unicos
        FROM vendas
        {filtro}
        GROUP BY data_venda, loja_id
        ORDER BY data_venda, loja_id
    """).df()


def calcular_frequencia_cliente(con: duckdb.DuckDBPyConnection, data_ref: Optional[date] = None) -> pd.DataFrame:
    """Frequência de compra por cliente x mês."""
    filtro = f"WHERE data_venda <= '{data_ref}'" if data_ref else ""
    return con.sql(f"""
        SELECT
            cliente_id,
            STRFTIME(data_venda, '%Y-%m') AS mes,
            COUNT(*) AS compras,
            ROUND(SUM(valor_total), 2) AS gasto_total,
            COUNT(DISTINCT loja_id) AS lojas_visitadas
        FROM vendas
        {filtro}
        GROUP BY cliente_id, mes
        ORDER BY cliente_id, mes
    """).df()


def calcular_share_categoria(con: duckdb.DuckDBPyConnection, data_ref: Optional[date] = None) -> pd.DataFrame:
    """Share de vendas por categoria dentro de cada loja."""
    where_clause = f"AND v.data_venda <= '{data_ref}'" if data_ref else ""
    return con.sql(f"""
        WITH total_loja AS (
            SELECT loja_id, SUM(valor_total) AS total
            FROM vendas
            WHERE 1=1 {'AND data_venda <= \'' + str(data_ref) + '\'' if data_ref else ''}
            GROUP BY loja_id
        )
        SELECT
            v.loja_id,
            v.categoria,
            ROUND(SUM(v.valor_total) / t.total * 100, 2) AS share_pct,
            ROUND(SUM(v.valor_total), 2) AS receita_categoria,
            ROUND(SUM(v.valor_total) - SUM(v.custo), 2) AS margem_categoria
        FROM vendas v
        JOIN total_loja t ON v.loja_id = t.loja_id {where_clause}
        GROUP BY v.loja_id, v.categoria, t.total
        ORDER BY v.loja_id, share_pct DESC
    """).df()


def calcular_crescimento_mom(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Crescimento mês-sobre-mês por loja."""
    return con.sql("""
        WITH mensal AS (
            SELECT
                loja_id,
                STRFTIME(data_venda, '%Y-%m') AS mes,
                SUM(valor_total) AS receita
            FROM vendas
            GROUP BY loja_id, mes
        )
        SELECT
            loja_id,
            mes,
            receita,
            ROUND((receita - LAG(receita) OVER (PARTITION BY loja_id ORDER BY mes)) / NULLIF(LAG(receita) OVER (PARTITION BY loja_id ORDER BY mes), 0) * 100, 2) AS crescimento_mom_pct
        FROM mensal
        ORDER BY loja_id, mes
    """).df()


def calcular_recencia_cliente(con: duckdb.DuckDBPyConnection, data_ref: date) -> pd.DataFrame:
    """Dias desde a última compra de cada cliente."""
    return con.sql(f"""
        SELECT
            cliente_id,
            MAX(data_venda) AS ultima_compra,
            DATEDIFF('day', MAX(data_venda), DATE '{data_ref}') AS dias_desde_ultima
        FROM vendas
        GROUP BY cliente_id
        ORDER BY dias_desde_ultima
    """).df()


def calcular_tendencia_7d(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Média móvel 7 dias de vendas por loja."""
    return con.sql("""
        WITH diario AS (
            SELECT data_venda, loja_id, SUM(valor_total) AS receita
            FROM vendas
            GROUP BY data_venda, loja_id
        )
        SELECT
            data_venda,
            loja_id,
            receita,
            ROUND(AVG(receita) OVER (
                PARTITION BY loja_id
                ORDER BY data_venda
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 2) AS media_movel_7d,
            ROUND(100.0 * (receita - AVG(receita) OVER (
                PARTITION BY loja_id
                ORDER BY data_venda
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )) / NULLIF(AVG(receita) OVER (
                PARTITION BY loja_id
                ORDER BY data_venda
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 0), 2) AS desvio_pct
        FROM diario
        ORDER BY data_venda, loja_id
    """).df()


def calcular_sazonalidade_dia(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Índice de sazonalidade por dia da semana e loja."""
    return con.sql("""
        WITH diario AS (
            SELECT
                loja_id,
                DAYOFWEEK(CAST(data_venda AS DATE)) AS dia_semana,
                SUM(valor_total) AS receita
            FROM vendas
            GROUP BY loja_id, dia_semana
        ),
        media_loja AS (
            SELECT loja_id, AVG(receita) AS media
            FROM diario
            GROUP BY loja_id
        )
        SELECT
            d.loja_id,
            CASE d.dia_semana
                WHEN 0 THEN 'Domingo'
                WHEN 1 THEN 'Segunda'
                WHEN 2 THEN 'Terça'
                WHEN 3 THEN 'Quarta'
                WHEN 4 THEN 'Quinta'
                WHEN 5 THEN 'Sexta'
                WHEN 6 THEN 'Sábado'
            END AS dia_da_semana,
            d.dia_semana,
            ROUND(d.receita, 2) AS receita_media,
            ROUND(d.receita / m.media, 2) AS indice_sazonal
        FROM diario d
        JOIN media_loja m ON d.loja_id = m.loja_id
        ORDER BY d.loja_id, d.dia_semana
    """).df()


def calcular_top_produtos(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Top 5 produtos por receita em cada loja x mês."""
    return con.sql("""
        WITH ranking AS (
            SELECT
                loja_id,
                STRFTIME(data_venda, '%Y-%m') AS mes,
                produto_id,
                SUM(valor_total) AS receita,
                ROW_NUMBER() OVER (PARTITION BY loja_id, STRFTIME(data_venda, '%Y-%m') ORDER BY SUM(valor_total) DESC) AS rank
            FROM vendas
            GROUP BY loja_id, mes, produto_id
        )
        SELECT loja_id, mes, produto_id, receita, rank
        FROM ranking
        WHERE rank <= 5
        ORDER BY loja_id, mes, rank
    """).df()


def salvar_features(con: duckdb.DuckDBPyConnection, features: dict, caminho_saida: str):
    """Salva todas as features em um arquivo DuckDB."""
    con_out = duckdb.connect(caminho_saida)
    for nome, df in features.items():
        con_out.register(nome, df)
        con_out.execute(f"CREATE TABLE IF NOT EXISTS {nome} AS SELECT * FROM {nome}")
    con_out.close()
