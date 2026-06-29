"""
Gerador de dados sintéticos de vendas para Feature Engine Agent.

Gera um arquivo Parquet com ~200K linhas simulando transações
de uma rede de varejo com 10 lojas, 100 produtos, 5K clientes,
ao longo de 12 meses (2025).
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)

N_LOJAS = 10
N_PRODUTOS = 100
N_CLIENTES = 5000
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
CATEGORIAS = ["Eletrônicos", "Vestuário", "Alimentos", "Móveis", "Esportes", "Livros", "Beleza", "Brinquedos", "Ferramentas", "Automotivo"]
CANAIS = ["online", "fisico", "app"]
DATA_INICIO = date(2025, 1, 1)
DATA_FIM = date(2025, 12, 31)

os.makedirs("data", exist_ok=True)


def gerar_base_clientes(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "cliente_id": range(1, n + 1),
        "cidade_cliente": np.random.choice(
            ["São Paulo", "Rio de Janeiro", "Brasília", "Salvador", "Fortaleza",
             "Belo Horizonte", "Manaus", "Curitiba", "Recife", "Porto Alegre"], n, p=[0.25, 0.15, 0.10, 0.08, 0.07, 0.12, 0.05, 0.06, 0.05, 0.07]
        ),
    })


def gerar_base_produtos(n: int) -> pd.DataFrame:
    precos = {
        "Eletrônicos": (50, 999),
        "Vestuário": (20, 300),
        "Alimentos": (5, 80),
        "Móveis": (100, 2000),
        "Esportes": (30, 500),
        "Livros": (15, 120),
        "Beleza": (10, 200),
        "Brinquedos": (20, 300),
        "Ferramentas": (25, 400),
        "Automotivo": (30, 600),
    }

    categorias_produto = list(CATEGORIAS) * (n // len(CATEGORIAS)) + CATEGORIAS[: n % len(CATEGORIAS)]
    np.random.shuffle(categorias_produto)

    produtos = []
    for i in range(n):
        cat = categorias_produto[i]
        lo, hi = precos[cat]
        produtos.append({
            "produto_id": i + 1,
            "categoria": cat,
            "preco_base": round(np.random.uniform(lo, hi), 2),
            "custo": round(np.random.uniform(lo * 0.3, hi * 0.6), 2),
        })

    return pd.DataFrame(produtos)


def gerar_transacoes(
    n_lojas: int, n_produtos: int, n_clientes: int,
    inicio: date, fim: date, produtos_df: pd.DataFrame,
    clientes_df: pd.DataFrame,
) -> pd.DataFrame:
    dias = (fim - inicio).days + 1
    datas = [inicio + timedelta(days=i) for i in range(dias)]

    lojas_regiao = {i: np.random.choice(REGIOES) for i in range(1, n_lojas + 1)}
    lojas_canal = {i: np.random.choice(CANAIS, p=[0.40, 0.35, 0.25]) for i in range(1, n_lojas + 1)}

    n_transacoes = 200_000
    transacoes = []

    for _ in range(n_transacoes):
        loja_id = np.random.randint(1, n_lojas + 1)
        cliente_id = np.random.randint(1, n_clientes + 1)
        produto_id = np.random.randint(1, n_produtos + 1)
        data_venda = np.random.choice(datas)

        produto = produtos_df[produtos_df["produto_id"] == produto_id].iloc[0]
        preco = produto["preco_base"] + np.random.normal(0, preco["preco_base"] * 0.05)
        preco = max(preco, produto["custo"] * 1.1)

        quantidade = np.random.randint(1, 20)
        valor_total = round(preco * quantidade, 2)

        transacoes.append({
            "data_venda": data_venda,
            "loja_id": loja_id,
            "produto_id": produto_id,
            "categoria": produto["categoria"],
            "quantidade": quantidade,
            "valor_unitario": round(preco, 2),
            "valor_total": valor_total,
            "custo": round(produto["custo"] * quantidade, 2),
            "cliente_id": cliente_id,
            "canal_venda": lojas_canal[loja_id],
            "regiao": lojas_regiao[loja_id],
        })

    df = pd.DataFrame(transacoes)
    df["data_venda"] = pd.to_datetime(df["data_venda"]).dt.date
    df = df.sort_values("data_venda").reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Gerando base de clientes...")
    clientes = gerar_base_clientes(N_CLIENTES)

    print("Gerando base de produtos...")
    produtos = gerar_base_produtos(N_PRODUTOS)

    print(f"Gerando {200_000:,} transações...")
    vendas = gerar_transacoes(N_LOJAS, N_PRODUTOS, N_CLIENTES, DATA_INICIO, DATA_FIM, produtos, clientes)

    print(f"Salvando data/vendas.parquet ({len(vendas):,} linhas, {vendas.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB)")
    vendas.to_parquet("data/vendas.parquet", index=False)
    clientes.to_parquet("data/clientes.parquet", index=False)
    produtos.to_parquet("data/produtos.parquet", index=False)

    print("Resumo:")
    print(f"  Período: {vendas['data_venda'].min()} → {vendas['data_venda'].max()}")
    print(f"  Lojas: {vendas['loja_id'].nunique()}")
    print(f"  Produtos: {vendas['produto_id'].nunique()}")
    print(f"  Clientes: {vendas['cliente_id'].nunique()}")
    print(f"  Receita total: R$ {vendas['valor_total'].sum():,.2f}")
    print(f"  Ticket médio: R$ {vendas['valor_total'].mean():,.2f}")
