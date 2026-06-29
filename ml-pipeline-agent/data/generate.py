"""
Gera dados sintéticos para ML Pipeline Agent.

200 lojas × 24 meses de features + target (faturamento do mês seguinte).
Features incluem: sazonalidade, tendência, ticket médio, categorias, região.
"""

import os

import numpy as np
import pandas as pd

np.random.seed(42)

N_LOJAS = 200
N_MESES = 24
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
CATEGORIAS = ["Eletrônicos", "Vestuário", "Alimentos", "Móveis", "Esportes", "Livros"]

os.makedirs("data", exist_ok=True)


def gerar_dados(n_lojas: int, n_meses: int) -> pd.DataFrame:
    linhas = []

    for loja_id in range(1, n_lojas + 1):
        regiao = REGIOES[loja_id % len(REGIOES)]
        porte = np.random.choice(["Pequeno", "Médio", "Grande"], p=[0.3, 0.5, 0.2])
        area_m2 = {"Pequeno": np.random.randint(50, 150), "Médio": np.random.randint(150, 400), "Grande": np.random.randint(400, 1000)}[porte]
        funcionarios = {"Pequeno": np.random.randint(3, 8), "Médio": np.random.randint(8, 20), "Grande": np.random.randint(20, 50)}[porte]
        categoria_principal = np.random.choice(CATEGORIAS)

        base_receita = {
            "Norte": 80_000, "Nordeste": 90_000, "Centro-Oeste": 100_000,
            "Sudeste": 150_000, "Sul": 120_000,
        }[regiao] * (area_m2 / 200)

        base_ticket = base_receita / 300
        base_clientes = int(base_receita / base_ticket)

        crescimento_anual = np.random.normal(0.05, 0.03)

        for mes_idx in range(n_meses):
            t = mes_idx + 1
            sazonal = 1.0 + 0.15 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.02)
            tendencia = 1.0 + crescimento_anual * (t / 12)
            ruido = np.random.normal(0, 0.05)

            receita = base_receita * sazonal * tendencia * (1 + ruido)
            ticket_medio = base_ticket * sazonal * tendencia * (1 + ruido * 0.3)
            n_clientes = int(base_clientes * sazonal * (1 + ruido * 0.5))

            share_eletro = 0.3 + np.random.normal(0, 0.03) if categoria_principal == "Eletrônicos" else 0.15 + np.random.normal(0, 0.02)
            share_vest = 0.2 + np.random.normal(0, 0.02)
            share_alim = 0.25 + np.random.normal(0, 0.02)
            share_moveis = 0.1 + np.random.normal(0, 0.02)
            share_esportes = 0.08 + np.random.normal(0, 0.01)
            share_livros = 0.07 + np.random.normal(0, 0.01)

            total_share = share_eletro + share_vest + share_alim + share_moveis + share_esportes + share_livros
            share_eletro /= total_share
            share_vest /= total_share
            share_alim /= total_share
            share_moveis /= total_share
            share_esportes /= total_share
            share_livros /= total_share

            custo_pct = 0.55 + np.random.normal(0, 0.03)
            margem = receita * (1 - custo_pct)

            # target: faturamento do mes seguinte (não existe pro último mês)
            target = None

            linhas.append({
                "loja_id": loja_id,
                "regiao": regiao,
                "porte": porte,
                "categoria_principal": categoria_principal,
                "area_m2": area_m2,
                "funcionarios": funcionarios,
                "mes": mes_idx + 1,
                "ano": 2025 if mes_idx < 12 else 2026,
                "receita": round(receita, 2),
                "ticket_medio": round(ticket_medio, 2),
                "n_clientes": n_clientes,
                "custo_pct": round(custo_pct, 4),
                "margem": round(margem, 2),
                "share_eletronicos": round(share_eletro, 4),
                "share_vestuario": round(share_vest, 4),
                "share_alimentos": round(share_alim, 4),
                "share_moveis": round(share_moveis, 4),
                "share_esportes": round(share_esportes, 4),
                "share_livros": round(share_livros, 4),
                "crescimento_3m": 0.0,
                "crescimento_6m": 0.0,
                "ticket_3m_avg": 0.0,
                "clientes_3m_avg": 0,
                "sazonalidade_mes": round(sazonal, 4),
                "dia_semana_pico": np.random.randint(0, 7),
            })

    df = pd.DataFrame(linhas)

    # Calcula features derivadas (médias móveis e crescimento)
    for loja_id in df["loja_id"].unique():
        mask = df["loja_id"] == loja_id
        loja_df = df[mask].sort_values("mes")

        receitas = loja_df["receita"].values
        tickets = loja_df["ticket_medio"].values
        clientes = loja_df["n_clientes"].values

        for i in range(len(loja_df)):
            if i >= 2:
                df.loc[loja_df.index[i], "crescimento_3m"] = round((receitas[i] / receitas[i - 3] - 1) * 100, 2) if i >= 3 else 0
            if i >= 5:
                df.loc[loja_df.index[i], "crescimento_6m"] = round((receitas[i] / receitas[i - 6] - 1) * 100, 2)
            if i >= 2:
                df.loc[loja_df.index[i], "ticket_3m_avg"] = round(np.mean(tickets[max(0, i - 2):i + 1]), 2)
                df.loc[loja_df.index[i], "clientes_3m_avg"] = int(np.mean(clientes[max(0, i - 2):i + 1]))

    # Target: receita do mês seguinte (shift -1)
    targets = []
    for loja_id in df["loja_id"].unique():
        mask = df["loja_id"] == loja_id
        loja_df = df[mask].sort_values("mes")
        receitas_futuras = loja_df["receita"].shift(-1).values
        targets.extend(receitas_futuras)

    df["target"] = targets
    df = df.dropna(subset=["target"])

    return df


if __name__ == "__main__":
    print(f"Gerando dados: {N_LOJAS} lojas × {N_MESES} meses...")
    df = gerar_dados(N_LOJAS, N_MESES)

    print(f"Salvando data/vendas_ml.parquet ({len(df):,} linhas)")
    df.to_parquet("data/vendas_ml.parquet", index=False)

    print(f"Resumo:")
    print(f"  Lojas: {df['loja_id'].nunique()}")
    print(f"  Features: {len(df.columns) - 2} (excluindo loja_id e target)")
    print(f"  Target média: R$ {df['target'].mean():,.2f}")
    print(f"  Target range: R$ {df['target'].min():,.2f} — R$ {df['target'].max():,.2f}")
    print(f"  Features nulas: {df.drop(columns=['target']).isnull().sum().sum()}")
