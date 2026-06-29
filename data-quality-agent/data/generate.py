"""
Gera dados sintéticos de clientes e-commerce com falhas de qualidade intencionais.

500 clientes com problemas em 5 dimensões:
- Completude: nulos em telefone (~8%), email_alt (~12%), renda (~4%)
- Frescor: 12% dos registros não atualizados há >2 anos
- Unicidade: 2 CPFs duplicados
- Consistência: idade negativa (3), email inválido (2), data_nasc > data_cadastro (4)
- Acurácia: 5 outliers (renda > 3σ, compras_12m > 3σ)
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500
NOMES = ["Ana Silva", "Bruno Costa", "Carla Mendes", "Diego Santos", "Elena Ferreira",
         "Felipe Lima", "Gabriela Rocha", "Henrique Alves", "Isabela Dias", "João Pereira",
         "Karen Oliveira", "Lucas Martins", "Marina Ribeiro", "Nelson Cardoso", "Patrícia Gomes"]
SOBRENOMES = ["Souza", "Barbosa", "Teixeira", "Cunha", "Azevedo", "Moraes", "Castro", "Nunes"]
ESTADOS = ["SP", "RJ", "MG", "RS", "BA", "PR", "SC", "PE", "CE", "DF"]
EMAILS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "protonmail.com"]

os.makedirs("data", exist_ok=True)


def gerar_cpf() -> str:
    return f"{np.random.randint(100, 999)}.{np.random.randint(100, 999)}.{np.random.randint(100, 999)}-{np.random.randint(10, 99):02d}"


def gerar_email(nome: str) -> str:
    partes = nome.lower().split()
    return f"{partes[0]}.{partes[-1]}@{np.random.choice(EMAILS)}"


def gerar_telefone() -> str:
    ddd = np.random.choice([11, 21, 31, 41, 51, 61, 71, 81, 91])
    return f"({ddd}) 9{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}"


if __name__ == "__main__":
    print(f"Gerando {N} registros com falhas de qualidade...")

    hoje = date.today()
    data_atualizacao_base = hoje - timedelta(days=np.random.randint(1, 365))

    dados = []
    cpfs_usados = set()

    for i in range(N):
        nome = np.random.choice(NOMES)
        sobrenome = np.random.choice(SOBRENOMES)
        nome_completo = f"{nome} {sobrenome}"
        idade = np.random.randint(18, 75)
        estado = np.random.choice(ESTADOS)
        renda = round(np.random.lognormal(mean=8.5, sigma=0.6), 2)
        compras_12m = np.random.poisson(lam=renda / 1000)

        cpf = gerar_cpf()

        data_nasc = date(hoje.year - idade - np.random.randint(0, 2), np.random.randint(1, 13), np.random.randint(1, 29))

        dias_desde_att = np.random.randint(1, 1400)
        data_att = hoje - timedelta(days=dias_desde_att)
        data_cadastro = data_att - timedelta(days=np.random.randint(7, 1825))

        dados.append({
            "cliente_id": i + 1,
            "nome": nome_completo,
            "cpf": cpf,
            "email": gerar_email(nome),
            "email_alt": None,
            "telefone": gerar_telefone(),
            "data_nascimento": pd.Timestamp(data_nasc),
            "idade": idade,
            "estado": estado,
            "renda": round(renda, 2),
            "compras_12m": int(compras_12m),
            "data_cadastro": pd.Timestamp(data_cadastro),
            "data_atualizacao": pd.Timestamp(data_att),
        })

    df = pd.DataFrame(dados)

    # --- INJETAR FALHAS ---

    # Completude: nulos
    mask_tel = np.random.choice(N, size=int(N * 0.08), replace=False)
    df.loc[mask_tel, "telefone"] = None

    mask_email_alt = np.random.choice(N, size=int(N * 0.12), replace=False)
    # email_alt já é None, então 12% nulo natural + alguns extras
    df.loc[mask_email_alt[:10], "email_alt"] = "secundario@exemplo.com"  # alguns preenchidos
    # 12% permanece nulo

    mask_renda = np.random.choice(N, size=int(N * 0.04), replace=False)
    df.loc[mask_renda, "renda"] = None

    # Unicidade: CPF duplicado
    idx_dup = np.random.choice(N, size=2, replace=False)
    df.loc[idx_dup[1], "cpf"] = df.loc[idx_dup[0], "cpf"]

    # Consistência: idade negativa
    mask_idade_neg = np.random.choice(N, size=3, replace=False)
    df.loc[mask_idade_neg, "idade"] = -np.random.randint(1, 10)

    # Consistência: email inválido (sem @)
    mask_email_inv = np.random.choice(N, size=2, replace=False)
    for idx in mask_email_inv:
        df.loc[idx, "email"] = df.loc[idx, "email"].replace("@", "")

    # Consistência: data_nasc > data_cadastro
    mask_data = np.random.choice(N, size=4, replace=False)
    for idx in mask_data:
        df.loc[idx, "data_nascimento"] = df.loc[idx, "data_cadastro"] + pd.Timedelta(days=np.random.randint(30, 365))

    # Acurácia: outliers
    mask_out_renda = np.random.choice(N, size=2, replace=False)
    df.loc[mask_out_renda, "renda"] = df.loc[mask_out_renda, "renda"] * np.random.uniform(8, 15)

    mask_out_compras = np.random.choice(N, size=3, replace=False)
    df.loc[mask_out_compras, "compras_12m"] = np.random.randint(500, 1000)

    print(f"Salvando data/clientes.parquet ({len(df):,} linhas)")
    df.to_parquet("data/clientes.parquet", index=False)

    print("Resumo de falhas injetadas:")
    print(f"  Completude:  telefone {df['telefone'].isnull().sum()} nulos, email_alt {df['email_alt'].isnull().sum()} nulos, renda {df['renda'].isnull().sum()} nulos")
    print(f"  Frescor:     {(pd.to_datetime(df['data_atualizacao']) < pd.Timestamp(hoje - timedelta(days=730))).sum()} registros >2 anos sem atualização")
    print(f"  Unicidade:   {df['cpf'].duplicated().sum()} CPFs duplicados")
    print(f"  Consistencia: {df['email'].str.contains('@').sum()} emails válidos (esperado 498)")
    print(f"  Acurácia:    renda max={df['renda'].max():.0f} (média={df['renda'].mean():.0f}), compras max={df['compras_12m'].max()}")
