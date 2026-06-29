# Feature Engine Agent

Pipeline de feature engineering orquestrado por agentes LangGraph.

## Execução real — 200K transações

```text
=== Feature Engine Agent ===
Input:  data/vendas.parquet
Output: features.duckdb
Início: 23:25:24

[ingest] Carregando dados...
[ingest] 200,000 linhas carregadas
[validate] Validando schema...
[validate] Schema OK
[transform] Limpando dados...
[transform] 0 linhas removidas, 200,000 restantes
[features] Calculando features...
  - ticket_medio...
  - frequencia_cliente...
  - share_categoria...
  - crescimento_mom...
  - recencia_cliente...
  - tendencia_7d...
  - sazonalidade_dia...
  - top_produtos...
[features] 8 tabelas geradas, 71,021 linhas totais
[save] Salvando features em features.duckdb...
[save] Arquivo salvo: 2.5 MB

=== Pipeline concluído ===
Status: done
Features geradas: 8 tabelas
Tempo total: 7.6s
```

### Dataset gerado

```text
Período:  2025-01-01 → 2025-12-31
Lojas:    10
Produtos: 100
Clientes: 5,000
Receita:  R$ 583,968,919.97
Ticket:   R$ 2,919.84 (médio)
```

### Features produzidas

| Tabela | Linhas | Descrição |
|---|---|---:|---|
| `ticket_medio` | 3,650 | Ticket médio por loja x dia |
| `frequencia_cliente` | 57,831 | Compras, gasto, lojas visitadas por cliente x mês |
| `share_categoria` | 100 | % de receita e margem por categoria em cada loja |
| `crescimento_mom` | 120 | Crescimento mês-sobre-mês por loja |
| `recencia_cliente` | 5,000 | Dias desde última compra por cliente |
| `tendencia_7d` | 3,650 | Média móvel 7 dias com % de desvio |
| `sazonalidade_dia` | 70 | Índice sazonal por dia da semana e loja |
| `top_produtos` | 600 | Top 5 produtos por receita em cada loja x mês |

### Amostras dos resultados

**Ticket médio por loja (top 5):**
```
 loja_id   ticket_medio
       9       2,954.17
       6       2,947.85
       8       2,947.05
       3       2,934.75
       7       2,924.86
```

**Sazonalidade — loja 1:**
```
dia_da_semana  indice_sazonal
      Domingo            0.99
      Segunda            0.98
        Terça            1.04
       Quarta            1.01
       Quinta            1.02
        Sexta            0.99
       Sábado            0.99
```

**Crescimento MoM — loja 1 (últimos 3 meses):**
```
     mes    receita  crescimento_mom_pct
 2025-12 5,036,463               +7.69%
 2025-11 4,676,911               -4.07%
 2025-10 4,875,102               +2.15%
```

**Top produtos — loja 1, jan/2025:**
```
 produto_id   receita
         94   251,325
         73   211,273
         85   167,729
         77   156,906
         59   140,270
```

---

## Fluxo

```
┌─────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌─────────┐
│ Ingest  │───→│ Validate │───→│ Transform  │───→│ Feature  │───→│  Save   │
│ (CSV)   │    │ (schema) │    │ (clean)    │    │ (calc)   │    │(DuckDB) │
└─────────┘    └──────────┘    └────────────┘    └──────────┘    └─────────┘
     │              │                │                 │               │
     └──────────────┴────────────────┴─────────────────┴───────────────┘
                              checkpoint (retomada de falha)
```

## Uso

```bash
# Gera dados sintéticos
python data/generate.py

# Executa o pipeline
python agent.py

# Testes
pytest tests/ -v
```

