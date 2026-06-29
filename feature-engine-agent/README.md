# Feature Engine Agent

Pipeline de feature engineering orquestrado por agentes LangGraph.

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

Cada nó é um step atômico. Se um step falha, o checkpoint permite retomar dali, não do zero.

## Uso

```bash
# Gera dados sintéticos de vendas
python data/generate.py

# Executa o pipeline
python agent.py

# Com parâmetros customizados
python agent.py --input data/vendas.parquet --output features.duckdb

# Testes
pytest tests/ -v
```

## Estrutura dos dados

O gerador cria dados de vendas de uma rede de lojas:

| Coluna | Tipo | Exemplo |
|---|---|---|
| `data_venda` | date | 2025-01-15 |
| `loja_id` | int | 1 a 10 |
| `produto_id` | int | 1 a 100 |
| `categoria` | string | "Eletrônicos", "Vestuário" |
| `quantidade` | int | 1 a 20 |
| `valor_unitario` | float | 9.90 a 999.90 |
| `valor_total` | float | quantidade * valor_unitario |
| `cliente_id` | int | 1 a 5000 |
| `canal_venda` | string | "online", "fisico", "app" |
| `regiao` | string | "Norte", "Sul", "Sudeste", "Nordeste", "Centro-Oeste" |

## Features geradas

O agente calcula features por loja, produto e período:

| Feature | Descrição | Granularidade |
|---|---|---|
| `ticket_medio` | Valor total / número de vendas | Loja x Dia |
| `frequencia_cliente` | Compras por cliente no período | Cliente x Mês |
| `share_categoria` | % de vendas por categoria | Loja x Mês |
| `crescimento_mom` | Crescimento mês-sobre-mês | Loja x Mês |
| `recencia_cliente` | Dias desde última compra | Cliente |
| `tendencia_7d` | Média móvel 7 dias de vendas | Loja x Dia |
| `sazonalidade_dia` | Índice de vendas por dia da semana | Loja x Dia da Semana |
| `top_produtos` | Top 5 produtos por receita | Loja x Mês |
