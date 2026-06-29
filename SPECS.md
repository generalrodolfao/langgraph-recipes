# Specs — LangGraph Recipes

Especificações técnicas dos 3 agentes em produção.

---

## 1. Feature Engine Agent

**Objetivo:** Pipeline de feature engineering orquestrado por agente LangGraph, gerando features analíticas a partir de dados transacionais brutos.

**Fluxo:** `ingest → validate → transform → feature_calc → save`

**Stack:** LangGraph, DuckDB, Pandas

**Dados:** 200.000 transações sintéticas de varejo (10 lojas, 100 produtos, 5.000 clientes, 12 meses)

**Features geradas (8 tabelas):**

| Tabela | Granularidade | Métricas |
|---|---|---|
| `ticket_medio` | Loja × Dia | Receita total, nº vendas, ticket, clientes únicos |
| `frequencia_cliente` | Cliente × Mês | Compras, gasto, lojas visitadas |
| `share_categoria` | Loja × Categoria | % receita, receita, margem |
| `crescimento_mom` | Loja × Mês | Receita, crescimento % vs mês anterior |
| `recencia_cliente` | Cliente | Dias desde última compra |
| `tendencia_7d` | Loja × Dia | Receita, média móvel 7d, % desvio |
| `sazonalidade_dia` | Loja × Dia da Semana | Receita média, índice sazonal |
| `top_produtos` | Loja × Mês | Top 5 por receita |

**Resultado:** 200K linhas processadas em 7.6s, 71K linhas de features, DuckDB 2.5MB

---

## 2. ML Pipeline Agent

**Objetivo:** Pipeline de machine learning end-to-end com checkpoint para retomada de falha, prevendo faturamento do próximo mês por loja.

**Fluxo:** `load → split → train → evaluate → select → deploy`

**Stack:** LangGraph, scikit-learn (RandomForest, LinearRegression), Pandas

**Dados:** 200 lojas, 23 meses, 29 features incluindo categóricas com one-hot encoding

**Modelos:**

| Modelo | R² | RMSE | MAE |
|---|---|---|---|
| RandomForest | 0.9881 | 14,324 | 8,768 |
| LinearRegression | 0.9881 | 14,355 | 9,431 |

**Resultado:** 4.600 amostras, treino/teste 80/20, modelo salvo em 9.5MB, pipeline completo em 1.5s

---

## 3. Data Quality Agent

**Objetivo:** Squad de 5 agentes especialistas auditando qualidade de dados em paralelo (fan-out/fan-in), gerando diagnóstico unificado com score e alertas.

**Fluxo:** `load → (completude | frescor | unicidade | consistencia | acuracia) → diagnostico`

**Stack:** LangGraph, Pandas, NumPy

**Dados:** 500 clientes com falhas intencionais (nulos, duplicatas, violações, outliers, datas antigas)

**Dimensões auditadas:**

| Agente | O que audita | Problemas encontrados |
|---|---|---|
| Completude | % nulos por coluna | 2 colunas críticas (telefone 8%, email_alt 98%) |
| Frescor | Data da última atualização | 49% sem atualização há >2 anos |
| Unicidade | CPF duplicado | 1 duplicata |
| Consistência | Regras de negócio | 9 violações (idade negativa, email inválido, data inválida) |
| Acurácia | Outliers (z-score) | 7 outliers (renda, compras) |

**Resultado:** Score 45/100, severidade ALTA, 5 alertas (3 críticos, 2 warnings), 0.1s

---

## Arquitetura Comum

### Padrão de Estado

Cada agente usa `TypedDict` para estado serializável (msgpack-safe) e `_runtime` dict para objetos não-serializáveis (DataFrame, conexão DuckDB, modelos sklearn).

```python
class EstadoAgent(TypedDict):
    input_path: str
    status: str
    ...

_runtime: dict = {}  # df, con, modelos
```

### Checkpoint

`MemorySaver` em todos os pipelines. Se um nó falha, o checkpoint permite retomar do último estado válido sem reprocessar.

### Padrão Fan-out/Fan-in (Data Quality Agent)

```python
builder.add_edge("load", "completude")
builder.add_edge("load", "frescor")
builder.add_edge("load", "unicidade")
builder.add_edge("load", "consistencia")
builder.add_edge("load", "acuracia")
# Cada agente retorna apenas seu resultado parcial
builder.add_edge("completude", "diagnostico")
# ... todos convergem no nó diagnostico
```

Cada agente retorna apenas seu campo (`{"resultado_completude": ...}`), evitando conflito de escrita em canais paralelos.

### Serialização

Numpy types (`int64`, `float64`) não são msgpack-safe. Função `_serializar()` converte recursivamente para tipos Python nativos antes de retornar ao estado do LangGraph.

---

## Lições Aprendidas

1. **TypedDict do LangGraph só aceita tipos serializáveis.** DataFrame, conexão DB, modelos ML precisam ficar em `_runtime` separado.
2. **Nós paralelos não podem retornar `{**state, ...}`.** Cada nó deve retornar apenas seu campo específico quando em fan-out.
3. **DuckDB in-memory é ideal para pipelines analíticos.** Zero configuração, performance comparável a Postgres.
4. **Checkpoint via MemorySaver é suficiente para dev.** Em produção, trocar por `PostgresSaver` ou `SqliteSaver`.
5. **Dados sintéticos com falhas reais são essenciais para testar qualidade.** Sem isso, o data-quality-agent não exercita os caminhos de erro.
