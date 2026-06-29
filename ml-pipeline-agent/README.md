# ML Pipeline Agent

Pipeline de machine learning end-to-end orquestrado por agentes LangGraph com checkpoint para retomada de falha.

Tarefa: prever o faturamento do próximo mês por loja usando features históricas.

## Fluxo

```
┌────────┐   ┌───────┐   ┌──────────┐   ┌───────────┐   ┌────────┐   ┌────────┐
│  Load  │──→│ Split │──→│  Train   │──→│ Evaluate  │──→│ Select │──→│ Deploy │
│ (CSV)  │   │ 80/20 │   │ 3 models │   │ R², RMSE  │   │  best  │   │  .pkl  │
└────────┘   └───────┘   └──────────┘   └───────────┘   └────────┘   └────────┘
     │           │              │               │              │            │
     └───────────┴──────────────┴───────────────┴──────────────┴────────────┘
                              checkpoint (retomada de falha)
```

## Execução

```bash
python data/generate.py          # gera dados sintéticos de vendas
python agent.py                  # executa o pipeline ML completo
python agent.py --target loja_id  # filtra por loja específica
pytest tests/ -v                 # 8 testes
```

## Modelos treinados

| Modelo | Hiperparâmetros |
|---|---|
| `RandomForestRegressor` | n_estimators=100, max_depth=10 |
| `XGBRegressor` | n_estimators=100, max_depth=6, learning_rate=0.1 |
| `LinearRegression` | default |

## Métricas de avaliação

| Métrica | Descrição |
|---|---|
| R² | Coeficiente de determinação (0→1, quanto maior melhor) |
| RMSE | Raiz do erro quadrático médio (na unidade da target) |
| MAE | Erro absoluto médio (robusto a outliers) |

## Resultado real

```text
=== ML Pipeline Agent ===
Input:  data/vendas_ml.parquet
Output: model.pkl
Início: 23:36:39

[load]    200 lojas, 23 meses de histórico, 29 features
[split]   Treino: 3,680 amostras, Teste: 920 amostras
[train]   RandomForest... LinearRegression... (1.5s)
[evaluate] RandomForest  R²=0.9881  RMSE=14,324  MAE=8,768
[evaluate] LinearReg     R²=0.9881  RMSE=14,355  MAE=9,431
[select]  Melhor modelo: RandomForest (R²=0.9881)
[deploy]  Modelo salvo: model.pkl (9.5 MB)

=== Pipeline concluído ===
Status: done | Tempo: 1.5s
```
```
