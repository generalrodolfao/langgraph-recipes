# Data Quality Agent

Squad de agentes LangGraph que audita qualidade de dados em 5 dimensões.

Cada agente é especialista em uma dimensão. Executam em paralelo e convergem em um diagnóstico unificado.

## Fluxo

```
                         ┌───────────────┐
                    ┌───→│  Completude   │───┐
                    │    └───────────────┘   │
                    │    ┌───────────────┐   │
                    ├───→│   Frescor     │───┤
┌──────┐    ┌──────┤    └───────────────┘   ├───→┌─────────┐    ┌────────┐
│ Load │───→│ Audit│───→┌───────────────┐   │    │ Report  │───→│ Alerts │
└──────┘    └──────┤    │  Unicidade    │───┤    └─────────┘    └────────┘
                   │    └───────────────┘   │
                   │    ┌───────────────┐   │
                   ├───→│ Consistência  │───┤
                   │    └───────────────┘   │
                   │    ┌───────────────┐   │
                   └───→│   Acurácia    │───┘
                        └───────────────┘
```

## Execução

```bash
python data/generate.py          # gera dados com falhas reais
python agent.py                  # audita qualidade
python agent.py --threshold 0.05 # tolerância de 5% para alertas
pytest tests/ -v                 # 10 testes
```

## Dimensões

| Agente | O que audita | Exemplos |
|---|---|---|
| Completude | % de nulos por coluna | emails faltando, telefones nulos |
| Frescor | Data da última atualização | dados de 2022 em 2026 |
| Unicidade | Linhas e IDs duplicados | cliente duplicado, PK repetida |
| Consistência | Regras de negócio | idade negativa, data fim < data início |
| Acurácia | Outliers estatísticos | vendas 100x acima da média |

## Resultado típico

```text
=== Data Quality Audit ===
Amostras: 500 clientes de e-commerce

[completude]   3 colunas com >5% nulos (telefone: 8.2%, email_alt: 11.6%, renda: 4.0%)
[frescor]      ⚠ 12.4% dos registros sem atualização há >2 anos
[unicidade]    ✅ OK — sem duplicatas
[consistencia] ⚠ 9 violações: idade negativa (3), data inválida (4), email inválido (2)
[acuracia]     ⚠ 5 outliers: renda (2), compras_12m (3)

=== Diagnóstico ===
Score qualidade: 72/100
Severidade: MÉDIA
Alertas: 3 críticos, 2 warnings
```
