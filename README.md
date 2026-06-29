<h1 align="center">LangGraph Recipes</h1>
<p align="center">
  <em>Exemplos práticos de agentes em produção com LangGraph — ML, dados, feature engineering, ETL</em>
  <br/>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-00154E?style=flat-square" /></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11%2B-00154E?style=flat-square&logo=python&logoColor=F17405" /></a>
</p>

---

Cada receita é um projeto autocontido com agente LangGraph funcional, dados de exemplo e testes.

---

## Receitas

| # | Projeto | Descrição | Stack |
|---|---|---|---|
| 1 | [**feature-engine-agent**](feature-engine-agent/) | Pipeline de feature engineering: ingest → validate → transform → feature → save. 8 features, 200K transações, 7.6s | LangGraph, DuckDB, Pandas |
| 2 | [**ml-pipeline-agent**](ml-pipeline-agent/) | Pipeline ML end-to-end com checkpoints: load → split → train → evaluate → select → deploy. 2 modelos, R²=0.988 | LangGraph, scikit-learn |
| 3 | [**data-quality-agent**](data-quality-agent/) | Squad de 5 agentes em paralelo: completude, frescor, unicidade, consistência, acurácia. Score 45/100, 0.1s | LangGraph, Pandas, NumPy |
| ⬜ | **etl-orchestrator** | Spec ETL em linguagem natural → DAG Airflow monitorada por agente | LangGraph, Airflow |

---

## Estrutura

```
langgraph-recipes/
├── feature-engine-agent/
│   ├── agent.py          # Grafo do agente (nós + arestas)
│   ├── features.py       # Lógica de feature engineering
│   ├── data/generate.py  # Gerador de dados sintéticos
│   ├── tests/
│   └── README.md
├── ml-pipeline-agent/    # ✅ load → split → train → evaluate → select → deploy
├── data-quality-agent/   # ✅ 5 agentes paralelos: completude, frescor, unicidade, consistencia, acuracia
└── etl-orchestrator/     # (em breve)
```

---

## Quickstart

```bash
git clone https://github.com/generalrodolfao/langgraph-recipes.git
cd langgraph-recipes/feature-engine-agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/generate.py   # gera dados sintéticos
python agent.py           # executa o pipeline de features
```

---

<p align="center">
  <a href="https://github.com/generalrodolfao"><img src="https://img.shields.io/badge/github-generalrodolfao-00154E?style=flat-square&logo=github&logoColor=F17405" /></a>
</p>
