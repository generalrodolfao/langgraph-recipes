<h1 align="center">LangGraph Recipes</h1>
<p align="center">
  <em>Exemplos práticos de agentes em produção com LangGraph — ML, dados, feature engineering, qualidade</em>
  <br/>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-00154E?style=flat-square" /></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11%2B-00154E?style=flat-square&logo=python&logoColor=F17405" /></a>
  <a href="https://generalrodolfao.github.io/langgraph-recipes/"><img src="https://img.shields.io/badge/GitHub_Pages-landing-00154E?style=flat-square&logo=github" /></a>
  <a href="SPECS.md"><img src="https://img.shields.io/badge/specs-SDD-00154E?style=flat-square" /></a>
</p>

---

Cada receita é um projeto autocontido com agente LangGraph funcional, dados de exemplo, testes e CI.

---

## Resultados Consolidados

| # | Projeto | Pipeline | Métricas | Tempo |
|---|---|---|---|---|
| 1 | **Feature Engine** | ingest → validate → transform → feature → save | 200K transações, 8 features, 71K linhas | 7.6s |
| 2 | **ML Pipeline** | load → split → train → evaluate → select → deploy | R²=0.988, 200 lojas, 29 features | 1.5s |
| 3 | **Data Quality** | 5 agentes em paralelo → diagnóstico | Score 45/100, 5 alertas (3 críticos) | 0.1s |

[SPECS.md](SPECS.md) — especificações técnicas, arquitetura, lições aprendidas.  
[Landing Page](https://generalrodolfao.github.io/langgraph-recipes/) — dashboard visual com resultados de execução real.

---

## Estrutura

```
langgraph-recipes/
├── feature-engine-agent/    # ✅ DuckDB, Pandas, 8 features
├── ml-pipeline-agent/       # ✅ scikit-learn, RandomForest, R²=0.988
├── data-quality-agent/      # ✅ 5 agentes paralelos, fan-out/fan-in
├── etl-orchestrator/        # ⬜ em breve
├── SPECS.md                 # Especificações e arquitetura
├── index.html               # Landing page (GitHub Pages)
└── .github/workflows/       # CI: 3 jobs, 30 testes
```

---

## Quickstart

```bash
git clone https://github.com/generalrodolfao/langgraph-recipes.git

# Feature Engine
cd feature-engine-agent && python data/generate.py && python agent.py

# ML Pipeline
cd ml-pipeline-agent && python data/generate.py && python agent.py

# Data Quality
cd data-quality-agent && python data/generate.py && python agent.py
```

---

## Arquitetura

Todos os pipelines compartilham o mesmo padrão:

- **TypedDict** para estado serializável (msgpack-safe)
- **`_runtime` dict** para objetos não-serializáveis (DataFrame, conexão DB, modelos)
- **MemorySaver** para checkpoint — retoma do último nó se houver falha
- **Fan-out/fan-in** no Data Quality Agent: 5 agentes paralelos convergindo em diagnóstico

---

<p align="center">
  <a href="https://github.com/generalrodolfao"><img src="https://img.shields.io/badge/github-generalrodolfao-00154E?style=flat-square&logo=github&logoColor=F17405" /></a>
  <a href="mailto:rodolfo@dtsqd.com"><img src="https://img.shields.io/badge/-rodolfo@dtsqd.com-00154E?style=flat-square&logo=gmail&logoColor=F17405" /></a>
</p>
