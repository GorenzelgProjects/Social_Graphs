# Future Work & Roadmap

> **Placement**: Put this file at the **repo root** as `FUTURE_WORK.md`. Add a link to it in `README.md` under **Table of Contents**, near “Team Workflow.”
>
> **Who is this for?** Everyone collaborating on the project. Treat this as the living backlog.

---

## Table of Contents
- [1. Scope & Goals](#1-scope--goals)
- [2. Network Science Extensions](#2-network-science-extensions)
- [3. NLP & Discourse Extensions](#3-nlp--discourse-extensions)
- [4. Experimental Design & Statistics](#4-experimental-design--statistics)
- [5. Simulation Enhancements (AgentSociety)](#5-simulation-enhancements-agentsociety)
- [6. Datasets & Sampling](#6-datasets--sampling)
- [7. Infrastructure & Reproducibility](#7-infrastructure--reproducibility)
- [8. Visualization & Reporting](#8-visualization--reporting)
- [9. Ethics, Risk, & Governance](#9-ethics-risk--governance)
- [10. Actionable To‑Do Lists](#10-actionable-to-do-lists)
- [11. Milestones & Timeline Template](#11-milestones--timeline-template)
- [12. Appendix: Suggested Libraries & Module Stubs](#12-appendix-suggested-libraries--module-stubs)

---

## 1. Scope & Goals
Extend the baseline benchmark (structural + linguistic/affective fidelity) to **deeper network structure**, **richer discourse analysis**, and **stronger statistical guarantees**. Target deliverables: a robust pipeline that can (a) detect meaningful mismatches between simulated and human behavior; (b) quantify uncertainty; and (c) guide **model iteration** for more human-like simulations.

---

## 2. Network Science Extensions
**Structural complexity & dynamics** beyond the baseline:

- **Degree mixing & rich-club**: Compute rich-club coefficients and compare curves.
- **Reciprocity** (directed graphs) and **edge weight distributions** (reply intensity).
- **k-core / k-truss decompositions** for cohesion; **bow-tie structure** in directed graphs.
- **Motif & triad census** for local interaction patterns; compare motif profiles.
- **Spectral properties**: eigenvalue spectra, algebraic connectivity, spectral gap.
- **Assortativity by attributes**: subreddit/topic, role, agent persona traits.
- **Community stability**: track community membership across time windows, NMI/ARI.
- **Temporal networks**: sliding-window graphs; latency distributions; burstiness.
- **Multiplex** (topic/thread layers) and **bipartite** (user–thread) projections.
- **Link prediction**: AUC/precision@k for future replies; triadic closure signals.
- **Null models**: configuration model, degree-preserving rewires; baseline expectations.

**Example snippets** (NetworkX):
```python
# Reciprocity (directed)
import networkx as nx
recip = nx.reciprocity(G)  # fraction of mutual dyads

# k-core & rich-club
core = nx.core_number(G.to_undirected())
rich = nx.rich_club_coefficient(G.to_undirected(), normalized=True)

# Triad census (on directed)
triads = nx.triadic_census(G)

# Spectral gap (undirected)
L = nx.normalized_laplacian_matrix(G.to_undirected())
# convert to dense for small graphs or use scipy.sparse.linalg.eigsh on large
```

---

## 3. NLP & Discourse Extensions
**Beyond TF–IDF + VADER:**

- **Contextual embeddings**: Sentence/paragraph embeddings (e.g., `sentence-transformers`) for semantic similarity, clustering, and retrieval of matched human/sim threads.
- **Neural sentiment / emotion**: Compare VADER (lexicon) with transformer sentiment; add emotion categories.
- **Topic modeling**: LDA/NMF (gensim, scikit-learn) or **BERTopic** (topic evolution over time).
- **Discourse acts**: Question, answer, agreement, disagreement, moderation, escalation; dialog-act tagging.
- **Toxicity & politeness**: Classifiers for toxicity/politeness; match to Reddit moderation norms.
- **Pragmatics**: Sarcasm cues, hedging, stance; ambiguity and uncertainty markers.
- **Conversation dynamics**: reply depth distribution; branching factor; lifespan of threads; turn-taking regularity; **response latency** per dyad.
- **Style/dialect**: Style distance (perplexity under style LM), lexical diversity (TTR, MTLD), burstiness of novel n‑grams.

**Example snippets**:
```python
# Sentence embeddings (sentence-transformers)
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer("all-MiniLM-L6-v2")
E_reddit = model.encode(reddit_messages["text"].tolist(), show_progress_bar=False)
E_agents = model.encode(agent_messages["text"].tolist(), show_progress_bar=False)
# cosine similarities, clustering, etc.

# BERTopic (topic modeling with embeddings)
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(ngram_range=(1,2), min_df=10)
topic_model = BERTopic(vectorizer_model=vectorizer_model)
topics, _ = topic_model.fit_transform(reddit_messages["text_clean"].tolist())
```

---

## 4. Experimental Design & Statistics
- **Matching & controls**: Topic-matching between AgentSociety and Reddit threads; control for thread size, time-of-day, and subreddit norms.
- **Multiple testing**: FDR control across many metrics.
- **Permutation & bootstrap**: Non-parametric tests for metric differences; CIs.
- **Graph-aware tests**: Compare to **null models** (degree-preserving rewires) for significance.
- **Effect sizes**: Cliff’s delta, Cohen’s d, CLES.
- **Robustness**: Sensitivity to sampling, preprocessing thresholds, alternate tokenization.
- **Ablations**: Toggle agent parameters (personas, memory, temperature) and re-benchmark.

---

## 5. Simulation Enhancements (AgentSociety)
- **Agent heterogeneity**: Personas with stable traits; role distributions; newcomer churn.
- **Moderation & norms**: Agents that enforce rules; model deletion/downvoting analogs.
- **Memory & context windows**: Longer/shorter memory; cross-thread recall.
- **Prompting strategies**: Few-shot exemplars based on subreddit norms; self-critique loops.
- **Exogenous events**: Seed shocks that trigger coordinated activity; topic shifts.
- **Export hooks**: Ensure structured logs (message id, reply_to, timestamps, persona ID, topic), plus **event logs** (joins, leaves, bans).

---

## 6. Datasets & Sampling
- **More subreddits**: Pick communities with varied norms (Q&A, news, hobbies).
- **Stratified sampling**: Match size/age of threads; balanced topic mix.
- **Temporal slices**: Week-by-week windows; pre/post event comparisons.
- **Human filtering**: Remove bots and mass-posters where detectable.
- **Data cards**: Document provenance, filters, and caveats.

---

## 7. Infrastructure & Reproducibility
- **Data versioning** with DVC (or Git LFS) for large artifacts.
- **Hydra/omegaconf**-style configs; single CLI entrypoint.
- **Pre-commit**: `black`, `ruff`, `isort`, `pyproject.toml`.
- **Type hints** + `mypy` for core modules.
- **Caching**: Memoize heavy computations (embeddings, community detection).
- **CI**: GitHub Actions to run lint + a tiny smoke test on PRs.
- **Experiment logging**: Weights & Biases or MLflow for metrics/artifacts.

---

## 8. Visualization & Reporting
- **Standard figures**: degree CCDFs, motif profiles, community size distributions, temporal activity plots.
- **Embeddings**: UMAP of message embeddings colored by dataset/community.
- **Dashboards**: Plotly/Dash or Streamlit with controls for subreddit, time, metric.
- **Tables**: unified `benchmark.csv` with effect sizes, p-values, and CIs; auto-render LaTeX tables for the paper.

---

## 9. Ethics, Risk, & Governance
- **Bias & fairness**: Evaluate whether agents amplify/mitigate toxicity or bias vs. human baselines.
- **PII/ToS**: Respect dataset licenses and platform ToS; strip PII.
- **Transparency**: Clear labeling of simulations; avoid over-claiming representativeness.
- **Reproducible claims**: Every figure/table should be regenerable from a single config.

---

## 10. Actionable To‑Do Lists
Below are prioritized checklists (P0 = near-term critical, P1 = important, P2 = nice-to-have).

### A) Data & Schema
- [ ] **P0**: Add schema validator for unified `Message` records (pydantic or pandera).
- [ ] **P0**: Deterministic export from AgentSociety (IDs, timestamps, reply_to, persona).
- [ ] **P1**: Stratified Reddit sampling (by subreddit, thread size, period).
- [ ] **P1**: Data card markdown auto-generator.

### B) Simulation
- [ ] **P0**: Configurable personas/roles; seed scripts under `configs/sim_config.yaml`.
- [ ] **P1**: Moderation agent and deletion events in logs.
- [ ] **P2**: Event-shock scenarios (topic shifts) with toggles.

### C) Network Analysis
- [ ] **P0**: Implement reciprocity, k-core, rich-club, triad census in `src/metrics/structure.py`.
- [ ] **P1**: Temporal sliding-window metrics; community stability (NMI/ARI).
- [ ] **P1**: Degree-preserving rewiring null model and expectation bands.
- [ ] **P2**: Link prediction baselines (common neighbors, Adamic/Adar, resource allocation).

### D) NLP & Discourse
- [ ] **P0**: `sentence-transformers` embeddings + cosine similarity analysis.
- [ ] **P0**: Transformer sentiment (compare with VADER); emotion tags.
- [ ] **P1**: Topic modeling (LDA/NMF; BERTopic optional).
- [ ] **P1**: Dialog-act tagger (rule-based baseline → ML).
- [ ] **P2**: Politeness/toxicity classifiers; sarcasm heuristics.

### E) Benchmarking & Stats
- [ ] **P0**: Bootstrap CIs + permutation tests for key metrics; FDR correction.
- [ ] **P1**: Null-model comparisons with expectation envelopes on plots.
- [ ] **P2**: Effect size dashboard with Cliff’s delta & CLES.

### F) Infra & Repro
- [ ] **P0**: CLI: `python -m sgproj.run --config configs/analysis.yaml` (single entrypoint).
- [ ] **P0**: Pre-commit + CI with smoke tests on tiny samples.
- [ ] **P1**: Caching layer (joblib) for embeddings and community detection.
- [ ] **P2**: Experiment tracking (W&B/MLflow) hooks.

### G) Visualization & Reporting
- [ ] **P0**: Save standardized plots to `reports/figures/` with consistent filenames.
- [ ] **P1**: UMAP/TSNE plots of embeddings split by dataset/community.
- [ ] **P2**: Streamlit/Plotly dashboard prototype.

### H) Writing & Paper
- [ ] **P0**: Auto-generate LaTeX tables from `benchmark.csv`.
- [ ] **P1**: Methods appendix for all metrics & tests.
- [ ] **P2**: Ethics/bias audit section with concrete checks.

---

## 11. Milestones & Timeline Template
Use/modify this template in the course’s milestone system.

- **M1 (Week 1–2)**: Data & sim schema stable; baseline metrics pass on toy data.
- **M2 (Week 3–4)**: Extended network & NLP metrics; bootstrap/permutation tests added.
- **M3 (Week 5)**: Topic-matched evaluation; null-model envelopes; ablations.
- **M4 (Week 6)**: Interactive figures + reproducible paper tables; writing pass.

---

## 12. Appendix: Suggested Libraries & Module Stubs
**Python libs** (optional additions):
- Network: `python-louvain`, `igraph` (performance), `graph-tool` (advanced, C++), `networkit` (fast), `tqdm`.
- NLP: `sentence-transformers`, `transformers`, `gensim`, `bertopic`, `umap-learn`.
- Stats: `scipy`, `statsmodels`, `scikit-posthocs`.
- Infra: `hydra-core`, `omegaconf`, `pydantic`/`pandera`, `joblib`, `dvc`, `mlflow` or `wandb`.

**Module stubs to add** (consistent with README structure):
```
src/
├── data/
│   ├── validate_schema.py          # pandera/pydantic checks
│   └── sampling.py                 # stratified samples, temporal windows
├── metrics/
│   ├── temporal.py                 # sliding windows, stability (NMI/ARI)
│   └── null_models.py              # degree-preserving rewires, envelopes
├── nlp/
│   ├── contextual.py               # sentence-transformers, transformer sentiment
│   ├── topics.py                   # LDA/NMF/BERTopic wrappers
│   └── discourse.py                # dialog acts, toxicity/politeness
└── viz/
    ├── embeddings.py               # UMAP/TSNE scatter
    └── dashboards.py               # Streamlit/Plotly app
```

---


