# Benchmarking Social Fidelity: AgentSociety × Reddit

*A DTU Social Graphs & Social Network Analysis exam project*

**TL;DR.** We simulate online conversations with LLM-based agents (AgentSociety) and compare them to real Reddit interactions (ConvoKit). We then benchmark **structural** network properties (degree distributions, clustering, path lengths, assortativity, modularity) and **linguistic/affective** behavior (TF–IDF features, VADER sentiment, conversational dynamics). Everything is packaged so the whole team can reproduce, extend, and collaborate.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Links](#key-links)
- [Repo Structure](#repo-structure)
- [Quick Start](#quick-start)
- [Data](#data)
  - [Reddit (human) data](#reddit-human-data)
  - [AgentSociety (simulated) data](#agentsociety-simulated-data)
- [Unified Schema](#unified-schema)
- [Pipelines](#pipelines)
  - [1) Build Graphs](#1-build-graphs)
  - [2) Network Analysis](#2-network-analysis)
  - [3) NLP Analysis](#3-nlp-analysis)
  - [4) Benchmark & Statistics](#4-benchmark--statistics)
- [Notebooks](#notebooks)
- [Reproducibility](#reproducibility)
- [Team Workflow](#team-workflow)
- [Ethics & Data Use](#ethics--data-use)
- [How to Cite / References](#how-to-cite--references)

---

## Project Overview
This project evaluates how **human-like** agent-based simulations are when we look at both **who-talks-to-whom** (network structure) and **how they talk** (language and affect). We:

1. **Simulate** a Reddit-style discussion process with **AgentSociety** agents.
2. **Sample** matched **human** conversations from ConvoKit’s Reddit corpora.
3. **Construct** interaction graphs (users/agents as nodes; replies/mentions as edges).
4. **Measure** structural fidelity (degree distributions, clustering, path lengths, assortativity/homophily, and community modularity).
5. **Measure** linguistic/affective fidelity (VADER sentiment, TF–IDF topic vectors, conversational dynamics).
6. **Compare** distributions/statistics between AgentSociety and Reddit.

The output is a compact benchmark indicating where simulations match human behavior and where gaps remain.

---

## Key Links
- **Course · Project Assignments:** https://github.com/suneman/socialgraphs2025/wiki/Project-Assignments
- **AgentSociety docs:** https://agentsociety.readthedocs.io/
- **ConvoKit (Reddit corpora):** https://convokit.cornell.edu/documentation/subreddit.html
- **Network Science (Barabási):** https://networksciencebook.com/
- **NLTK book:** https://www.nltk.org/book/
- **NLTK API (VADER):** https://www.nltk.org/api/nltk.sentiment.vader.html
- **NetworkX:** https://networkx.org/
- **scikit-learn (TF–IDF):** https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

---

## Repo Structure
```
.
├── README.md
├── environment.yml              # optional: conda env for reproducibility
├── requirements.txt             # pin runtime deps (pip)
├── configs/
│   ├── sim_config.yaml          # AgentSociety scenario & export settings
│   ├── reddit_config.yaml       # subreddits, time ranges, sample sizes
│   └── analysis.yaml            # graph/NLP metric settings & seeds
├── data/
│   ├── raw/
│   │   ├── reddit/              # downloaded ConvoKit corpora (unzipped)
│   │   └── agentsociety/        # exported simulation logs
│   └── processed/
│       ├── reddit/              # unified JSONL/Parquet
│       └── agentsociety/
├── notebooks/
│   ├── 01_explore_reddit.ipynb
│   ├── 02_explore_agents.ipynb
│   ├── 03_graph_analysis.ipynb
│   └── 04_nlp_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── download_reddit.py
│   │   ├── simulate_agents.py
│   │   ├── harmonize_schema.py
│   │   └── build_graph.py
│   ├── nlp/
│   │   ├── preprocess.py
│   │   ├── tfidf.py
│   │   └── sentiment.py
│   ├── metrics/
│   │   ├── structure.py
│   │   ├── comparisons.py
│   │   └── stats.py
│   └── viz/
│       ├── netplots.py
│       └── tables.py
├── scripts/
│   ├── quickstart.sh            # one-liners to run the pipeline end-to-end
│   ├── run_all.sh
│   └── make_figures.sh
├── reports/
│   ├── figures/
│   └── tables/
└── LICENSE
```

---

## Quick Start
### 0) Prerequisites
- Python ≥ 3.11
- macOS / Linux / Windows (WSL recommended on Windows)
- (Optional) Conda / mamba

### 1) Create and activate an environment
**Conda (recommended):**
```bash
mamba create -n agentsociety-env python=3.11 -y
mamba activate agentsociety-env
```

**Or pip/venv:**
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
python - <<'PY'
import nltk
for pkg in ["vader_lexicon", "stopwords", "punkt", "wordnet"]:
    nltk.download(pkg)
print("Downloaded NLTK data.")
PY
```
> If you use community detection via Louvain, also `pip install python-louvain`.

### 3) Configure the project
Edit files under `configs/` to choose subreddits, simulation parameters, and analysis options.

### 4) Run the end-to-end demo
```bash
bash scripts/quickstart.sh
```
This will: download a small Reddit sample, run (or load) a minimal AgentSociety simulation, harmonize schemas, build graphs, compute metrics, and output comparison tables/plots under `reports/`.

---

## Data
### Reddit (human) data
- Install ConvoKit and download a test-sized corpus:
```python
from convokit import Corpus, download
corpus = Corpus(filename=download("reddit-corpus-small"))
corpus.print_summary_stats()
```
- For a **specific subreddit**, use the `subreddit-<Name>` pattern (e.g., `subreddit-Cornell`):
```python
from convokit import Corpus, download
cornell = Corpus(filename=download("subreddit-Cornell"))
a2c = Corpus(filename=download("subreddit-ApplyingToCollege"))
merged = cornell.merge(a2c)
merged.print_summary_stats()
```
- The ConvoKit Reddit corpora provide utterance-level fields like `id`, `speaker`, `conversation_id`, `reply_to`, `timestamp`, and `text` (plus metadata like `score`, `permalink`).

**Storage tip.** After downloading, export to JSONL/Parquet so later steps don’t depend on ConvoKit’s on-disk format.

### AgentSociety (simulated) data
- Follow the AgentSociety **Quick Start** to install prerequisites and run a minimal scenario.
- Configure the scenario so that each **message** or **reply** is exported with at least:
  - `message_id`, `conversation_id`, `reply_to`, `timestamp`, `text`
  - `sender_id`, `receiver_id` (or thread parent to infer edges)
  - optional: `topic`, `role`, `community`, `location`
- Use the built-in **metrics collection** / **message interception** features or write a thin logger to CSV/JSONL.

**Output location.** Save logs under `data/raw/agentsociety/`.

---

## Unified Schema
To compare human and simulated data, we harmonize both into a single **Message** schema (JSONL/Parquet):

```json
{
  "message_id": str,
  "conversation_id": str,
  "reply_to": str|null,
  "timestamp": int|str,       // unix or ISO8601
  "sender_id": str,
  "receiver_id": str|null,    // for reply graphs; else infer from reply_to
  "text": str,
  "platform": "reddit"|"agent",
  "community": str|null,      // subreddit or simulated community
  "meta": { ... }             // optional extras (score, permalinks, roles...)
}
```

Run:
```bash
python -m src.data.harmonize_schema \
  --reddit data/raw/reddit/ --agents data/raw/agentsociety/ \
  --out data/processed/
```

---

## Pipelines

### 1) Build Graphs
We construct interaction graphs from the unified messages:
- **Directed graph**: edge `u→v` if `u` replies to `v` (from `reply_to` or resolved parent)
- **Undirected weighted graph**: symmetrize counts of mutual interactions

```bash
python -m src.data.build_graph \
  --in data/processed/reddit/ --out data/processed/reddit_graph.parquet \
  --mode directed
python -m src.data.build_graph \
  --in data/processed/agentsociety/ --out data/processed/agents_graph.parquet \
  --mode directed
```

**Notes**
- Compute on the **giant (weakly) connected component** for path-based metrics.
- Keep both **weighted** and **unweighted** variants for robustness.

### 2) Network Analysis
Core structural metrics (via NetworkX + community detection):

```python
import networkx as nx
from collections import Counter

# G is a directed reply graph
GC = max(nx.weakly_connected_components(G.to_undirected()), key=len)
H = G.subgraph(GC).copy()

# Degree & distribution
in_deg = [d for n, d in H.in_degree()]
out_deg = [d for n, d in H.out_degree()]

# Clustering (undirected)
C = nx.average_clustering(H.to_undirected())

# Shortest paths (undirected for small-worldness)
L = nx.average_shortest_path_length(H.to_undirected())

# Assortativity
r_degree = nx.degree_pearson_correlation_coefficient(H.to_undirected())
# Attribute assortativity, if "community" or "role" is present on nodes
# r_attr = nx.attribute_assortativity_coefficient(H, "community")

# Community detection & modularity (requires python-louvain)
from community import community_louvain
parts = community_louvain.best_partition(H.to_undirected())
communities = {}
for n, g in parts.items():
    communities.setdefault(g, set()).add(n)
Q = nx.algorithms.community.quality.modularity(H.to_undirected(), communities.values())
```

Export all stats to `reports/tables/structure.csv`.

### 3) NLP Analysis
Preprocessing and features:

```python
# src/nlp/preprocess.py
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop = set(stopwords.words("english"))
lem = WordNetLemmatizer()

def clean(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"@[\w_]+", "", t)
    t = re.sub(r"[^a-z0-9\s?!.]", " ", t)
    return " ".join(lem.lemmatize(w) for w in t.split() if w not in stop)
```

**VADER sentiment**
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
msg["sent"] = msg["text"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
```

**TF–IDF (bag-of-words)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=20_000, ngram_range=(1,2), min_df=5)
X_reddit = vec.fit_transform(reddit_messages["text_clean"])  # vocabulary
X_agents = vec.transform(agent_messages["text_clean"])       # same space
```

**Conversational dynamics (examples)**
- **Response latency** per dyad from timestamps
- **Question share** (`"?"` density)
- **Burstiness** (inter-event CV)

Export per-message/per-user aggregates to `reports/tables/nlp.csv`.

### 4) Benchmark & Statistics
We compare **distributions** and **scalar** summaries between Reddit and AgentSociety:

- Two-sample **KS tests** for degree distributions, sentiment, message lengths
- **Bootstrap CIs** for clustering, assortativity, modularity

```python
import numpy as np
from scipy.stats import ks_2samp

D, p = ks_2samp(in_deg_reddit, in_deg_agents)

# simple bootstrap CI example
rng = np.random.default_rng(7)
vals = np.array(clustering_reddit)
ci = np.percentile([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)], [2.5, 97.5])
```

Write a single comparison table with all endpoints to `reports/tables/benchmark.csv` and generate plots under `reports/figures/`.

---

## Notebooks
- `01_explore_reddit.ipynb` — load ConvoKit, peek at fields, sample conversations, basic stats
- `02_explore_agents.ipynb` — load AgentSociety logs, check schema & volumes
- `03_graph_analysis.ipynb` — degree, clustering, path lengths, assortativity, communities
- `04_nlp_analysis.ipynb` — TF–IDF, VADER, distribution comparison, dynamics

Each notebook is lightweight; the heavy lifting lives in `src/`.

---

## Reproducibility
- **Seeds:** configure RNG seeds in `analysis.yaml` (used by numpy, Python, and any randomized routines)
- **Environment:** use `environment.yml` or `requirements.txt` and record `pip freeze > reports/pip-freeze.txt`
- **Data footprints:** store raw data outside git (large!), keep processed Parquet/CSV small where possible
- **Prompts & configs:** version any AgentSociety prompts/configs under `configs/`

Example `requirements.txt` (pin as needed):
```
convokit>=3.5
networkx>=3.5
nltk>=3.9
scikit-learn>=1.5
scipy>=1.13
pandas>=2.2
python-louvain>=0.16
matplotlib>=3.9
pyarrow>=17
```

---

## Team Workflow
- **Branches:** `feature/<thing>`, `fix/<bug>`, `exp/<study>`
- **Issues & PRs:** open an issue for analysis ideas; reference issue IDs in PRs
- **Code style:** `black`, `isort`, and `ruff` (optional) via pre-commit hooks
- **Data contracts:**
  - Don’t commit anything under `data/raw/`
  - Only small derived artifacts under `data/processed/`
  - Scripts should be **idempotent** and **config-driven**
- **Milestones** (align with course wiki):
  1) **Assignment A:** 1-minute concept video + preliminary stats
  2) **Assignment B:** 5-page paper + explainer notebook

---

## Ethics & Data Use
- Respect platform ToS and dataset licenses
- Remove PII where appropriate; don’t attempt re-identification
- Be clear that simulations are **simulations**; avoid over-claiming external validity

---

## How to Cite / References
- AgentSociety documentation (installation, quick start, metrics)
- ConvoKit Reddit corpora (subreddit-specific and small sample)
- Network Science (Barabási) for structural metrics
- NLTK & VADER for lexicon-based sentiment
- scikit-learn TF–IDF for lexical features

If you use this repository in academic work, please cite the course project and any upstream datasets/software you relied on.

