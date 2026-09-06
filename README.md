<div align="center">

# 🎫 SAP Ticket Triage Assistant

### *Retrieval-Augmented Ticket Analysis Engine for SAP Service Desks*

Route incoming SAP support tickets to the right module — and the right engineer — in seconds,
with a human-readable justification for every decision.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FFB300?style=for-the-badge">
  <img alt="Gemini" src="https://img.shields.io/badge/Google%20Gemini-Embeddings%20%2B%20LLM-4285F4?style=for-the-badge&logo=google&logoColor=white">
  <img alt="Groq" src="https://img.shields.io/badge/Groq-Bulk%20Inference-F55036?style=for-the-badge">
</p>

<p>
  <img alt="Accuracy" src="https://img.shields.io/badge/Accuracy-0.830-success?style=flat-square">
  <img alt="Macro F1" src="https://img.shields.io/badge/Macro%20F1-0.805-success?style=flat-square">
  <img alt="ECE" src="https://img.shields.io/badge/ECE-0.070-success?style=flat-square">
  <img alt="Baseline" src="https://img.shields.io/badge/vs%20always--Basis%20baseline-%2B16.8%20pts-blue?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/status-demo%20%2F%20PoC-orange?style=flat-square">
</p>

</div>

---

## 📌 The Problem

A SAP support desk receives incident tickets through an ITSM tool. Every ticket must be routed to
exactly one of **six SAP functional areas** before anyone can start work — and today that is a
manual decision made by a human reading free-text written by end users.

| Cost | What it looks like on the desk |
|---|---|
| ⏱️ **Latency** | The ticket waits in an unassigned queue while the SLA clock runs |
| 🔁 **Misroutes** | *"I can't log in to the SAP portal"* is a **Basis** ticket — nothing in that sentence says so |
| 🧠 **Knowledge concentration** | Correct routing lives in the heads of a few senior people |
| ⚖️ **Uneven workload** | Even after the right module, *which engineer* gets it is ad hoc |

And the obvious fix — "just train a classifier" — collapses on the real data:

<div align="center">

| Module | Historic tickets | Share |
|---|---:|---:|
| **Basis** | 4,302 | ██████████████ 66.2% |
| **HR & Payroll** | 935 | ███ 14.4% |
| **Procurement** | 378 | █ 5.8% |
| **Connections** | 354 | █ 5.4% |
| **FICO** | 266 | ▊ 4.1% |
| **ABAP** | 262 | ▊ 4.0% |

</div>

> A model that predicts `Basis` for everything scores **66% accuracy** while being operationally
> worthless. Four of the six classes have only a few hundred examples each.

---

## 💡 The Approach

**Don't train a classifier — retrieve precedent and reason over it.**

When a ticket arrives, the engine finds the most similar historic tickets using *both*
meaning-based and keyword-based search, hands those precedents to an LLM alongside the new
ticket, and asks it to decide, justify, and state its own confidence. A separate assignment
engine then picks an available engineer.

```
                    ┌──────────────────────────────────────────────┐
   New ticket  ───▶ │   🧹  Cleaning  ·  strip emails, URLs, IDs   │
  (Summary +        └──────────────────────┬───────────────────────┘
   Description)                            │
                    ┌──────────────────────▼───────────────────────┐
                    │           🔍  HYBRID RETRIEVAL               │
                    │  ┌────────────────┐   ┌───────────────────┐  │
                    │  │ Dense (vector) │   │ Sparse (BM25)     │  │
                    │  │ meaning,       │   │ exact codes,      │  │
                    │  │ paraphrase     │   │ T-codes, Z-progs  │  │
                    │  └────────┬───────┘   └─────────┬─────────┘  │
                    │           └──────┬──────────────┘            │
                    │        Reciprocal Rank Fusion (RRF)          │
                    └──────────────────────┬───────────────────────┘
                                           │  top-k precedents
                    ┌──────────────────────▼───────────────────────┐
                    │        🤖  RAFT-STYLE LLM REASONING          │
                    │   module · confidence · step-by-step         │
                    │   reasoning · key indicators · precedent     │
                    │   analysis · needs-review flag               │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────▼───────────────────────┐
                    │      👤  ASSIGNMENT ENGINE                   │
                    │  module coverage · daily cap · availability  │
                    └──────────────────────────────────────────────┘
```

### Why this beats a classifier

| Hard property of the data | How the engine handles it |
|---|---|
| **Severe class imbalance** | Retrieval learns no class prior — a 262-example module retrieves its own precedents and competes on equal footing at query time |
| **Vocabulary ≠ label** | Semantic embeddings match *"can't log in to portal"* to historic Basis logon tickets with **zero shared keywords** |
| **Exact identifiers matter** | BM25 keeps literal T-codes, error numbers and `Z*` program names that embeddings blur |
| **Noisy templated text** | A cleaning stage strips signatures, emails, URLs and ticket IDs before indexing |
| **Black boxes aren't trusted** | Every answer carries reasoning, key indicator terms, the precedents themselves, and an honest **needs review** flag |

---

## ✨ Features

- 🎯 **Six-way SAP module classification** — Basis · HR & Payroll · Procurement · Connections · FICO · ABAP
- 🔀 **Hybrid retrieval** — dense semantic + BM25 sparse, fused with Reciprocal Rank Fusion
- 🧾 **Explainable by construction** — reasoning steps, key indicators, and the top-5 precedents shown alongside every prediction
- 🚦 **Calibrated confidence** — low-confidence predictions are flagged `needs review` instead of silently guessing (ECE **0.070**)
- 👥 **Workload-aware assignment** — routes to an engineer who covers the module, is available, and is under their daily cap
- 📦 **Batch mode** — upload CSV/XLSX, download predictions appended
- 🔄 **Feedback loop** — human corrections flow back into the corpus and the index is rebuilt, so the system learns from its mistakes
- 📊 **Full evaluation suite** — retrieval, classification, calibration, consistency and explanation-quality metrics with pass/fail gates
- 🧪 **LLM augmentation** — minority modules rebalanced before indexing to keep retrieval fair

---

## 📊 Measured Results

> 200 sampled test tickets · RAFT prompting · hybrid retrieval · `top_k=10`

<div align="center">

| Metric | Value | Gate | |
|---|---:|---:|:--:|
| Accuracy | **0.830** | ≥ 0.78 | ✅ |
| Macro F1 | **0.805** | ≥ 0.75 | ✅ |
| Weighted F1 | 0.829 | — | — |
| Expected Calibration Error | **0.070** | ≤ 0.10 | ✅ |

</div>

**Macro F1 is the primary metric** — with 66% of the corpus in one class, plain accuracy rewards
getting Basis right and barely penalises getting Connections wrong.

### Per module

| Module | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Basis | 0.952 | 0.857 | **0.902** | 70 |
| Procurement | 0.783 | 1.000 | **0.878** | 18 |
| FICO | 0.844 | 0.871 | **0.857** | 31 |
| HR & Payroll | 0.732 | 0.882 | **0.800** | 34 |
| ABAP | 0.792 | 0.704 | **0.745** | 27 |
| Connections | 0.706 | 0.600 | **0.649** | 20 |

<details>
<summary><b>📉 Where it still gets things wrong (click to expand)</b></summary>

<br>

- **Connections is the weakest row — 8 of 20 missed**, 3 to FICO and 4 to ABAP. Connections
  tickets describe integrations *between* systems, so they legitimately carry the vocabulary of
  whatever they connect. This is genuine boundary ambiguity, not sloppiness.
- **Basis leaks into HR & Payroll** — 8 of 70, the largest off-diagonal cell. Likely user-account
  and authorisation tickets raised by HR users.
- **Procurement over-claims** — recall 1.000 at precision 0.783: nothing is missed, but four other
  modules' tickets get pulled in.
- **Caveat:** this baseline runs over the whole test set, which includes the ~47% LLM-generated
  slice — so it is **optimistic**. See `05_EVALUATION.md` for the honest breakdown and the
  prioritised improvement list.

</details>

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | **3.10 – 3.12** | ⚠️ Avoid 3.13+ — ChromaDB's stack has no wheels there yet |
| Disk | ~2 GB | Vector store, embedding pickles, RAFT dataset ≈ 70 MB + deps |
| RAM | 8 GB | The BM25 index is unpickled fully into memory |
| Google Gemini API key | required | Embeddings + user-facing LLM |
| Groq API key | required for bulk jobs | Augmentation, RAFT, fast batch evaluation |

```bash
python --version    # check this first — it is the #1 cause of failed installs
```

### Run the demo

```bash
# 1 · Clone
git clone <your-repo-url>
cd Ticket_Analysis_Engine

# 2 · Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# 3 · Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4 · Configure
copy .env.example .env         # Windows
cp .env.example .env           # Linux / macOS
#    ...then add your GOOGLE_API_KEY and GROQ_API_KEY

# 5 · Verify
python scripts/05_run_evals.py --suite retrieval --strategy sparse_only --sample 10

# 6 · Launch 🎉
streamlit run app/streamlit_app_enhanced.py
```

> 💡 **First run tip:** the engineer roster starts **empty**. Add at least one engineer per module
> under **Employee Management**, or every classification returns *unassigned* — correct behaviour,
> but it looks like a bug during a demo.

<details>
<summary><b>🔧 Rebuild everything from raw data (click to expand)</b></summary>

<br>

Place your historic export at `data/raw/Historic_Data_CSV.xlsx`, then:

```bash
# Clean, split
python scripts/01_prepare_data.py

# Rebalance minority modules via LLM augmentation (long-running, Groq-bound)
python scripts/01a_augment_single_module.py HR_Payroll
python scripts/01a_augment_single_module.py Procurement
python scripts/01a_augment_single_module.py Connections
python scripts/01a_augment_single_module.py FICO
python scripts/01a_augment_single_module.py ABAP
python scripts/01b_combine_augmented.py

# Build the dense + sparse indexes
python scripts/02_build_vector_store.py
python scripts/02c_build_vector_store_multikey.py

# RAFT dataset — optional
python scripts/03_create_raft_dataset.py
```

</details>

---

## 🗂️ Project Structure

```
Ticket_Analysis_Engine/
├── app/                          🖥️  Streamlit UI
│   └── streamlit_app_enhanced.py     classify · assign · manage · dashboard
├── src/
│   ├── config.py                 ⚙️  central configuration
│   ├── data_preparation/         🧹  cleaner · augmenter · splitter
│   ├── embeddings/               🧬  Gemini + local embedders
│   ├── vector_store/             🗄️  ChromaDB wrapper
│   ├── retrieval/                🔍  dense · sparse · hybrid (RRF)
│   ├── raft/                     📚  distractor selection · dataset generator
│   ├── rag/                      🤖  pipeline · prompt templates
│   ├── assignment/               👥  assignment engine · employee manager
│   └── evaluation/               📊  evaluator · metrics
├── scripts/                      ▶️  numbered end-to-end pipeline (01 → 05)
├── data/                         💾  raw · processed · embeddings · vector_store
├── testing_data_extraction.py    🌐  Ivanti resolved-ticket extraction (Selenium)
├── testing_data_classifier.py    🧪  batch-classify extracted tickets
├── testing_feedback_loop.py      🔄  fold human corrections back into the corpus
└── report_automation.py          📨  scheduled reporting
```

### 📚 Documentation

| Document | Covers |
|---|---|
| [`01_FUNCTIONAL_SPECIFICATION.md`](01_FUNCTIONAL_SPECIFICATION.md) | The problem, the solution, and why it works — business view |
| [`02_TECHNICAL_IMPLEMENTATION.md`](02_TECHNICAL_IMPLEMENTATION.md) | How it's built step by step, including problems hit and fixes applied |
| [`03_ARCHITECTURE_AND_SCENARIOS.md`](03_ARCHITECTURE_AND_SCENARIOS.md) | Architecture diagrams, component wiring, runtime scenarios |
| [`04_SETUP_GUIDE.md`](04_SETUP_GUIDE.md) | Full installation, configuration, operations and troubleshooting |
| [`05_EVALUATION.md`](05_EVALUATION.md) | The eval suite, every metric explained, and current measured results |

---

## 🧰 Tech Stack

| Layer | Choice |
|---|---|
| **Embeddings** | Google Gemini (`gemini-embedding-001`) · local embedder fallback |
| **Vector store** | ChromaDB (persistent, HNSW) |
| **Sparse search** | `rank-bm25` |
| **Fusion** | Reciprocal Rank Fusion (`k=60`) |
| **Reasoning LLM** | Gemini 2.5 Flash (interactive) · Groq Llama 3.1 (bulk, with automatic key rotation) |
| **Local LLM** | Ollama (optional) |
| **UI** | Streamlit |
| **Data** | pandas · numpy · openpyxl |
| **Eval** | scikit-learn · matplotlib · seaborn |
| **Automation** | Selenium (Ivanti extraction) |

---

## 🔄 Feedback Loop

The engine improves from its own mistakes rather than from a retraining run:

```
  Resolved tickets ──▶ batch classify ──▶ specialist validates
       (Ivanti)                           (Approved / Rejected + correct module)
                                                     │
       rebuilt index ◀── add corrections to corpus ◀──┘
```

```bash
python testing_data_extraction.py     # pull resolved tickets
python testing_data_classifier.py     # classify them
#   ...specialist fills the validation column...
python testing_feedback_loop.py       # fold corrections in, rebuild the index
```

---

## 📈 Evaluation

```bash
python scripts/05_run_evals.py --suite retrieval --sample 300              # cheap, run often
python scripts/05_run_evals.py --suite classification --slice original --sample 150
python scripts/05_run_evals.py --suite all --sample 200 --gate             # CI-style gates
```

Five suites — **retrieval**, **classification**, **calibration**, **consistency** and
**explanation quality** — because end-to-end accuracy hides *which* stage failed. The
`context_hit_rate` vs `accuracy` pair is the core diagnostic: high context + low accuracy means a
reasoning problem; low context means a retrieval problem.

---

## 🔐 Security & Scope Notes

- **Status: demo / proof-of-concept.** Not deployed to production.
- **No Outlook / Entra ID / Azure AD integration is included** in this repository — that
  integration requires confidential tenant secrets and is withheld for company privacy.
- `.env` is gitignored. Never commit API keys — use `.env.example` as the template.
- Ticket data, embeddings, the vector store and generated outputs are gitignored by design.

---

<div align="center">

**Built for the SAP service desk at nxzen** · Author: **Sai Aryan Nampally**

*Retrieval over training. Precedent over prediction. Explanation over confidence.*

</div>
