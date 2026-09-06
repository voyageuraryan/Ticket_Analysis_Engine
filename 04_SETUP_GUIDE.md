# SAP Ticket Triage Assistant — Setup Guide

**Document 4 of 5 — Installation and operation**
**Target:** a working local demo on Windows, macOS or Linux

This is a local demo. There is no deployment step, no container and no server to provision —
everything runs on one machine against two hosted LLM APIs.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Two setup paths](#2-two-setup-paths)
3. [Path A — run the existing demo](#3-path-a--run-the-existing-demo)
4. [Path B — rebuild everything from raw data](#4-path-b--rebuild-everything-from-raw-data)
5. [Getting API keys](#5-getting-api-keys)
6. [Configuration reference](#6-configuration-reference)
7. [Verifying the installation](#7-verifying-the-installation)
8. [Using the application](#8-using-the-application)
9. [Routine operations](#9-routine-operations)
10. [Troubleshooting](#10-troubleshooting)
11. [Security notes](#11-security-notes)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.10 – 3.12** | 3.9 works. **Avoid 3.13+** — several dependencies (ChromaDB and its transitive stack) have no wheels there yet, and installs fail in confusing ways |
| pip | recent | `python -m pip install --upgrade pip` |
| Disk | ~2 GB free | Vector store, embedding pickles and the RAFT dataset are ~70 MB, plus dependencies |
| RAM | 8 GB | The BM25 index is unpickled fully into memory |
| Network | outbound HTTPS | To `generativelanguage.googleapis.com` and `api.groq.com` |
| Google Gemini API key | required | Embeddings + user-facing LLM |
| Groq API key | required for bulk jobs | Augmentation, RAFT, fast batch evaluation |
| Chrome + Chromedriver | optional | Only for the Ivanti extraction scripts |

Check your Python version first — this is the most common cause of a failed install:

```bash
python --version
```

If it reports 3.13 or newer, install 3.12 alongside it and create the virtual environment with
that interpreter explicitly:

```bash
py -3.12 -m venv venv          # Windows
python3.12 -m venv venv        # Linux/Mac
```

---

## 2. Two setup paths

| | Path A — run the existing demo | Path B — rebuild from raw data |
|---|---|---|
| **When** | The `data/` directory already has indexes | Fresh clone, or you changed the corpus |
| **Time** | ~10 minutes | Several hours (augmentation dominates) |
| **API cost** | Per classification only | Thousands of calls |
| **Needs Groq keys** | No | Yes — several, for rotation |

Check which applies:

```bash
# Windows
dir data\vector_store\chroma.sqlite3 data\embeddings\bm25_index.pkl
# Linux/Mac
ls -l data/vector_store/chroma.sqlite3 data/embeddings/bm25_index.pkl
```

Both present → **Path A**. Either missing → **Path B**.

---

## 3. Path A — run the existing demo

### A1. Create and activate a virtual environment

```bash
# Windows (PowerShell or cmd)
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now show `(venv)`. If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### A2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This pulls ChromaDB and its transitive stack, so expect a few minutes.

### A3. Create your `.env`

```bash
# Windows
copy .env.example .env
# Linux / macOS
cp .env.example .env
```

Open `.env` and set at minimum:

```
GOOGLE_API_KEY=AIza...your_real_key
GROQ_API_KEY=gsk_...your_real_key
```

Everything else has a working default. See §5 for where to get keys.

### A4. Verify (see §7 for the full check)

```bash
python -c "from src.config import Config; Config.validate(); print('config OK')"
python scripts/05_run_evals.py --suite retrieval --strategy sparse_only --sample 10
```

The second command exercises config, the BM25 index, retrieval and metrics **without making a
single API call** — it is the fastest way to prove the installation is sound.

### A5. Launch

```bash
streamlit run app/streamlit_app_enhanced.py
```

Then open <http://localhost:8501>.

Or use the menu launchers, which check Python, activate `venv`, install missing dependencies,
warn about a missing `.env` and offer a choice of app:

```bash
start.bat        # Windows — or double-click it
./start.sh       # Linux / macOS  (chmod +x start.sh once)
```

---

## 4. Path B — rebuild everything from raw data

Do steps **A1–A3** first, then continue here.

> Budget several hours and **add Groq backup keys before you start**. Augmentation and RAFT
> generation are thousands of calls each; a single free-tier key will rate-limit and the job
> will crawl. With `GROQ_API_KEY_BACKUP1..5` set, rotation keeps it running unattended.

### B1. Place the raw data

```
data/raw/Historic_Data_CSV.xlsx
```

Required columns: `Incident, Service, Module, Status, Summary, Description, Priority, Team`.
`Module` must contain only: `Basis`, `HR & Payroll`, `Procurement`, `Connections`, `FICO`, `ABAP`.

### B2. Clean, augment, split

```bash
python scripts/01_prepare_data.py
```

The long step. Expect:

| Output | Rows |
|---|---:|
| `data/processed/cleaned_tickets.csv` | 6,497 |
| `data/processed/augmented_tickets.csv` | 12,302 |
| `data/processed/train_test_split/train.csv` | 8,611 |
| `.../val.csv` | 1,845 |
| `.../test.csv` | 1,846 |

**If augmentation keeps stalling on rate limits**, run it module by module instead — a stall
then costs one module's progress rather than the whole run:

```bash
python scripts/01a_augment_single_module.py HR_Payroll
python scripts/01a_augment_single_module.py Procurement
python scripts/01a_augment_single_module.py Connections
python scripts/01a_augment_single_module.py FICO
python scripts/01a_augment_single_module.py ABAP
python scripts/01b_combine_augmented.py
```

Each writes `data/processed/augmented_<Module>.csv`; the combine step merges them with the
cleaned originals and produces the splits.

### B3. Build the indexes

```bash
python scripts/02_build_vector_store.py
```

With two or more Gemini keys in `.env` (`GOOGLE_API_KEY_1`, `GOOGLE_API_KEY_2`), the rotating
builder is roughly N times faster:

```bash
python scripts/02c_build_vector_store_multikey.py
```

Expect:

```
data/embeddings/train_embeddings.pkl     (embedding cache)
data/embeddings/bm25_index.pkl           (keyword index)
data/vector_store/chroma.sqlite3         (+ HNSW segment directories)
log: "Vector store built with 8611 documents"
```

> **The embedding cache is load-bearing.** Re-running this script reuses
> `train_embeddings.pkl` and skips regeneration — which makes rebuilds fast. But if `train.csv`
> changed and you did **not** delete the pickle, you will index stale vectors against a new
> corpus. No error is raised; retrieval just quietly degrades. **Delete the pickle whenever the
> training corpus changes outside the feedback-loop script.**

### B4. RAFT dataset — optional, and genuinely optional

```bash
python scripts/03_create_raft_dataset.py
```

Hours, resumable from `data/raft/raft_checkpoint.json`. **Nothing in the serving path reads its
output** — the shipped system applies RAFT as a prompting pattern at inference time, not as a
fine-tuned model. Run this only if you intend to fine-tune later. See
`02_TECHNICAL_IMPLEMENTATION.md` §16, issue 2.

### B5. Verify and launch

Continue with §7, then §3 step A5.

---

## 5. Getting API keys

### Google Gemini (required)

1. Go to <https://aistudio.google.com/apikey>
2. Sign in and create an API key
3. Put it in `.env` as `GOOGLE_API_KEY=AIza...`

Used for embeddings (index build **and** every query) and for the user-facing classification LLM.

### Groq (required for bulk jobs)

1. Go to <https://console.groq.com/keys>
2. Create an API key
3. Put it in `.env` as `GROQ_API_KEY=gsk_...`

### Why you want several Groq keys

Free-tier limits are per key. Augmentation is thousands of calls and RAFT generation is
thousands more; a single key will spend most of its time rate-limited. The augmenter, RAFT
generator and classifier all rotate automatically across:

```
GROQ_API_KEY
GROQ_API_KEY_BACKUP1 .. GROQ_API_KEY_BACKUP5
```

On a rate limit the code advances to the next key; when all are exhausted it waits 60 seconds,
resets to the first and continues. The job never dies on a rate limit — worst case it slows down.

Three or more keys makes bulk work practical. **Path A does not need any of this.**

---

## 6. Configuration reference

All settings live in `.env` and are read by `src/config.py`. `.env.example` documents every
option inline. The ones you are most likely to touch:

| Variable | Default | Effect |
|---|---|---|
| `GOOGLE_API_KEY` | — | **Required.** `Config.validate()` fails without it |
| `GROQ_API_KEY` | — | Required for bulk scripts |
| `GROQ_API_KEY_BACKUP1..5` | — | Rotation pool |
| `LLM_PROVIDER` | `groq` | Provider for **batch scripts**. The Streamlit apps always use Gemini |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | **Changing this invalidates every stored vector** — delete the cache and rebuild |
| `LLM_MODEL` | `gemini-2.5-flash` | UI classification model |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Bulk model |
| `TEMPERATURE` | `0.2` | Low, for consistent classification |
| `TOP_K_RETRIEVAL` | `10` | Context documents per query |
| `RRF_K` | `60` | Fusion constant; 60 is the standard value |
| `API_DELAY_SECONDS` | `2.0` | Pause between batches. Lower only when rotating keys |
| `BATCH_SIZE` | `100` | Texts per embedding call (Gemini max) |
| `RANDOM_SEED` | `42` | **Keep fixed** so eval runs stay comparable |

Two gotchas worth internalising:

1. **`LLM_PROVIDER` does not affect the UI.** Both Streamlit apps hard-code Gemini. Setting it to
   `groq` and expecting the app to change will confuse you.
2. **`models/` prefix on `EMBEDDING_MODEL` is mandatory** for the installed
   `google.generativeai` library. The bare `text-embedding-004` form belongs to the newer
   `google.genai` library and will fail here.

---

## 7. Verifying the installation

Run these in order. Each one isolates a layer, so the first failure tells you where the problem is.

**1 — Config loads and the Gemini key is present**

```bash
python -c "from src.config import Config; Config.validate(); print('config OK')"
```

**2 — Data files are where they should be**

```bash
python -c "
from pathlib import Path
for p in ['data/processed/train_test_split/train.csv',
          'data/processed/train_test_split/test.csv',
          'data/embeddings/bm25_index.pkl',
          'data/vector_store/chroma.sqlite3']:
    print(('OK   ' if Path(p).exists() else 'MISS '), p)
"
```

**3 — The vector store is populated** (an empty collection returns nothing and raises no error —
this is the check that catches it)

```bash
python -c "
from src.config import Config
from src.vector_store import ChromaVectorStore
vs = ChromaVectorStore(persist_directory=str(Config.get_paths()['vector_store']))
vs.create_collection(name=Config.COLLECTION_NAME, reset=False)
print('documents in store:', vs.collection.count())
"
```

Expect ~8,611. **A count of 0 means the store was never built** — go to Path B step B3.

**4 — Retrieval works end to end, with zero API cost**

```bash
python scripts/05_run_evals.py --suite retrieval --strategy sparse_only --sample 10
```

This exercises config, the BM25 index, the retrieval stack, metrics and output writing without a
single API call.

**5 — The full stack, including the LLM** (one API call)

```bash
python -c "
from src.config import Config
from src.embeddings import GeminiEmbedder
from src.vector_store import ChromaVectorStore
from src.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from src.rag import SAPModuleClassifier
p = Config.get_paths()
emb = GeminiEmbedder(api_key=Config.GOOGLE_API_KEY, model_name=Config.EMBEDDING_MODEL)
vs = ChromaVectorStore(persist_directory=str(p['vector_store']))
vs.create_collection(name=Config.COLLECTION_NAME, reset=False)
sp = SparseRetriever(index_path=str(Config.DATA_DIR / 'embeddings' / 'bm25_index.pkl'))
hy = HybridRetriever(DenseRetriever(vs, emb), sp, k=Config.RRF_K)
clf = SAPModuleClassifier(llm_provider='gemini', api_key=Config.GOOGLE_API_KEY,
                          hybrid_retriever=hy, sap_modules=Config.SAP_MODULES,
                          model_name=Config.LLM_MODEL, temperature=Config.TEMPERATURE)
r = clf.predict('Cannot login to SAP Portal',
                'User receives an error message at the top of the browser page when logging in.')
print('module    :', r['module'])
print('confidence:', r['confidence'])
print('precedents:', len(r['similar_tickets']))
"
```

A sensible module (very likely `Basis`) with 5 precedents means everything is wired correctly.

**6 — The app starts**

```bash
streamlit run app/streamlit_app_enhanced.py
```

---

## 8. Using the application

### Classify & Assign

- **Single ticket** — enter a summary and description, optionally tick auto-assign, submit.
  You get the module, a confidence badge, step-by-step reasoning, the key indicator terms, the
  top 5 precedents, and the assigned engineer with their remaining daily capacity.
- **Batch** — upload a CSV/XLSX with `Summary` and `Description` columns; download the file with
  predictions appended.

Predictions marked **needs review** are the ones a human should check. Treat that flag as the
system's honest admission of uncertainty rather than noise.

### Employee Management

Three tabs: list, add, edit/delete. Each engineer gets a name, email, one or more covered
modules, a daily ticket cap and a status (`available` / `on_leave` / `busy`).

The roster starts empty. **Add at least one engineer per module you want auto-assignment to
cover** — otherwise every classification returns *unassigned*, which is correct behaviour but
looks like a bug during a demo.

### Assignment Dashboard

Today's totals, engineers available / at capacity / on leave, per-module remaining capacity, and
a manual reset button.

Note: the daily reset is **lazy, not scheduled**. It fires when the manager next loads its data
and sees a new date — there is no background timer.

### Settings

Displays the active configuration.

---

## 9. Routine operations

### Feedback cycle (improve the system from corrections)

```bash
python testing_data_extraction.py     # Ivanti → data/testing/*.xlsx  (needs Chrome)
python testing_data_classifier.py     # → outputs/testing_results/classified_tickets_*.xlsx
#   ... a specialist fills the validation column: Approved / Rejected + correct module ...
python testing_feedback_loop.py       # corrections → train.csv → rebuild indexes
```

The feedback script handles the embedding cache correctly — it embeds only the new rows.
**Restart Streamlit afterwards**, or `@st.cache_resource` will keep serving the old index for
the life of the process.

### Evaluation

```bash
python scripts/05_run_evals.py --suite retrieval --sample 300           # free-ish, run often
python scripts/05_run_evals.py --suite classification --slice original --sample 150
python scripts/05_run_evals.py --suite all --sample 200 --gate
```

See `05_EVALUATION.md`. Note especially §4 — the honest number comes from `--slice original`.

The legacy evaluator still works:

```bash
python scripts/04_evaluate_model.py
```

### Backup

Worth keeping before any rebuild:

```
data/processed/train_test_split/     the corpus, including feedback additions
data/embeddings/                     embedding cache + BM25 index
data/vector_store/                   the vector database
data/assignments/                    the roster
```

The feedback script already snapshots `*_backup_<epoch>.pkl` before rebuilding.

### Resetting the demo

```bash
# Clear the roster and daily counters
rm data/assignments/employees_master.json data/assignments/daily_assignments.json

# Force a full re-embed and index rebuild
rm data/embeddings/train_embeddings.pkl
python scripts/02_build_vector_store.py
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` fails on chromadb | Python 3.13+ | Use Python 3.10–3.12 (§1) |
| `GOOGLE_API_KEY not set in environment variables` | No `.env`, or run from the wrong directory | `cp .env.example .env`, add the key, run from the repo root |
| `ValueError: BM25 index not initialized` | Index never built | `python scripts/02_build_vector_store.py` |
| `Test data not found` | Data prep never run | `python scripts/01_prepare_data.py` |
| Vector store count is 0 | Store created but never populated | Rebuild (Path B, step B3) |
| Retrieval returns irrelevant precedents | Stale embedding cache | Delete `train_embeddings.pkl`, rebuild |
| Everything predicts `Unknown` | LLM output format drifted | Inspect `raw_response`; tighten the format block in `prompt_templates.py` |
| Bulk job crawls or stalls | Single rate-limited key | Add `GROQ_API_KEY_BACKUP1..5` |
| `429 / RateLimitError` in the UI | Gemini free-tier limit | Wait, or raise `API_DELAY_SECONDS` |
| Every ticket comes back *unassigned* | Empty roster, or nobody covers that module | Add engineers under Employee Management |
| App shows stale data after a rebuild | `@st.cache_resource` | Restart the Streamlit process |
| `Batch size ... greater than max batch size` | Unbatched Chroma insert | Already fixed — check `batch_size=5000` in `chroma_store.py` |
| PowerShell blocks `venv\Scripts\activate` | Execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Selenium scripts fail to start Chrome | Chrome/driver mismatch | `webdriver-manager` normally resolves it; update Chrome |
| Import errors running a script | Wrong working directory | Run everything from the repository root |

Logs are the fastest diagnosis:

```
outputs/logs/data_preparation.log
outputs/logs/build_vector_store.log
outputs/logs/create_raft.log
outputs/logs/evaluation.log
outputs/evals/<run_id>/eval.log
```

---

## 11. Security notes

**Rotate the keys currently in `.env` before sharing this repository.** The working copy contains
live Gemini and Groq keys, including one stray bare key on its own line. `.gitignore` does
exclude `.env`, but the file has been sitting in a shared folder and keys were quoted in some of
the older documentation that these five documents replace. Treat all of them as exposed.

Other points worth stating plainly:

- **Ticket text is sent to third-party APIs.** Every classification transmits the summary and
  description to Google (and to Groq for bulk jobs). The corpus contains employee names, staff
  IDs and phone numbers. Cleaning removes emails and URLs, not names. Raise this before any
  real-world use.
- **No authentication on the demo app.** Anyone who can reach `localhost:8501` has full access,
  including roster editing. Do not expose the port beyond the local machine.
- **Roster JSON has no locking.** Two processes writing at once will lose data.
- **Never commit `.env`.** Use `.env.example` as the shareable template.
