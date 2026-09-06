# SAP Ticket Triage Assistant — Technical Implementation

**Document 2 of 5 — Engineering view**
**Audience:** Developers, reviewers, anyone who has to rebuild or extend this system

This document covers the technical flow, the implementation process in the order it actually
happened, every intermediate artifact produced along the way, the problems hit in the middle
and the fixes applied, and a step-by-step guide to reproducing the whole thing from scratch.

Read `01_FUNCTIONAL_SPECIFICATION.md` first for the problem and the intent.
Read `03_ARCHITECTURE_AND_SCENARIOS.md` for diagrams and component wiring.

---

## Table of Contents

1. [Technology stack and why each piece](#1-technology-stack-and-why-each-piece)
2. [Repository layout](#2-repository-layout)
3. [The complete technical flow](#3-the-complete-technical-flow)
4. [Stage 1 — Data cleaning](#4-stage-1--data-cleaning)
5. [Stage 2 — LLM augmentation and rebalancing](#5-stage-2--llm-augmentation-and-rebalancing)
6. [Stage 3 — Splitting](#6-stage-3--splitting)
7. [Stage 4 — Embeddings and the two indexes](#7-stage-4--embeddings-and-the-two-indexes)
8. [Stage 5 — Hybrid retrieval and RRF](#8-stage-5--hybrid-retrieval-and-rrf)
9. [Stage 6 — RAFT dataset generation](#9-stage-6--raft-dataset-generation)
10. [Stage 7 — The RAG classification pipeline](#10-stage-7--the-rag-classification-pipeline)
11. [Stage 8 — Assignment engine](#11-stage-8--assignment-engine)
12. [Stage 9 — Streamlit applications](#12-stage-9--streamlit-applications)
13. [Stage 10 — The feedback loop](#13-stage-10--the-feedback-loop)
14. [Intermediate artifacts reference](#14-intermediate-artifacts-reference)
15. [Problems faced and fixes applied](#15-problems-faced-and-fixes-applied)
16. [Open technical issues](#16-open-technical-issues)
17. [Step-by-step rebuild guide](#17-step-by-step-rebuild-guide)

---

## 1. Technology stack and why each piece

| Layer | Choice | Why this and not the alternative |
|---|---|---|
| Language | Python 3.9+ | Ecosystem for everything below |
| Embeddings | Google Gemini `models/gemini-embedding-001` | Free tier, batch endpoint accepting up to 100 texts per call, good quality on short technical text. A local sentence-transformer was the alternative; the hosted batch API was faster to get working and needed no GPU. |
| Vector DB | ChromaDB (persistent, local) | Zero-infrastructure, file-backed, HNSW index, cosine space. Pinecone/Weaviate would need an account and network; for a demo that is friction with no benefit. |
| Keyword search | `rank_bm25` (BM25Okapi) | Pure-Python, no server, pickles to disk. Elasticsearch would be the production answer and massive overkill here. |
| Fusion | Reciprocal Rank Fusion, k=60 | Score-free fusion. Dense returns cosine similarity, sparse returns unbounded BM25 scores — they are not comparable, so fusing on rank is the only safe merge without a calibration step. |
| LLM (quality path) | Gemini 2.5 Flash | User-facing classification and reasoning quality |
| LLM (throughput path) | Groq `llama-3.1-8b-instant` | Bulk jobs. Roughly an order of magnitude faster per request than the free Gemini path, which is what makes an 8,611-example generation job finish at all. |
| UI | Streamlit | Fastest path from Python functions to a usable multi-page app |
| Config | `python-dotenv` + a single `Config` class | One import, one source of truth, environment-overridable |
| Logging | `loguru` | File + console sinks with levels, one-line setup |
| Metrics | `scikit-learn`, `matplotlib`, `seaborn` | Standard classification metrics and plots |
| Scraping | `selenium` + `webdriver-manager` | Ivanti has no usable API in this environment |

### The two-provider decision

This is the most important architectural choice in the project and it is driven entirely by
free-tier rate limits.

```
                 quality matters          throughput matters
                 ┌──────────────┐         ┌──────────────────┐
   Gemini  ◄──── │ user-facing  │         │ 12k augmentations│ ────► Groq
                 │ classification│         │ 8.6k RAFT gens   │       + key rotation
                 │ embeddings   │         │ 200-sample evals │
                 └──────────────┘         └──────────────────┘
```

Both are driven through one `SAPModuleClassifier` with an `llm_provider` switch, so nothing
downstream cares which one is in use.

---

## 2. Repository layout

```
History Tool/
├── src/
│   ├── config.py                       # Single Config class, env-backed
│   ├── data_preparation/
│   │   ├── cleaner.py                  # TicketCleaner
│   │   ├── augmenter.py                # TicketAugmenter (Groq/Gemini/Ollama)
│   │   └── splitter.py                 # DataSplitter (stratified 70/15/15)
│   ├── embeddings/
│   │   ├── gemini_embedder.py          # GeminiEmbedder — batch + fallback
│   │   └── local_embedder.py           # Local alternative (unused in main path)
│   ├── vector_store/
│   │   └── chroma_store.py             # ChromaVectorStore — batched insert, cosine
│   ├── retrieval/
│   │   ├── dense_retriever.py          # DenseRetriever
│   │   ├── sparse_retriever.py         # SparseRetriever (BM25, pickled)
│   │   └── hybrid_retriever.py         # HybridRetriever — RRF fusion
│   ├── raft/
│   │   ├── generator.py                # RAFTGenerator + ParallelRAFTGenerator
│   │   └── distractor_selector.py      # Tiered distractor / golden selection
│   ├── rag/
│   │   ├── pipeline.py                 # SAPModuleClassifier — the core
│   │   └── prompt_templates.py         # System / RAG / RAFT prompts
│   ├── evaluation/
│   │   ├── evaluator.py                # ModelEvaluator — runs, plots, reports
│   │   └── metrics.py                  # metrics + calibration (ECE)
│   └── assignment/
│       ├── employee_manager.py         # Roster CRUD, daily counters, reset
│       └── assignment_engine.py        # Load-balanced selection
├── scripts/
│   ├── 01_prepare_data.py              # clean → augment → split (one shot)
│   ├── 01a_augment_single_module.py    # augment ONE module (rate-limit friendly)
│   ├── 01b_combine_augmented.py        # merge per-module outputs → split
│   ├── 02_build_vector_store.py        # embed → Chroma + BM25
│   ├── 02c_build_vector_store_multikey.py  # same, rotating N Gemini keys
│   ├── 03_create_raft_dataset.py       # RAFT dataset generation
│   ├── 04_evaluate_model.py            # original evaluation entry point
│   └── 05_run_evals.py                 # RAG eval suite (see 05_EVALUATION.md)
├── app/
│   ├── streamlit_app.py                # classification only
│   └── streamlit_app_enhanced.py       # 4-page app with assignment
├── testing_data_extraction.py          # Ivanti → Excel (Selenium)
├── testing_data_classifier.py          # Excel → classified Excel
├── testing_feedback_loop.py            # validated Excel → corpus → rebuild
├── report_automation.py                # separate reporting automation
├── data/                               # all intermediates (see §14)
└── outputs/                            # logs, results, eval runs
```

---

## 3. The complete technical flow

```
                              BUILD (offline, once)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Historic_Data_CSV.xlsx (6,497 rows, 8 cols)                              │
 │        │ TicketCleaner.clean_dataset()                                   │
 │        ▼ cleaned_tickets.csv  (+Summary_Clean, Description_Clean,        │
 │        │                        Combined_Text)                           │
 │        │ TicketAugmenter.augment_dataset()  [Groq llama-3.1-8b-instant]  │
 │        ▼ augmented_tickets.csv (12,302 rows, +Augmented tag)             │
 │        │ DataSplitter.split() — stratified on Module                     │
 │        ▼ train.csv 8,611 · val.csv 1,845 · test.csv 1,846                │
 │        │                                                                 │
 │        ├─► GeminiEmbedder.embed_batch()  ──► train_embeddings.pkl        │
 │        │        │                              (3072-dim vectors)        │
 │        │        ▼ ChromaVectorStore.add_documents(batch=5000)            │
 │        │          data/vector_store/chroma.sqlite3 + HNSW segments       │
 │        │                                                                 │
 │        └─► SparseRetriever(corpus) ──► data/embeddings/bm25_index.pkl    │
 └──────────────────────────────────────────────────────────────────────────┘
                                      │
                              SERVE (per ticket)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ query = f"{Summary} [SEP] {Description}"                                 │
 │        │                                                                 │
 │        ├─► DenseRetriever  : embed_query → Chroma.query(n=2k)            │
 │        ├─► SparseRetriever : BM25.get_scores → top 2k                    │
 │        ▼                                                                 │
 │   HybridRetriever._reciprocal_rank_fusion() → top-k (k=10)               │
 │        │                                                                 │
 │        ▼ SAPModuleClassifier.predict()                                   │
 │          group by module → golden = 2 from majority module               │
 │                          → distractors = up to 2 from other modules      │
 │          PromptTemplates.format_raft_prompt(...)                         │
 │        ▼                                                                 │
 │   LLM call (Gemini or Groq, with retry + key rotation)                   │
 │        ▼ _parse_response() regex → module / confidence / reasoning /     │
 │          key_indicators; needs_review = confidence < 0.6                 │
 │        ▼                                                                 │
 │   AssignmentEngine.assign_ticket(module) → lowest-load eligible engineer │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1 — Data cleaning

**Module:** `src/data_preparation/cleaner.py` → `TicketCleaner`
**Input:** `data/raw/Historic_Data_CSV.xlsx` — 6,497 rows, columns
`Incident, Service, Module, Status, Summary, Description, Priority, Team`
**Output:** `data/processed/cleaned_tickets.csv`

`clean_text()` applies, in order:

1. Null/NaN → empty string, cast to `str`
2. Collapse runs of whitespace
3. Strip special characters, keeping basic punctuation
4. Remove embedded ticket IDs matching the `INC\d+` shape — they are identifiers, not signal,
   and leaving them in gives BM25 a high-IDF token that matches nothing useful
5. Remove email addresses (both noise and PII)
6. Remove URLs
7. Trim

`clean_dataset()` then:

- Repairs the one record with a null `Summary`
- Produces `Summary_Clean` and `Description_Clean`
- Builds `Combined_Text` = `"Summary: {s} Description: {d}"` — this single field is what gets
  embedded and what BM25 indexes. Keeping the field labels inside the text is deliberate: it
  gives the embedding model a hint about which half is the terse title and which is the body.
- Drops records that are empty after cleaning
- Validates every `Module` value against the six known labels

All 6,497 records survived cleaning.

---

## 5. Stage 2 — LLM augmentation and rebalancing

**Module:** `src/data_preparation/augmenter.py` → `TicketAugmenter`
**Scripts:** `scripts/01_prepare_data.py` (all modules) or
`scripts/01a_augment_single_module.py <MODULE>` (one at a time)
**Output:** `data/processed/augmented_<Module>.csv` per module, then
`data/processed/augmented_tickets.csv`

### Target counts

Set in `Config.TARGET_COUNTS`, overridable per module by env var:

| Module | Original | Target | Generated |
|---|---:|---:|---:|
| Basis | 4,302 | 4,302 | 0 (already dominant) |
| HR & Payroll | 935 | 2,000 | 1,065 |
| Procurement | 378 | 1,500 | 1,122 |
| Connections | 354 | 1,500 | 1,146 |
| FICO | 266 | 1,500 | 1,234 |
| ABAP | 262 | 1,500 | 1,238 |
| **Total** | **6,497** | **12,302** | **5,805** |

### Two generation modes

- **Paraphrase** (4,061 rows) — an existing ticket restated. Used while original examples last.
- **Synthetic** (1,744 rows) — a new plausible ticket for the module, used once paraphrase
  sources are exhausted.

Generated rows get `Incident = AUG_<Module>_<n>` and `Augmented = Paraphrase|Synthetic`;
originals keep their real incident number and `Augmented = Original`. That tag is the only
thing that makes honest evaluation possible later — see §16, issue #3.

### Why augment at all if retrieval doesn't learn a prior?

Retrieval is not prior-free in practice. With 262 ABAP tickets against 4,302 Basis tickets, an
ABAP query's top-10 neighbours are frequently mostly Basis simply because there are more Basis
vectors in every region of the space. Augmentation raises the density of minority-module
vectors so a minority query actually retrieves its own kind.

---

## 6. Stage 3 — Splitting

**Module:** `src/data_preparation/splitter.py` → `DataSplitter`
**Output:** `data/processed/train_test_split/{train,val,test}.csv`

Two-step stratified split on `Module`, seed 42:

1. train (70%) vs holdout (30%)
2. holdout → val (15%) / test (15%)

| Split | Rows | Basis | HR&P | FICO | ABAP | Conn | Proc |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 8,611 | 3,011 | 1,400 | 1,050 | 1,050 | 1,050 | 1,050 |
| val | 1,845 | — | — | — | — | — | — |
| test | 1,846 | — | — | — | — | — | — |

Stratification is verified after splitting and logged.

> **Known methodological issue, stated up front:** the split happens *after* augmentation, so
> the test set is 972 Original / 609 Paraphrase / 265 Synthetic. A paraphrase of ticket X can
> land in train while X itself lands in test. Incident IDs do not collide, so this is not exact
> duplication — it is near-duplicate leakage, which inflates the headline score. The eval suite
> handles this by reporting an Original-only slice; the proper fix is to split first and augment
> the training split only. See §16, issue #3.

---

## 7. Stage 4 — Embeddings and the two indexes

**Script:** `scripts/02_build_vector_store.py` (single key) or
`scripts/02c_build_vector_store_multikey.py` (N keys, ~N× faster)

### Embedding generation

`GeminiEmbedder.embed_batch(texts, batch_size=100, delay=2.0)`:

- Sends up to 100 texts per `genai.embed_content` call, which is dramatically cheaper in
  wall-clock than 8,611 individual calls
- Sleeps `API_DELAY_SECONDS` between batches to stay inside the RPM cap
- On a batch failure, **falls back to per-text embedding** for that batch rather than losing it
- If an individual text also fails, inserts a zero vector as a placeholder so array indices stay
  aligned with the dataframe (a silent-but-aligned failure beats a shifted corpus)
- Result is pickled to `data/embeddings/train_embeddings.pkl`

**The cache is load-bearing.** On every subsequent run the script checks for the pickle first
and skips regeneration entirely. Rebuilding the store after a feedback cycle therefore costs
seconds, not a full re-embed.

Query embeddings use `task_type="retrieval_query"` while documents use
`task_type="retrieval_document"` — the asymmetric task types the Gemini embedding API expects.

### Vector store

`ChromaVectorStore`:

- `chromadb.PersistentClient(path="data/vector_store")`
- Collection `sap_tickets` created with `metadata={"hnsw:space": "cosine"}`
- Documents inserted in **batches of 5,000** (see §15, problem #2)
- `ids` = incident numbers; `metadatas` = `Module`, `Priority`, `Team`

### BM25 index

`SparseRetriever(corpus=texts, metadata=metadatas, index_path=...)`:

- Tokenisation is `text.lower().split()` — intentionally simple; SAP identifiers like `SM37` or
  `ZFI_REPORT` must survive tokenisation intact, and a stemmer would damage them
- `BM25Okapi` over the tokenised corpus
- Pickles `{bm25, corpus, metadata}` to `data/embeddings/bm25_index.pkl`

Both indexes are built from the **same** `Combined_Text` list in the same order, which is what
makes their results comparable.

---

## 8. Stage 5 — Hybrid retrieval and RRF

**Module:** `src/retrieval/hybrid_retriever.py`

For a request of `k` documents, the hybrid retriever asks **each** backend for `2k` candidates,
then fuses:

```python
rrf_score(doc) = Σ over retrievers   1 / (k_rrf + rank_in_that_retriever)
# k_rrf = 60 (Config.RRF_K)
```

Documents are re-sorted by summed RRF score and the top `k` returned, each carrying
`rrf_score` and `hybrid_rank`.

**Why RRF and not weighted score blending:** dense returns `1 - cosine_distance` in roughly
`[0,1]`; BM25 returns an unbounded score whose scale depends on corpus statistics and query
length. Blending them requires per-query normalisation that is itself a tuning problem. RRF
only needs the ordering, and k=60 (the value from the original RRF paper) damps the difference
between rank 1 and rank 5 so a single retriever cannot dominate the fused list.

`retrieve_with_diversity()` also exists for RAFT generation: it over-fetches `3k`, guarantees at
least one document from each requested module, then fills the remaining slots by rank.

> **Important:** RRF fusion is currently degraded by an ID mismatch between the two retrievers —
> a document found by both is not recognised as the same document, so it never receives the
> summed boost RRF is supposed to give it. Full detail and fix in §16, issue #1.

---

## 9. Stage 6 — RAFT dataset generation

**Script:** `scripts/03_create_raft_dataset.py`
**Modules:** `src/raft/generator.py`, `src/raft/distractor_selector.py`
**Output:** `data/raft/raft_dataset.json` (5,886 examples), `raft_checkpoint.json`

For each training ticket:

1. Hybrid-retrieve up to 50 candidates with module diversity
2. `DistractorSelector.select_golden_docs()` — same module, excluding the ticket itself, sorted
   by similarity, take top-n
3. `DistractorSelector.select_distractors()` — **tiered** (see §15, problem #4):
   - Tier 1: different module, similarity ≥ 0.6 — maximally confusing, best training signal
   - Tier 2: different module, similarity 0.3–0.6
   - Tier 3: any different-module document
   Fill from Tier 1 first, drop to lower tiers only when short.
4. Ask the LLM (Groq, rotating keys) for a chain-of-thought justification of the *known* label
5. Emit `{question, context: [{type: golden|distractor, module, text}], reasoning, answer}`

Checkpointed to `raft_checkpoint.json` every 25 successful examples so a multi-hour run survives
interruption and resumes.

Two execution strategies exist: `ParallelRAFTGenerator` (one worker per key, `ThreadPoolExecutor`)
for fewer than 4 keys, and sequential-with-rotation for 4 or more — rotation degrades more
gracefully under rate limits than parallel workers that all hit the wall simultaneously.

**Actual output: 5,886 of 8,611 examples (68%).** The remainder were skipped, mostly where no
acceptable distractor existed even at Tier 3.

**Status of this artifact:** the RAFT dataset is *generated but not consumed*. Nothing in the
serving path reads `raft_dataset.json`; no fine-tuning stage exists. What ships is RAFT as a
**prompting pattern** — the same golden/distractor framing applied live at inference time in
`SAPModuleClassifier.predict()`. Whether generating the dataset was worth ~9 hours of compute
given nothing consumes it is a fair question; it is retained as the input a future fine-tuning
run would need.

---

## 10. Stage 7 — The RAG classification pipeline

**Module:** `src/rag/pipeline.py` → `SAPModuleClassifier`

### `predict()` step by step

**1. Build the query.** `query = f"{summary} [SEP] {description}"` — the same shape used for
indexed documents, so query and document live in comparable regions of the space.

**2. Retrieve.** `hybrid_retriever.retrieve(query, k=top_k)` (default 10).

**3. Frame as RAFT.** When `use_raft` is on and at least 3 documents came back:

- Group retrieved documents by their `Module` metadata
- **Golden** = up to 2 documents from the module with the most representatives
- **Distractors** = up to 2 documents drawn from *other* modules
- Both are passed to `format_raft_prompt()`, which labels each context document
  `HIGH (same module)` or `UNCERTAIN (different module)` and instructs the model that some
  context may be misleading

Note what this does *not* do: golden selection uses the retrieval majority, not the true label
(which is unknown at inference). So "golden" here means "the retrieval consensus", and the
prompt's job is to make the model check that consensus against the ticket text rather than
rubber-stamp it. When fewer than 3 documents are available it falls back to
`format_classification_prompt()` — plain RAG with 5 labelled precedents.

**4. Call the LLM.** `_call_llm()` dispatches by provider:

- *Gemini*: single call, `temperature` and `max_output_tokens=2048` from config
- *Groq*: `_call_groq_with_retry()` with up to 5 attempts. On `RateLimitError` it rotates to
  the next key and retries; when all keys are exhausted it sleeps 60 s, resets to key #1 and
  continues. Non-rate-limit exceptions get a 2 s backoff.

**5. Parse.** `_parse_response()` runs regexes over the response for `Module:`, `Confidence:`,
`Reasoning:` and `Key Indicators:`. Confidence text maps to a score:
`High → 0.9`, `Medium → 0.7`, `Low → 0.5`, unrecognised → 0.7. A missing module yields
`"Unknown"`, never a guess.

**6. Enrich.** Attach the top 5 precedents (id, module, 200-char excerpt, similarity, rank), the
original input, and `needs_review = confidence < 0.6` — which given the mapping means Low
confidence or a failed call.

`predict_batch()` is a simple sequential loop over `predict()`. It is not parallel by design:
concurrent requests on a free-tier key trip the rate limiter faster than the rotation logic can
absorb.

### Prompt design

`get_system_prompt()` supplies the six module descriptions, six behavioural guidelines
(including *"be honest about uncertainty"* and *"choose the PRIMARY module"* for cross-module
tickets), and the exact output format. The rigid output format is what makes regex parsing
viable — the parser is only as reliable as the format instruction that precedes it.

---

## 11. Stage 8 — Assignment engine

**Modules:** `src/assignment/employee_manager.py`, `assignment_engine.py`
**State:** `data/assignments/employees_master.json`, `daily_assignments.json`

`EmployeeManager` handles roster CRUD, per-employee daily counters, and `_check_daily_reset()`,
which compares the stored date against today on every load and zeroes the counters on a date
change.

`AssignmentEngine.assign_ticket(module, ticket_id, priority)`:

1. `get_available_employees(module)` — covers the module **and** status is not `on_leave`
   **and** daily count < cap
2. Sort ascending by current daily count
3. Take the first — lowest load wins
4. `increment_assignment()` and persist
5. Return the assignment with `daily_count`, `max_daily` and `remaining`

If nobody qualifies it returns an explicit `status: 'unassigned'` with a reason rather than
raising or silently over-allocating.

---

## 12. Stage 9 — Streamlit applications

Two apps share one backend:

**`app/streamlit_app.py`** — single-ticket and batch classification, prediction display with
reasoning and precedents.

**`app/streamlit_app_enhanced.py`** — four pages: Classify & Assign, Employee Management
(list / add / edit tabs), Assignment Dashboard, Settings.

`initialize_system()` is wrapped in `@st.cache_resource`, so the embedder, Chroma collection,
BM25 index, retrievers, classifier and assignment engine are constructed **once per server
process** and reused across reruns. Without that decorator every widget interaction would
re-open Chroma and unpickle the BM25 index.

Both apps hard-code `llm_provider="gemini"` regardless of `Config.LLM_PROVIDER`, because the
config default (`groq`) is tuned for batch jobs while the UI wants the higher-quality path.

---

## 13. Stage 10 — The feedback loop

Three scripts, run in order:

**`testing_data_extraction.py`** — Selenium drives Chrome into Ivanti, filters Incidents by
Team `SAP - ENZEN`, Status `Resolved` and a resolved-date window, scrapes the ticket fields and
writes `data/testing/resolved_tickets_<timestamp>.xlsx`.

**`testing_data_classifier.py`** — finds the newest extraction file, classifies every row
through `SAPModuleClassifier`, and writes
`outputs/testing_results/classified_tickets_<timestamp>.xlsx` with predicted module, confidence,
reasoning and an empty validation column.

**Human step** — a specialist marks each row Approved or Rejected and, for rejections, enters
the correct module and an optional comment.

**`testing_feedback_loop.py`** — reads the validated workbook, splits rows into
corrected / approved / pending, converts corrections into training rows (with `Combined_Text`
rebuilt), appends them to `train.csv`, then rebuilds the vector store and BM25 index so the
corrected precedent becomes retrievable.

Because embeddings are cached per-text, the rebuild only embeds the newly appended rows.

---

## 14. Intermediate artifacts reference

Every file produced along the way, what makes it, and what consumes it:

| Artifact | Produced by | Consumed by | Notes |
|---|---|---|---|
| `data/raw/Historic_Data_CSV.xlsx` | — (source export) | `01_prepare_data.py` | 6,497 rows |
| `data/processed/cleaned_tickets.csv` | `TicketCleaner` | augmenter, `01b_combine` | 6,497 rows, +3 derived cols |
| `data/processed/augmented_<Module>.csv` | `01a_augment_single_module.py` | `01b_combine_augmented.py` | one per module |
| `data/processed/augmented_tickets.csv` | augmenter / combine | splitter | 12,302 rows |
| `data/processed/train_test_split/train.csv` | `DataSplitter` | vector store build, feedback loop | 8,611 — **grows** with feedback |
| `.../val.csv` | `DataSplitter` | (reserved for tuning) | 1,845 |
| `.../test.csv` | `DataSplitter` | evaluation | 1,846 |
| `data/embeddings/train_embeddings.pkl` | `GeminiEmbedder` | vector store build | **cache — delete to force re-embed** |
| `data/embeddings/bm25_index.pkl` | `SparseRetriever.save_index()` | sparse retrieval | `{bm25, corpus, metadata}` |
| `data/embeddings/*_backup_<epoch>.pkl` | feedback loop | disaster recovery | pre-rebuild snapshots |
| `data/vector_store/chroma.sqlite3` + UUID dirs | ChromaDB | dense retrieval | one dir per collection version |
| `data/raft/raft_checkpoint.json` | RAFT generator | resume-on-restart | written every 25 examples |
| `data/raft/raft_dataset.json` | RAFT generator | **nothing today** | 5,886 examples; future fine-tuning input |
| `data/assignments/employees_master.json` | `EmployeeManager` | assignment engine, UI | roster |
| `data/assignments/daily_assignments.json` | `EmployeeManager` | assignment engine, dashboard | resets daily |
| `data/testing/resolved_tickets_*.xlsx` | extraction script | classifier script | raw Ivanti pull |
| `outputs/testing_results/classified_tickets_*.xlsx` | classifier script | human validator | validation column |
| `outputs/results/evaluation_results.json` | `ModelEvaluator` | reporting | metrics + first 100 predictions |
| `outputs/results/confusion_matrix.png`, `per_class_f1.png` | `ModelEvaluator` | reporting | plots |
| `outputs/logs/*.log` | loguru | debugging | rotated at 10 MB |

**Stale-artifact trap:** several of these are caches. If `train.csv` changes but
`train_embeddings.pkl` is not deleted, `02_build_vector_store.py` reuses the old embeddings and
silently indexes a corpus that no longer matches the vectors. Delete the pickle whenever the
training corpus changes outside the feedback-loop script.

---

## 15. Problems faced and fixes applied

This is the history of what actually went wrong, in the order it happened.

### Problem 1 — Gemini rate limits made bulk augmentation impossible

*Symptom:* augmenting ~5,800 tickets against the free Gemini tier stalled constantly on 429s.

*Attempt 1 — local LLM.* Ran Ollama with Mistral 7B locally to remove API limits entirely.
Correctness was fine; throughput was not — per-request latency on CPU made a 5,800-request job
untenable.

*Attempt 2 — simple delays.* Fixed sleeps between requests. Still hit limits, and the delay had
to be so conservative that the job length became the problem.

*Attempt 3 — module-at-a-time.* `01a_augment_single_module.py`, so a rate-limit stall only cost
one module's progress instead of the whole run. Better blast radius, same underlying limit.

**Fix — provider switch plus automatic key rotation.** Moved bulk work to Groq
(`llama-3.1-8b-instant`) and implemented rotation across `GROQ_API_KEY` +
`GROQ_API_KEY_BACKUP1..5`: on `RateLimitError`, advance to the next key and retry; when all keys
are exhausted, sleep 60 s, reset to key #1, continue. The job now runs unattended to completion.
The same rotation logic was later reused verbatim in the RAFT generator and the classifier.

### Problem 2 — ChromaDB rejected the insert

*Symptom:* `Batch size of 8611 is greater than max batch size of 5461`.

**Fix:** `ChromaVectorStore.add_documents()` chunks into batches of 5,000 and logs progress per
batch. 5,000 rather than 5,461 leaves headroom under a limit that is a function of the client's
SQLite variable cap and is not guaranteed stable across versions.

### Problem 3 — NaN in `Combined_Text` silently destroyed 47% of the data

*Symptom:* during RAFT generation roughly half the documents were skipped for no obvious reason.

*Root cause:* `train.csv` contained `NaN` in `Combined_Text` — string concatenation with a NaN
field yields NaN for the whole row, so those rows were unembeddable and unretrievable. Because
the skip was logged at DEBUG level in a very noisy run, it went unnoticed for a long time.

**Fix:** a one-off repair script rebuilt `Combined_Text` with `fillna('')` on both source
columns, and the guard was made permanent in `02_build_vector_store.py`:

```python
if 'Combined_Text' not in train_df.columns or train_df['Combined_Text'].isna().any():
    train_df['Combined_Text'] = ("Summary: " + train_df['Summary'].fillna('').astype(str)
                                 + " Description: " + train_df['Description'].fillna('').astype(str))
```

*Lesson recorded at the time:* a data-quality failure that manifests as a "skip" is far more
dangerous than one that raises, because the pipeline still completes and reports success.

### Problem 4 — RAFT skipped 25–35% of documents for lack of distractors

*Symptom:* a large fraction of tickets produced zero acceptable distractors and were dropped.

*Root cause:* the original rule demanded a different-module document above a 0.6 similarity
threshold. With Basis at 35% of the training corpus, retrieval for a Basis ticket returns
mostly Basis — so there frequently was no different-module candidate above threshold.

**Fix, two parts:**
1. Raised retrieval breadth from 20 to 50 candidates, giving module diversity room to appear.
2. Replaced the hard threshold with **tiered selection** (≥0.6 → 0.3–0.6 → any different
   module), filling from the best tier available.

Skip rate fell to roughly 5–10%. The explicit trade-off was *"a mediocre distractor beats no
example"*: a Tier-3 distractor is weaker training signal but is still a correct demonstration
of "ignore the irrelevant document".

### Problem 5 — a long RAFT run produced no file for six minutes

*Symptom:* 61 documents processed, zero output on disk, no way to tell whether it was working.

*Root cause:* checkpoints fired every 50 *successful* examples, and with a 30% skip rate that
took far longer than expected to reach.

**Fix:** checkpoint interval 50 → 25. First artifact now appears in ~3 minutes.

### Problem 6 — log spam hid actual progress

*Symptom:* 3–5 log lines per document × 8,611 documents ≈ 34,000 lines scrolling past.

**Fix:** a single `tqdm` progress bar showing percent, elapsed, rate, examples written and
skipped; retrieval logs demoted to DEBUG; console sink restricted to INFO+ while the file sink
keeps DEBUG. `03_create_raft_dataset.py` calls `logger.remove()` first and installs both sinks
explicitly.

### Problem 7 — the enhanced app crashed on every classification

*Symptom:* `AttributeError: 'SAPModuleClassifier' object has no attribute 'classify'`.

*Root cause:* the enhanced app was written against a method name that never existed
(`classify()` instead of `predict()`) and read `result['predicted_module']` instead of
`result['module']`.

**Fix:** corrected two call sites and four result-key reads. The deeper cause — two UI files
independently coupled to one backend with no shared adapter — is unresolved; see §16, issue #5.

### Problem 8 — embedding model name rejected by the SDK

*Symptom:* `text-embedding-004` failed against the installed `google.generativeai` client.

**Fix:** pinned `models/gemini-embedding-001`, with the reason recorded in `config.py`: the
deprecated `google.generativeai` library needs the `models/`-prefixed name; the newer
`google.genai` library uses the bare `text-embedding-004`. This is the kind of detail that costs
an hour to rediscover, so it stays in the code as a comment.

### Problem 9 — sequential RAFT generation was a 9–10 hour job

**Fix:** `ParallelRAFTGenerator` — one `RAFTGenerator` per API key, driven by a
`ThreadPoolExecutor`, giving roughly N× throughput for N keys. With 4+ keys, sequential rotation
was kept instead: parallel workers all hit the rate limit at the same moment, whereas rotation
staggers naturally.

### Problem 10 — vector store rebuild was a 2-hour block

**Fix:** `02c_build_vector_store_multikey.py` with `MultiKeyGeminiEmbedder`, rotating N Gemini
keys and scaling the inter-batch delay by key count (`base_delay / n_keys`). Three keys took the
rebuild from roughly 2 hours to roughly 40 minutes.

---

## 16. Open technical issues

These are real defects and gaps found in the current code. None are fixed; all are worth fixing.

### Issue 1 — RRF fusion never actually fuses (highest impact)

`02_build_vector_store.py` builds BM25 metadata from three columns only:

```python
metadatas = train_df[['Module', 'Priority', 'Team']].fillna('Unknown').to_dict('records')
ids       = train_df['Incident'].astype(str).tolist()     # ← ids go to Chroma only
```

`SparseRetriever.retrieve()` then does `self.metadata[idx].get('id', f'doc_{idx}')`. Since `id`
is not in the metadata dict, **every sparse result gets a synthetic id `doc_<index>`**, while
dense results carry real incident numbers. `_reciprocal_rank_fusion()` keys on `doc['id']`, so a
document returned by *both* retrievers is counted as two different documents and never receives
the summed score that is the entire point of RRF.

*Effect:* the hybrid retriever behaves as a rank-interleaved union of the two lists rather than
true fusion. It still returns sensible results — which is why this went unnoticed — but
consensus documents are not promoted.

*Fix:* include the incident id in the BM25 metadata at build time:

```python
metadatas = train_df[['Module', 'Priority', 'Team']].fillna('Unknown').to_dict('records')
for m, inc in zip(metadatas, train_df['Incident'].astype(str)):
    m['id'] = inc
```

Then rebuild `bm25_index.pkl`. Measure retrieval metrics before and after with
`scripts/05_run_evals.py --suite retrieval` — this should be a measurable improvement, and it is
the first thing to try when improving Connections recall.

### Issue 2 — the RAFT dataset is generated but never consumed

5,886 examples and roughly 9 hours of generation sit in `data/raft/raft_dataset.json` with no
reader. Either wire it into a fine-tuning step, or treat generation as an optional stage and say
so in the setup path. Right now it looks like a pipeline stage but functions as a dead end.

### Issue 3 — augmentation happens before the split

Described in §6. The test set is 47% LLM-generated and paraphrase/source pairs straddle the
split. *Fix:* split the **cleaned** data first, then augment only the training portion. That
change alone will lower the headline accuracy number — and make it true. Until then, quote the
Original-only slice from the eval suite.

### Issue 4 — confidence is three discrete values

`High/Medium/Low → 0.9/0.7/0.5`, self-reported by the model. ECE is computed over what is
effectively a 3-point scale, so 0.070 is less meaningful than it looks. Better options: ask for a
0–100 integer, or derive confidence from retrieval agreement (what fraction of the top-k share
the predicted module), which costs nothing extra and is grounded in evidence rather than
self-report.

### Issue 5 — two UI files duplicate backend wiring

`streamlit_app.py` and `streamlit_app_enhanced.py` each construct the full stack independently.
That duplication is exactly what caused problem #7. A shared `build_system()` factory used by
both would remove the class of bug.

### Issue 6 — `Config.validate()` demands `GOOGLE_API_KEY` unconditionally

Even a Groq-only path cannot start without a Gemini key. Since dense retrieval always needs
Gemini embeddings today this rarely bites, but it makes the failure message misleading for
someone trying a sparse-only or Groq-only run.

### Issue 7 — credentials hygiene

`.env` in this working copy holds live Gemini and Groq keys, including a stray bare key on its
own line. `.gitignore` does exclude `.env`, but the keys have been sitting in a shared folder
and were quoted in some of the older documentation. **Rotate every key before this repo is
shared**, and use `.env.example` (added alongside these docs) as the template.

### Issue 8 — `predict_batch()` is strictly sequential

100 tickets take 100 round-trips. A bounded worker pool combined with the existing rotation
logic would help, but it needs care — naive parallelism on free-tier keys makes throughput
worse, not better.

---

## 17. Step-by-step rebuild guide

Full environment setup is in `04_SETUP_GUIDE.md`. This is the pipeline sequence.

### Step 0 — Prerequisites

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then fill in your keys
```

Place the source export at `data/raw/Historic_Data_CSV.xlsx`.

### Step 1 — Clean, augment, split

```bash
python scripts/01_prepare_data.py
```

Runs all three stages in one shot. Expect this to be the long one — augmentation is thousands of
LLM calls. Verify:

```
data/processed/cleaned_tickets.csv          6,497 rows
data/processed/augmented_tickets.csv       12,302 rows
data/processed/train_test_split/train.csv   8,611 rows
```

**If augmentation keeps stalling**, run module-by-module instead, then combine:

```bash
python scripts/01a_augment_single_module.py HR_Payroll
python scripts/01a_augment_single_module.py Procurement
python scripts/01a_augment_single_module.py Connections
python scripts/01a_augment_single_module.py FICO
python scripts/01a_augment_single_module.py ABAP
python scripts/01b_combine_augmented.py
```

### Step 2 — Build the indexes

```bash
python scripts/02_build_vector_store.py
# or, with 2+ Gemini keys in .env (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...):
python scripts/02c_build_vector_store_multikey.py
```

Verify:

```
data/embeddings/train_embeddings.pkl        exists
data/embeddings/bm25_index.pkl              exists
data/vector_store/chroma.sqlite3            exists
log line: "Vector store built with 8611 documents"
```

> Re-running is safe and fast: the embedding pickle is reused. **Delete
> `data/embeddings/train_embeddings.pkl` first if `train.csv` changed**, or you will index a
> corpus that does not match its vectors.

### Step 3 — RAFT dataset (optional)

```bash
python scripts/03_create_raft_dataset.py
```

Hours, not minutes. Resumable from `raft_checkpoint.json`. Nothing downstream consumes the
output today (§16, issue #2) — skip it unless you intend to fine-tune.

### Step 4 — Run the app

```bash
streamlit run app/streamlit_app_enhanced.py     # full app
streamlit run app/streamlit_app.py              # classification only
```

Or use `start.bat` / `start.sh`, which check the environment and offer a menu.

### Step 5 — Evaluate

```bash
python scripts/05_run_evals.py --suite retrieval --strategy rrf --sample 300   # no LLM cost
python scripts/05_run_evals.py --suite classification --sample 200
python scripts/05_run_evals.py --suite all --sample 200 --gate
```

See `05_EVALUATION.md` for what each suite measures and how to read the output.

The legacy single-purpose evaluator is still available:

```bash
python scripts/04_evaluate_model.py
```

### Step 6 — Feedback cycle (optional)

```bash
python testing_data_extraction.py       # Ivanti → data/testing/*.xlsx  (needs Chrome)
python testing_data_classifier.py       # → outputs/testing_results/*.xlsx
#   ... a human fills in the validation column ...
python testing_feedback_loop.py         # corrections → train.csv → rebuild indexes
```

### Troubleshooting quick table

| Symptom | Cause | Action |
|---|---|---|
| `GOOGLE_API_KEY not set` | `.env` missing or not loaded | Create from `.env.example`; run from repo root |
| `BM25 index not found` | Step 2 not run | `python scripts/02_build_vector_store.py` |
| `Batch size ... greater than max` | Chroma insert not batched | Already fixed; check `batch_size=5000` in `chroma_store.py` |
| Everything predicts `Unknown` | LLM output format drifted | Inspect `raw_response`; tighten the format block in `prompt_templates.py` |
| Rate-limit stall on a bulk job | Too few keys | Add `GROQ_API_KEY_BACKUP1..5` to `.env` |
| Retrieval returns nothing sensible | Stale embedding cache | Delete `train_embeddings.pkl`, rebuild |
| App shows old data after feedback | Streamlit resource cache | Restart the Streamlit process |
