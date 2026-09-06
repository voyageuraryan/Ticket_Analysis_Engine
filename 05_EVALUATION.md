# SAP Ticket Triage Assistant — RAG Evaluation Suite

**Document 5 of 5 — Evaluation**
**Harness:** `scripts/05_run_evals.py`
**Audience:** Anyone changing retrieval, prompts, models or the corpus

A RAG system has more than one thing that can be wrong, and end-to-end accuracy hides which one
it is. If accuracy drops, was it retrieval failing to surface the right precedent, the prompt
failing to frame it, or the model failing to reason over it? This suite separates those layers
so a regression points at its own cause.

---

## Table of Contents

1. [What gets evaluated and why](#1-what-gets-evaluated-and-why)
2. [Running the suite](#2-running-the-suite)
3. [Suite 1 — Retrieval](#3-suite-1--retrieval)
4. [The data-slice problem](#4-the-data-slice-problem)
5. [Suite 2 — Classification](#5-suite-2--classification)
6. [Suite 3 — Robustness](#6-suite-3--robustness)
7. [Suite 4 — LLM-as-judge](#7-suite-4--llm-as-judge)
8. [Regression gates](#8-regression-gates)
9. [Current baseline](#9-current-baseline)
10. [Interpreting results — a decision tree](#10-interpreting-results--a-decision-tree)
11. [Cost and runtime](#11-cost-and-runtime)
12. [Extending the suite](#12-extending-the-suite)
13. [What this suite does not measure](#13-what-this-suite-does-not-measure)
14. [Verification status of the harness](#14-verification-status-of-the-harness)

---

## 1. What gets evaluated and why

A RAG pipeline has three failure surfaces stacked on top of each other:

```
   ┌──────────────────────────────────────────────────────────────┐
   │  Layer 3 — REASONING     did the model use the context well? │  suites: judge, robustness
   ├──────────────────────────────────────────────────────────────┤
   │  Layer 2 — FRAMING       was the prompt built correctly?     │  suites: robustness
   ├──────────────────────────────────────────────────────────────┤
   │  Layer 1 — RETRIEVAL     was the right precedent found?      │  suite: retrieval
   └──────────────────────────────────────────────────────────────┘
                              ↓
                  Layer 0 — the answer            suite: classification
```

A ceiling at Layer 1 caps everything above it: if a same-module precedent is never retrieved,
no prompt and no model can recover. That is why the retrieval suite runs first and runs cheapest.

| Suite | Question it answers | LLM calls | Typical n |
|---|---|---:|---:|
| `retrieval` | Is the right precedent being found? | 0 (embeddings only; `sparse_only` uses none) | 200–500 |
| `classification` | Is the final answer right, calibrated, parseable? | 1 per ticket | 150–300 |
| `robustness` | Does the answer survive perturbation and poisoned context? | 7 per ticket | 20–30 |
| `judge` | Is the reasoning grounded, or plausible-sounding filler? | 2 per ticket | 20–25 |

---

## 2. Running the suite

```bash
# Layer 1 first — cheap, deterministic, no LLM cost
python scripts/05_run_evals.py --suite retrieval --sample 300

# Is hybrid actually beating each retriever alone?
python scripts/05_run_evals.py --suite retrieval --compare-strategies --sample 200

# End-to-end, full test set
python scripts/05_run_evals.py --suite classification --sample 200

# End-to-end, genuine tickets only — the honest number (see section 4)
python scripts/05_run_evals.py --suite classification --slice original --sample 150

# Robustness probes (7 LLM calls per ticket — keep n small)
python scripts/05_run_evals.py --suite robustness --sample 25

# Reasoning quality
python scripts/05_run_evals.py --suite judge --sample 20

# Everything, with pass/fail gates — exits 1 on failure
python scripts/05_run_evals.py --suite all --sample 200 --gate
```

### Options

| Flag | Default | Purpose |
|---|---|---|
| `--suite` | `retrieval` | `retrieval` · `classification` · `robustness` · `judge` · `all` |
| `--sample` | 200 | Tickets to evaluate. `robustness`/`judge` cap themselves under `all` |
| `--slice` | `all` | `all` · `original` · `paraphrase` · `synthetic` — see §4 |
| `--strategy` | `rrf` | `rrf` · `dense_only` · `sparse_only` |
| `--compare-strategies` | off | Run retrieval across all three strategies |
| `--provider` | `gemini` | `gemini` (quality) or `groq` (fast, rotating keys) |
| `--top-k` | `Config.TOP_K_RETRIEVAL` (10) | Documents retrieved per query |
| `--seed` | 42 | Sampling seed — **keep fixed** so runs are comparable |
| `--gate` | off | Apply thresholds, exit 1 on failure |
| `--out` | `outputs/evals` | Output root |

### Output

Each run writes `outputs/evals/<timestamp>/`:

- `results.json` — everything, including per-ticket records for the classification suite
- `summary.txt` — the readable digest that is also printed to the console
- `eval.log` — DEBUG-level trace

The `meta` block in `results.json` records sample size, seed, strategy, provider, model names and
`RRF_K`, so a run is reproducible from its own output.

---

## 3. Suite 1 — Retrieval

**Relevance definition:** a retrieved document is relevant when its `Module` metadata equals the
query ticket's true module. This works because the goal is module classification, not
document-level ranking, and it needs no manual relevance judgements.

### Metrics

| Metric | Definition | Reads as |
|---|---|---|
| `hit_rate_at_k` | Share of queries with ≥1 same-module doc in the top k | **The ceiling.** Below this, the LLM cannot win |
| `precision_at_k` | Mean fraction of the top k that are same-module | Context purity. Low ⇒ mostly distractors |
| `mrr` | Mean of 1/(rank of first same-module doc) | How early the right precedent lands |
| `ndcg_at_k` | Binary-gain nDCG | Rank-weighted overall quality |
| `majority_vote_accuracy` | Accuracy of "just take the modal module of the top k" | **The baseline the LLM must beat** |
| `empty_result_count` | Queries that returned nothing | Should be 0 |
| `per_module` | hit rate / precision / MRR per true module | Where retrieval starves |

### `majority_vote_accuracy` is the most important number here

It is what you would get by deleting the LLM entirely and taking a vote among the retrieved
neighbours. If end-to-end classification accuracy is not clearly above it, the LLM is adding
latency and cost without adding judgement — and the RAFT framing is not doing its job.

### Strategy comparison

`--compare-strategies` runs `rrf`, `dense_only` and `sparse_only` over the same sample. This is
the ablation that justifies the hybrid design. Expect roughly:

- `dense_only` — strong MRR on paraphrased/symptom-described tickets
- `sparse_only` — strong on tickets quoting transaction codes or error strings; **zero API cost**
- `rrf` — should beat both on `hit_rate`, because the union covers both query styles

**If `rrf` does not beat both, investigate before tuning anything else.** There is a known defect
in the fusion — dense results are keyed by incident id while sparse results are keyed by a
synthetic `doc_<index>`, so a document found by both retrievers is never recognised as the same
document and never receives the summed RRF boost. Details and the one-line fix are in
`02_TECHNICAL_IMPLEMENTATION.md` §16, issue 1. **Run this suite before and after that fix** —
it is the cleanest available measurement of its value.

---

## 4. The data-slice problem

**This section matters more than any individual metric.**

The corpus was rebalanced with LLM-generated rows *before* the train/test split. The held-out
test set is therefore:

| Slice | Rows | Share | What it is |
|---|---:|---:|---|
| `Original` | 972 | 52.7% | Genuine historic tickets |
| `Paraphrase` | 609 | 33.0% | An LLM restating a real ticket |
| `Synthetic` | 265 | 14.4% | An LLM inventing a plausible ticket |

Two distinct problems follow:

1. **Generated tickets are easier.** They were produced by an LLM prompted with the module, so
   they carry cleaner, more canonical module vocabulary than a real user's message. Classifying
   them is closer to reading a label than reading a ticket.
2. **Near-duplicate leakage.** Incident IDs do not collide, so nothing is literally duplicated —
   but a paraphrase of ticket X can sit in *train* while X itself sits in *test*. Retrieval then
   surfaces a near-copy of the test ticket as a precedent. That is leakage, and it inflates the
   score.

**Consequence: the full-test-set number is optimistic and should not be quoted externally.**

The suite handles this in two ways:

- Every classification run reports a per-slice breakdown automatically
- `--slice original` restricts evaluation to genuine tickets only

```bash
python scripts/05_run_evals.py --suite classification --slice original --sample 150
```

**The `Original` slice is the honest headline.** Expect it to be lower than the full-set figure;
if the gap is large, the gap itself is the finding.

The real fix is upstream: split the cleaned data first, then augment only the training split.
That change will lower the reported numbers — and make them true. Until it lands, quote the
`Original` slice.

---

## 5. Suite 2 — Classification

End-to-end quality with the breakdown that makes a drop diagnosable.

### Metrics

| Metric | Reads as |
|---|---|
| `accuracy` | Overall correctness. Compare against 0.662 — the always-Basis baseline |
| `macro_f1` | **The primary metric.** All six modules weighted equally, so minority failure cannot hide |
| `weighted_f1` | Support-weighted; dominated by Basis |
| `ece` | Expected Calibration Error — does stated confidence match observed accuracy? |
| `parse_failure_rate` | Share predicted `Unknown` — output-format drift indicator |
| `missing_reasoning_rate` | Answers with no reasoning text. Should be ~0 |
| `needs_review_rate` | Share flagged for human review |
| `review_precision` | Of the flagged, what share were actually wrong — is the flag informative? |
| `context_hit_rate` | Share where a same-module precedent appeared in the shown context |
| `mean_latency_s` | Per-ticket wall-clock |
| `per_module` | Precision / recall / F1 / support per module |
| `confusion_matrix` | Which module gets mistaken for which |
| `slices` | Accuracy and macro F1 per `Original` / `Paraphrase` / `Synthetic` |

### Why macro F1 is the primary metric

With 66% of the corpus in one class, accuracy and weighted F1 both reward getting Basis right and
barely penalise getting Connections wrong. Macro F1 gives Connections the same weight as Basis —
which is what the operational goal actually requires, since a misrouted Connections ticket costs
exactly as much as a misrouted Basis ticket.

### `context_hit_rate` vs `accuracy` — the diagnostic pair

Read them together:

| `context_hit_rate` | `accuracy` | Diagnosis |
|---|---|---|
| Low | Low | **Retrieval problem.** Fix Layer 1 first; the prompt is irrelevant |
| High | Low | **Reasoning problem.** The evidence was there and was not used — prompt or model |
| High | High | Working as designed |
| Low | High | Suspicious — the model may be answering from parametric knowledge, not context |

That last row is worth taking seriously: a RAG system answering correctly *without* using its
context is not a RAG system, and it will fail silently on anything outside the model's training
data. The robustness suite's poisoned-context probe tests this directly.

### On calibration

`ece` is computed over 10 bins, but confidence takes only three values here
(`High → 0.9`, `Medium → 0.7`, `Low → 0.5`). The measurement is honest but coarse — it can only
tell you whether those three buckets are roughly right, not whether a continuous confidence would
be well calibrated. Improving this means changing the pipeline, not the metric: ask the model for
a 0–100 score, or derive confidence from retrieval agreement.

---

## 6. Suite 3 — Robustness

Accuracy on a static test set says nothing about whether an answer is stable. Two probes:

### Probe A — prediction stability

Each ticket is re-classified under five meaning-preserving perturbations:

| Perturbation | Simulates |
|---|---|
| `summary_only` | A ticket with a terse title and no body |
| `description_only` | A ticket logged with no title |
| `lowercased` | Case-inconsistent entry |
| `noise_prefix` | Service-desk boilerplate prepended |
| `whitespace_mangled` | Copy-paste formatting damage |

`prediction_stability` = share of perturbed variants that keep the baseline answer. All five
preserve meaning, so a stable system should keep its answer. `per_perturbation` shows which one
breaks it — and `summary_only` / `description_only` breaking hardest tells you how much the
system leans on each field, which is genuinely useful to know.

### Probe B — distractor resistance

**This is the direct test of the RAFT premise.** The retriever is swapped for a
`PoisonedRetriever` that returns *only* documents from a deliberately wrong module, then the
ticket is classified again.

```
normal:   retrieve → mixed context → classify
poisoned: retrieve → 10/10 documents from the WRONG module → classify
          ↓
          did the model still answer correctly?
```

`distractor_resistance` = share of poisoned runs that still answer correctly.

- **High** — the model is reasoning from the ticket's own content, exactly as the RAFT prompt
  instructs. The framing is earning its tokens.
- **Low** — the model is parroting retrieval consensus. The RAFT prompt is decorative, and any
  retrieval error becomes a classification error.

No other metric in this suite tests this. Accuracy on a normal test set cannot distinguish
"reasoned correctly" from "retrieval happened to agree".

### Probe C — degenerate input

Empty strings, `"?"`, and `"test"` are classified. The system must return a valid module or
`Unknown` without crashing. Reported as `degenerate_input_handled: n/3`.

---

## 7. Suite 4 — LLM-as-judge

Accuracy metrics are blind to explanation quality — a right answer with fabricated reasoning is
still a trust failure, and this system's entire value proposition is that a triager can read the
justification and decide whether to believe it.

A second LLM scores each output 1–5 on:

| Dimension | Question |
|---|---|
| `groundedness` | Is every claim supported by text actually in the ticket? Penalises invented details |
| `relevance` | Does the reasoning explain *this module choice*, or restate the ticket generically? |
| `specificity` | Are the key indicators concrete terms from the ticket, or vague category words? |

Also reported: `low_groundedness_rate` (share scoring ≤2), `judge_failures` (unparseable judge
responses), and up to 25 short notes with the incident number for manual inspection.

**Treat judge scores as soft signals.** The judge shares failure modes with the model it judges,
and a self-judging setup is biased. A *drop* between runs is a prompt to look at the notes;
an absolute score is not proof of anything.

---

## 8. Regression gates

`--gate` applies thresholds and exits 1 on failure, making the suite usable as a check before
merging a prompt or retrieval change.

| Gate | Threshold | Direction |
|---|---:|---|
| `retrieval.hit_rate_at_10` | 0.90 | ≥ |
| `retrieval.mrr` | 0.70 | ≥ |
| `classification.macro_f1` | 0.75 | ≥ |
| `classification.accuracy` | 0.78 | ≥ |
| `classification.parse_failure_rate` | 0.01 | ≤ |
| `classification.ece` | 0.10 | ≤ |
| `robustness.prediction_stability` | 0.85 | ≥ |
| `robustness.distractor_resistance` | 0.60 | ≥ |

Thresholds live in the `GATES` dict at the top of `scripts/05_run_evals.py`. Two rules:

1. **Only gates for suites that ran are checked.** A retrieval-only run checks retrieval gates.
2. **Do not raise a threshold to make a red run green.** The gate exists to catch exactly that
   run. If a threshold is genuinely wrong, change it in its own commit with the reasoning.

Sample size matters: at n=200 a single flipped prediction moves accuracy by 0.005. Keep `--seed`
fixed and `--sample` constant across comparisons, or you are measuring sampling noise.

---

## 9. Current baseline

From `outputs/results/evaluation_results.json` — 200 sampled test tickets, RAFT prompting,
hybrid retrieval, `top_k=10`.

### Overall

| Metric | Value | Gate | Status |
|---|---:|---:|---|
| Accuracy | 0.830 | ≥ 0.78 | ✅ |
| Macro F1 | 0.805 | ≥ 0.75 | ✅ |
| Weighted F1 | 0.829 | — | — |
| ECE | 0.070 | ≤ 0.10 | ✅ |

Against the always-Basis baseline of 0.662, the system adds **+16.8 points of accuracy** — and
far more in macro F1, which is the number that reflects the operational goal.

### Per module

| Module | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Basis | 0.952 | 0.857 | 0.902 | 70 |
| Procurement | 0.783 | 1.000 | 0.878 | 18 |
| FICO | 0.844 | 0.871 | 0.857 | 31 |
| HR & Payroll | 0.732 | 0.882 | 0.800 | 34 |
| ABAP | 0.792 | 0.704 | 0.745 | 27 |
| **Connections** | 0.706 | 0.600 | **0.649** | 20 |

### Confusion matrix

Rows = true, columns = predicted, order `Basis, HR & Payroll, Procurement, Connections, FICO, ABAP`:

```
                 Basis  HR&P  Proc  Conn  FICO  ABAP
Basis              60     8     0     1     0     1
HR & Payroll        1    30     1     2     0     0
Procurement         0     0    18     0     0     0
Connections         0     1     0    12     3     4     ← the problem row
FICO                1     1     2     0    27     0
ABAP                1     1     2     2     2    19
```

### What the matrix says

1. **Connections is the weakest row: 8 of 20 missed**, 3 to FICO and 4 to ABAP. Connections
   tickets describe integrations *between* systems, so they legitimately carry the vocabulary of
   whatever they connect to. This is a genuine boundary ambiguity, not sloppiness.
2. **Basis leaks into HR & Payroll** — 8 of 70 Basis tickets go to HR & Payroll, the single
   largest off-diagonal cell. Likely candidates: user-account and authorisation tickets raised
   by HR users. Worth reading those 8.
3. **Procurement recall is 1.000** at precision 0.783 — nothing is missed, but four other
   modules' tickets are pulled in. The model over-claims Procurement.
4. **ABAP recall 0.704** is the second gap: dumps and program errors get described in
   application-level language and land in the owning functional module.

### Priority order for improvement

1. Fix the RRF id defect (`02_TECHNICAL_IMPLEMENTATION.md` §16.1) and re-measure. Cheapest
   possible intervention with a plausible effect on exactly the confusable modules.
2. Add Connections precedents through the feedback loop.
3. Add explicit boundary rules to the system prompt — *"data moving between systems ⇒ prefer
   Connections even when the payload is financial"*.
4. Split-then-augment, and re-baseline honestly.

### Caveats on this baseline

- It is the whole test set, so it includes the 47% generated slice — **optimistic** (§4).
- n=200 sampled; per-module support is small (Procurement n=18), so per-module figures carry
  wide error bars.
- It predates the slice, robustness and judge suites. Re-run `--suite all` to establish a
  complete baseline, and record the `Original`-slice figure as the headline going forward.

---

## 10. Interpreting results — a decision tree

```
Classification accuracy dropped.
│
├─ Run: --suite retrieval --sample 300
│   │
│   ├─ hit_rate_at_10 dropped too
│   │   └─► RETRIEVAL problem. Check in this order:
│   │       · was train.csv changed without deleting train_embeddings.pkl? (stale cache)
│   │       · does the Chroma collection still hold ~8,611 documents?
│   │       · did EMBEDDING_MODEL change? old and new vectors are not comparable
│   │       · run --compare-strategies: is one retriever dead?
│   │
│   └─ hit_rate_at_10 unchanged
│       └─► Retrieval is fine. Continue below.
│
├─ Check parse_failure_rate
│   ├─ elevated ─► OUTPUT-FORMAT drift. The model stopped emitting "Module:".
│   │              Inspect raw_response; tighten prompt_templates.py.
│   │              Common after a model version change.
│   └─ normal ─► continue
│
├─ Compare context_hit_rate with accuracy
│   ├─ context high, accuracy low ─► REASONING problem.
│   │      Run --suite judge: is groundedness down?
│   │      Run --suite robustness: is distractor_resistance down?
│   │      Likely cause: prompt edit, temperature change, or model swap.
│   └─ both low ─► back to retrieval.
│
└─ Check the per-slice breakdown
    ├─ only the Original slice dropped ─► real degradation on genuine tickets. Take seriously.
    └─ only generated slices dropped ─► lower priority; those tickets are not real traffic.
```

---

## 11. Cost and runtime

Per ticket: 1 embedding call + 1 LLM call for classification; robustness costs 7 LLM calls.

| Command | LLM calls | Rough runtime |
|---|---:|---|
| `--suite retrieval --sample 300` | 0 (300 embeddings) | 2–5 min |
| `--suite retrieval --strategy sparse_only --sample 300` | 0 (no API at all) | seconds |
| `--suite classification --sample 200` | 200 | 8–15 min |
| `--suite robustness --sample 25` | ~175 | 10–20 min |
| `--suite judge --sample 20` | ~40 | 3–5 min |
| `--suite all --sample 200` | ~420 | 25–40 min |

Cost-control notes:

- `--strategy sparse_only` makes the retrieval suite completely free — use it for a quick
  smoke check after any corpus change.
- `--provider groq` is much faster for large classification runs and rotates keys automatically.
  Use Gemini when you are measuring the quality path the UI actually uses; do not mix providers
  within a comparison.
- Rate limits during a run degrade gracefully: rows that exhaust all retries are recorded as
  `Unknown` and counted in `parse_failure_rate`. **A spike in `parse_failure_rate` may be rate
  limiting rather than format drift** — check `eval.log` before concluding.

---

## 12. Extending the suite

### Add a metric

Return it from the suite function. Any numeric top-level key is automatically eligible for
gating as `<suite>.<key>` and appears in the summary — no registration needed.

### Add a perturbation

Add the name to `PERTURBATIONS` and a branch in `perturb()`. It is scored automatically and
appears in `per_perturbation`.

### Add a golden set

The highest-value addition. Hand-pick 30–50 genuine tickets that cover the confusable boundaries
(Connections vs FICO, ABAP vs owning module, Basis vs HR account issues), have a specialist
confirm each label, and store them as a CSV with the same columns as `test.csv`. Then point
`load_test_set()` at it. Advantages over sampling from `test.csv`: no generated rows, no
leakage, stable across runs, and deliberately weighted toward the cases that actually fail.

### Add a per-module gate

`GATES` currently holds only aggregates, which is exactly how Connections at 0.649 stays
invisible behind a macro F1 of 0.805. A `classification.min_module_f1` gate would surface it.
Compute `min(f1 for each module)` in the classification suite and add the gate.

---

## 13. What this suite does not measure

Stated plainly so nobody assumes coverage that is not there:

- **Assignment logic.** Load balancing, daily caps, leave handling and the midnight reset have
  no automated tests. `tests/` is empty. These are pure, deterministic functions — the easiest
  possible unit tests, and their absence is a real gap.
- **The feedback loop end to end.** No test asserts that a correction actually changes the next
  prediction (scenario S9 in doc 3 is a description, not a test).
- **Retrieval latency under load.** Single-user timings only.
- **Cost per ticket.** Not tracked.
- **Fairness across ticket authors or teams.** Not examined.
- **Data drift.** Nothing detects when incoming ticket vocabulary diverges from the 2026 corpus.
- **The Ivanti extraction scripts.** Selenium against a live UI, untested.

---

## 14. Verification status of the harness

Being precise about what has and has not been executed:

**Verified by direct execution** — every suite function was run against fake retrievers and fake
classifiers with asserted expectations, covering: nDCG (including the all-zero and empty cases),
retrieval metrics with known ground truth, the empty-result and exception paths, classification
metrics and slice grouping, parse-failure and review-precision accounting, all five
perturbations, `PoisonedRetriever` construction and filtering, the robustness suite including
the assertion that the real retriever is restored after poisoning, the judge suite on both
parseable and unparseable verdicts, gate evaluation in both directions (lower- and upper-bound),
and summary rendering. `load_test_set()` was verified against the real `test.csv`, including
each slice filter and the 265-row `Synthetic` slice. CLI flag wiring was verified statically.

**Not executed here** — a full live run against real ChromaDB, BM25 and LLM APIs. The
dependencies in `requirements.txt` (`loguru`, `rank_bm25`, `google-generativeai`, `groq`,
`matplotlib`, `seaborn`) are **not installed** in the environment where this was written, so the
`build_stack()` path and the real end-to-end run could not be exercised. Install the requirements
per `04_SETUP_GUIDE.md` and start with the free check:

```bash
python scripts/05_run_evals.py --suite retrieval --strategy sparse_only --sample 25
```

That path makes no API calls, so it validates the whole plumbing — config, BM25 index, retrieval,
metrics, output writing — for zero cost. If it passes, move on to the full suites.
