# SAP Ticket Triage Assistant — Functional Specification

**Document 1 of 5 — Functional view**
**Status:** Demo / proof-of-concept (no production deployment)
**Audience:** Business stakeholders, service-desk leads, project sponsors

| Related document | Covers |
|---|---|
| `02_TECHNICAL_IMPLEMENTATION.md` | How it is built, step by step, including problems hit and fixes applied |
| `03_ARCHITECTURE_AND_SCENARIOS.md` | Architecture diagrams, component connections, runtime scenarios |
| `04_SETUP_GUIDE.md` | Installing and running the demo |
| `05_EVALUATION.md` | Eval suite for the RAG system and current measured results |

---

## 1. Problem Statement

### 1.1 The operational problem

A SAP support desk receives incident tickets through an ITSM tool (Ivanti). Every incoming
ticket must be routed to exactly one of six SAP functional areas before anyone can work on it:

| Module | Scope |
|---|---|
| **Basis** | System administration, infrastructure, logons, performance, transports, database |
| **HR & Payroll** | Employee master data, payroll runs, time management, personnel administration |
| **Procurement** | Purchase orders, vendors, material management, purchasing workflow |
| **Connections** | Interfaces, integrations, third-party systems, data exchange |
| **FICO** | Financial accounting and controlling, GL, AP/AR, cost centres |
| **ABAP** | Custom development, program errors, dumps, Z-programs |

Today that routing is a **manual triage decision** made by a human reading the ticket text.
That creates four concrete costs:

1. **Latency before work starts.** A ticket sits in an unassigned queue until a triager
   reads it. The SLA clock is already running.
2. **Misroutes.** Ticket text is written by end users, not by SAP specialists. The words in
   the ticket rarely name the module. A ticket that says *"I can't log in to the SAP portal"*
   is a **Basis** ticket, but nothing in that sentence says so. Misrouted tickets bounce
   between teams, and each bounce costs a full triage cycle.
3. **Knowledge concentration.** Correct routing depends on years of accumulated pattern
   recognition held by a small number of senior people. When they are on leave, quality drops.
4. **Uneven workload.** Even after a ticket reaches the right module, choosing *which
   engineer* gets it is ad hoc. Some engineers are saturated while others are idle.

### 1.2 What makes this hard for a naive solution

The obvious fix — "train a text classifier" — runs into four properties of the real data,
all confirmed by inspecting the historic export of **6,497 tickets**:

**(a) Severe class imbalance.** Two thirds of all historic tickets are Basis:

| Module | Historic tickets | Share |
|---|---:|---:|
| Basis | 4,302 | 66.2% |
| HR & Payroll | 935 | 14.4% |
| Procurement | 378 | 5.8% |
| Connections | 354 | 5.4% |
| FICO | 266 | 4.1% |
| ABAP | 262 | 4.0% |

A classifier that predicts "Basis" for everything scores 66% accuracy while being
operationally worthless. The four minority modules have only a few hundred examples each —
too few to train a conventional supervised model that generalises.

**(b) The vocabulary is not the label.** Users describe symptoms, not systems. Module
membership has to be inferred from context, which is exactly what a bag-of-words or TF-IDF
model cannot do.

**(c) Noisy, templated, semi-structured text.** Tickets contain email signatures, staff IDs,
copy-pasted templates, embedded resolution notes, `(AutoClosed)` markers and free-form prose
all in the same field.

**(d) A black-box answer is not acceptable.** A triage tool that says "Connections" with no
justification will not be trusted by the desk. The people using it need to see *why*, and they
need to know when the system is unsure so they can take over.

### 1.3 Problem statement (formal)

> Given the free-text `Summary` and `Description` of an incoming SAP support ticket,
> automatically determine the correct SAP module, produce a human-readable justification for
> that determination, signal when confidence is too low to act on, and route the ticket to an
> available engineer who covers that module and is under their daily workload cap — all while
> learning from a training corpus in which four of the six classes are severely
> under-represented.

---

## 2. Proposed Solution

### 2.1 Solution in one paragraph

Instead of training a classifier, the system **retrieves precedent and reasons over it**. When
a new ticket arrives, it finds the most similar historic tickets — using both meaning-based and
keyword-based search — and hands those precedents to a large language model together with the
new ticket. The LLM decides the module, explains its reasoning by pointing at the evidence, and
states its own confidence. A separate assignment engine then picks an available engineer. This
is a **Retrieval-Augmented Generation (RAG)** system with a **RAFT-style** prompting discipline,
described below.

### 2.2 Why this approach solves the four hard properties

| Hard property | How the solution addresses it |
|---|---|
| Class imbalance | Retrieval does not learn a class prior. A minority-module ticket retrieves minority-module precedents, so a module with 262 examples is on equal footing at query time. The corpus is additionally rebalanced by LLM augmentation before indexing (§2.4). |
| Vocabulary ≠ label | Semantic embeddings match *"can't log in to portal"* to historic Basis logon tickets even with zero shared keywords. Keyword search is kept alongside for exact identifiers — transaction codes, error numbers, `Z*` program names — that embeddings blur. |
| Noisy text | A cleaning stage strips emails, URLs and ticket IDs and normalises whitespace before anything is indexed. |
| Explainability | The LLM must answer in a fixed structure: module, confidence, step-by-step reasoning, key indicator terms, and an analysis of the precedents it was shown. The precedents themselves are displayed alongside the answer. |

### 2.3 The three solution pillars

**Pillar 1 — Hybrid retrieval.**
Two independent searches run over the same historic corpus and their results are merged:

- *Dense (semantic)* search embeds the ticket into a vector and finds nearest neighbours by
  cosine similarity. Strong on paraphrase and synonymy; weak on rare literal tokens.
- *Sparse (BM25 keyword)* search matches literal terms. Strong on exact codes and error
  strings; blind to meaning.
- The two ranked lists are merged with **Reciprocal Rank Fusion (RRF)**, which combines by
  rank position rather than raw score, so two incomparable scoring scales can be fused safely.

Neither method alone is sufficient. The union of the two is what gives coverage across both
"describe the symptom" tickets and "quote the error code" tickets.

**Pillar 2 — RAFT-style prompting (reasoning against distractors).**
RAFT — *Retrieval-Augmented Fine-Tuning* — is the practice of deliberately showing the model
**both** relevant documents (*golden*) and irrelevant-but-plausible documents (*distractors*),
and instructing it to identify and ignore the distractors while citing the golden ones.

The problem it solves here is specific and real: standard RAG has a failure mode where the
model parrots the majority module among the retrieved neighbours. Because Basis dominates the
corpus, Basis leaks into almost every retrieval result, and a parroting model drifts toward
Basis on minority tickets. By explicitly labelling some context as possibly irrelevant and
instructing the model to justify its choice from the *ticket's own content*, the reasoning is
pushed to be evidence-driven instead of vote-driven.

This project applies RAFT as a **prompting discipline at inference time**. A RAFT training
dataset was also generated — 5,886 chain-of-thought examples with golden and distractor
context — as a downstream asset for fine-tuning. Its exact status is documented honestly in
`02_TECHNICAL_IMPLEMENTATION.md` §7.

**Pillar 3 — Explainable output with an uncertainty signal.**
Every prediction returns a fixed structure:

```
Module:      Basis
Confidence:  High
Reasoning:   - The ticket describes an inability to authenticate to the SAP Portal
             - Portal logon and account provisioning are administered in Basis
             - Precedents 1 and 3 are Basis logon failures with the same symptom
Key Indicators:
             - "cannot login": authentication, a Basis responsibility
             - "SAP Portal": Basis-administered component
Similar Tickets: 5 precedents shown with their module and similarity score
```

Predictions below the confidence floor are flagged `needs_review`, which is the system saying
*"a human should look at this one"* rather than guessing silently.

### 2.4 Supporting capability — corpus rebalancing

Before indexing, the historic corpus is expanded from **6,497 → 12,302 tickets** using an LLM
to generate two kinds of additional examples for the under-represented modules:

- **Paraphrases** (4,061) — the same incident restated in different words, which teaches the
  retriever that one problem has many phrasings.
- **Synthetic tickets** (1,744) — new plausible incidents for modules starved of examples.

Target counts per module were set so no module is left with only a few hundred precedents.
Every generated row is tagged `Paraphrase` or `Synthetic` in an `Augmented` column so it can
always be separated from genuine tickets. **This tagging matters for evaluation** — see §5.3.

### 2.5 Supporting capability — workload-aware assignment

Once the module is known, an assignment engine picks the engineer. It maintains a roster in
which each engineer has the modules they cover, a maximum number of tickets per day, and a
status (`available`, `on_leave`, `busy`). To assign a ticket it filters the roster to engineers
who cover the module, are not on leave and are below their daily cap, then picks the one with
the **lowest count so far today**. Counters reset automatically at midnight.

### 2.6 Supporting capability — human-in-the-loop feedback

The system improves from corrections rather than from retraining. Resolved tickets are
extracted from Ivanti, classified in bulk, and written to an Excel sheet with a validation
column. A human marks each prediction Approved or Rejected and, when rejecting, writes the
correct module. The feedback loop then appends the corrected tickets to the training corpus and
rebuilds the index — so the same mistake now has a correct precedent sitting in the store, and
the next similar ticket retrieves it.

### 2.7 Explicitly out of scope

- **No deployment.** This is a local demo. There is no hosting, container, CI/CD, HA or
  multi-user server component.
- **No write-back to Ivanti.** The system reads tickets and produces recommendations; it never
  updates the source ITSM record.
- **No fine-tuned model.** The RAFT dataset exists as an asset; no fine-tuning run was performed.
- **No authentication, RBAC or audit trail** on the demo UI.

---

## 3. Users and What They Do

| User | Goal | Where |
|---|---|---|
| **Triage analyst** | Paste or upload a ticket, get module + reasoning + suggested assignee | Classify page |
| **Service-desk lead** | Maintain the engineer roster, caps and leave status | Employee Management page |
| **Service-desk lead** | See today's load distribution and remaining capacity | Assignment Dashboard |
| **SAP specialist (validator)** | Review a batch of predictions in Excel and correct the wrong ones | Excel + feedback script |
| **Developer / evaluator** | Measure retrieval and classification quality, catch regressions | Eval suite (`05_EVALUATION.md`) |

---

## 4. Functional Requirements

### FR-1 — Single ticket classification
Given a `Summary` and `Description`, return one of the six modules, a confidence level
(High / Medium / Low), a step-by-step reasoning narrative, the key indicator terms that drove
the decision, and the top 5 precedent tickets with their module and similarity score.

**Acceptance:** an answer is returned for any non-empty input; when the model cannot produce a
parseable answer, the system returns `Unknown` with confidence 0 rather than a fabricated module.

### FR-2 — Batch classification
Accept a CSV/Excel of tickets, classify each, and return a downloadable file with predicted
module, confidence and reasoning appended as new columns.

### FR-3 — Uncertainty flagging
Any prediction whose confidence falls below the review threshold is marked `needs_review` and
visually distinguished in the UI, so a human takes over instead of the system acting silently.

### FR-4 — Precedent transparency
The precedents that informed the decision are always available to the user — never hidden —
including which module each belongs to and how similar it was judged to be.

### FR-5 — Roster management
Create, update and delete engineers; assign each one or more modules; set a daily ticket cap;
set status to available / on leave / busy.

### FR-6 — Workload-balanced assignment
Assign a classified ticket to the eligible engineer with the lowest current daily count. If
nobody is eligible, return an explicit *unassigned* result with the reason — never silently
drop the ticket or exceed a cap.

### FR-7 — Daily reset
Per-engineer daily counters reset automatically on date change, with a manual reset available
to an administrator.

### FR-8 — Assignment visibility
Show, for today: total tickets assigned, engineers available, engineers at capacity, engineers
on leave, and per-module remaining capacity.

### FR-9 — Validation export / import
Export a batch of predictions to Excel with an empty validation column; re-import the
human-completed sheet.

### FR-10 — Learn from corrections
Append human-corrected tickets to the training corpus and rebuild both indexes so future
retrievals surface the corrected precedent.

### FR-11 — Evaluation
Provide a repeatable eval suite that measures retrieval quality and end-to-end classification
quality, broken down per module and per data slice, with pass/fail thresholds.

---

## 5. Non-Functional Requirements and Constraints

### 5.1 Performance targets (demo scale)

| Operation | Target | Note |
|---|---|---|
| Single classification | ~2–4 s | Dominated by one embedding call and one LLM call |
| Batch of 100 | a few minutes | Sequential; rate-limit bound, not compute bound |
| Index rebuild (8.6k docs) | minutes | Embeddings are cached and reused |
| App cold start | seconds | Index loads from disk; nothing is recomputed |

### 5.2 Cost and rate-limit constraints

The demo runs entirely on **free-tier LLM APIs**, which is the single biggest constraint on the
design. Free tiers impose requests-per-minute caps that are hit almost immediately by any bulk
operation. Two provider roles were therefore separated:

- **Quality path** (user-facing classification, embeddings) — Google Gemini.
- **Throughput path** (bulk augmentation, RAFT generation, batch evaluation) — Groq, which is
  far faster per request, with **automatic API-key rotation** so hitting a rate limit on one key
  transparently continues on the next instead of aborting a multi-hour job.

### 5.3 Evaluation integrity constraint

Because the corpus was rebalanced with LLM-generated rows *before* the train/test split, the
test set contains generated as well as genuine tickets. Any headline accuracy figure computed
over the whole test set is therefore **optimistic** — generated tickets are easier than real
ones, and a paraphrase can land in test while its source ticket sits in train. The eval suite
consequently reports a **genuine-tickets-only slice** as the honest number, alongside the
full-set number. This is treated as a first-class requirement, not a footnote; the detail is in
`05_EVALUATION.md` §4.

### 5.4 Data handling

Ticket text contains employee names, staff IDs, email addresses and phone numbers. The cleaning
stage removes emails and URLs, but the corpus should still be treated as containing personal
data. For the demo everything stays on the local machine except the ticket text sent to the LLM
APIs at classification time — which is the one place real ticket content leaves the machine, and
is a point to raise before any real-world use.

---

## 6. How It Will Be Implemented (Functional View)

This section describes *what happens*, in order. The engineering detail sits in
`02_TECHNICAL_IMPLEMENTATION.md`.

### 6.1 Build phase — done once, offline

```
Historic export (6,497 tickets)
        │
        ▼
[1] CLEAN ......... strip emails, URLs, ticket IDs, normalise whitespace,
        │           drop empty records, validate module labels
        ▼
[2] REBALANCE ..... LLM paraphrases + synthetic tickets for starved modules
        │           6,497 → 12,302, each row tagged Original/Paraphrase/Synthetic
        ▼
[3] SPLIT ......... stratified 70 / 15 / 15  →  train 8,611 · val 1,845 · test 1,846
        │
        ▼
[4] INDEX ......... embed all 8,611 training tickets → vector store
        │           build BM25 keyword index over the same 8,611
        ▼
   Two searchable indexes on disk, ready to serve
```

### 6.2 Serve phase — every incoming ticket

```
   New ticket (Summary + Description)
        │
        ▼
[1] RETRIEVE ...... semantic search  ─┐
                    keyword search   ─┴─►  merge by rank (RRF)  ►  top precedents
        │
        ▼
[2] FRAME ......... split precedents into "golden" (agreeing) and
        │           "distractor" (disagreeing), and label them as such in the prompt
        ▼
[3] REASON ........ LLM reads ticket + labelled context, must justify from the
        │           ticket's own evidence, outputs module + confidence + reasoning
        ▼
[4] PARSE ......... extract structured fields; unparseable ⇒ Unknown, confidence 0
        │
        ▼
[5] FLAG .......... confidence below floor ⇒ needs_review = true
        │
        ▼
[6] ASSIGN ........ eligible engineers for that module, not on leave, under cap
        │           → lowest daily count wins → increment their counter
        ▼
   Module + reasoning + precedents + assignee (or explicit "unassigned + reason")
```

### 6.3 Improve phase — the correction cycle

```
Resolved tickets from Ivanti  ►  bulk classify  ►  Excel with a validation column
                                                          │
                                    human marks Approved / Rejected + correct module
                                                          │
                                                          ▼
                          corrected rows appended to training corpus
                                                          │
                                                          ▼
                            indexes rebuilt  ►  corrected precedent is now retrievable
```

The loop is deliberately **retrieval-based, not gradient-based**: adding a corrected precedent
to the store changes behaviour immediately, with no training run, and the change is auditable
because you can point at the exact row that caused it.

---

## 7. Current Measured Behaviour

Measured on a 200-ticket sample of the held-out test set (see `05_EVALUATION.md` for the full
method, caveats and the honest genuine-only slice):

| Metric | Value |
|---|---|
| Overall accuracy | 0.830 |
| Macro F1 (all six modules weighted equally) | 0.805 |
| Weighted F1 | 0.829 |
| Expected Calibration Error | 0.070 |

Per module:

| Module | Precision | Recall | F1 | Support | Read |
|---|---:|---:|---:|---:|---|
| Basis | 0.952 | 0.857 | 0.902 | 70 | Strongest — most precedents |
| Procurement | 0.783 | 1.000 | 0.878 | 18 | Catches all, over-claims slightly |
| FICO | 0.844 | 0.871 | 0.857 | 31 | Solid |
| HR & Payroll | 0.732 | 0.882 | 0.800 | 34 | Over-predicted; absorbs Basis misses |
| ABAP | 0.792 | 0.704 | 0.745 | 27 | Misses ~30% |
| **Connections** | 0.706 | 0.600 | **0.649** | 20 | **Weakest — the known gap** |

**The Connections problem is the headline functional finding.** Connections tickets describe
integrations between systems, so they legitimately contain vocabulary belonging to whatever
they connect *to* — financial interfaces read as FICO, custom interface programs read as ABAP.
The confusion matrix confirms it: of 20 Connections tickets, 3 went to FICO and 4 to ABAP. That
is where the next round of work should go — more Connections precedents, and sharper
module-boundary guidance in the prompt.

---

## 8. Success Criteria

### 8.1 Quantitative

| # | Criterion | Target | Now |
|---|---|---|---|
| 1 | Macro F1 across all six modules | ≥ 0.80 | 0.805 ✅ |
| 2 | No module below F1 0.70 | all ≥ 0.70 | Connections 0.649 ❌ |
| 3 | Overall accuracy | ≥ 0.80 | 0.830 ✅ |
| 4 | Calibration error | ≤ 0.10 | 0.070 ✅ |
| 5 | Macro F1 on genuine-tickets-only slice | ≥ 0.75 | to be established as the tracked baseline |
| 6 | Retrieval hit-rate @10 (a same-module precedent is retrieved) | ≥ 0.90 | measured by the eval suite |

Criterion 1 is met; criterion 2 is the open one — stated as a gap rather than averaged away,
because macro F1 alone would hide it.

### 8.2 Qualitative

- A triager reading the reasoning can tell whether to trust it without opening the precedents.
- Low-confidence cases are visibly flagged rather than silently guessed.
- A wrong prediction can be corrected, and the correction visibly changes behaviour on the next
  similar ticket.
- Nobody is assigned past their daily cap or while on leave.
- The system's own reported limits (§5.3, §7) are documented rather than hidden.

---

## 9. Known Functional Limitations

1. **Connections recall (0.60)** — the module boundary is genuinely ambiguous; needs more
   precedents and explicit disambiguation rules in the prompt.
2. **Confidence is coarse.** The model self-reports High / Medium / Low, which maps to three
   discrete scores. It is a usable triage signal, not a probability.
3. **Headline accuracy is optimistic** — see §5.3. The genuine-only slice is the number to
   quote externally.
4. **Single-label only.** Tickets that genuinely span two modules must be forced into one; the
   model is instructed to pick the primary module, and the reasoning usually says so.
5. **No fine-tuned model.** RAFT is applied as prompting; the generated RAFT dataset is unused
   downstream.
6. **Assignment ignores skill level, priority and SLA** — it balances count only.
7. **Roster state is JSON on disk**, single-user, with no locking. Fine for a demo, not for
   concurrent use.
8. **Ticket text is sent to third-party LLM APIs** at classification time.
