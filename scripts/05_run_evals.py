"""
RAG evaluation suite for the SAP Ticket Triage Assistant.

Four independent suites, runnable together or separately:

  retrieval       Is the right precedent being retrieved at all?  (no LLM calls)
  classification  End-to-end quality + calibration + data slices
  robustness      Does the answer survive perturbation and poisoned context?
  judge           LLM-as-judge on reasoning groundedness and context use

Usage:
    python scripts/05_run_evals.py --suite retrieval --sample 300
    python scripts/05_run_evals.py --suite retrieval --compare-strategies --sample 200
    python scripts/05_run_evals.py --suite classification --sample 200
    python scripts/05_run_evals.py --suite classification --slice original --sample 150
    python scripts/05_run_evals.py --suite robustness --sample 30
    python scripts/05_run_evals.py --suite all --sample 200 --gate

Results are written to outputs/evals/<run_id>/ as JSON plus a readable summary.
Exit code is 1 when --gate is set and any threshold fails, so this can be used as a
regression check.

See 05_EVALUATION.md for what each metric means and how to read the output.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.config import Config
from src.embeddings import GeminiEmbedder
from src.vector_store import ChromaVectorStore
from src.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from src.rag import SAPModuleClassifier
from src.evaluation.metrics import calculate_metrics, calculate_confidence_calibration


# ---------------------------------------------------------------------------
# Gate thresholds. A failing gate exits non-zero. Tune deliberately, not to
# make a red run go green.
# ---------------------------------------------------------------------------
GATES = {
    "retrieval.hit_rate_at_10":        0.90,
    "retrieval.mrr":                   0.70,
    "classification.macro_f1":         0.75,
    "classification.accuracy":         0.78,
    "classification.parse_failure_rate": 0.01,   # upper bound
    "classification.ece":              0.10,     # upper bound
    "robustness.prediction_stability": 0.85,
    "robustness.distractor_resistance": 0.60,
}
UPPER_BOUND_GATES = {
    "classification.parse_failure_rate",
    "classification.ece",
}


# ---------------------------------------------------------------------------
# Stack construction
# ---------------------------------------------------------------------------

def build_stack(provider: str, need_llm: bool = True):
    """Build embedder, retrievers and (optionally) the classifier."""
    Config.validate()
    paths = Config.get_paths()

    embedder = GeminiEmbedder(
        api_key=Config.GOOGLE_API_KEY,
        model_name=Config.EMBEDDING_MODEL,
    )

    vector_store = ChromaVectorStore(persist_directory=str(paths["vector_store"]))
    vector_store.create_collection(name=Config.COLLECTION_NAME, reset=False)

    doc_count = vector_store.collection.count()
    if doc_count == 0:
        raise RuntimeError(
            "Vector store is empty. Run scripts/02_build_vector_store.py first."
        )
    logger.info(f"Vector store holds {doc_count} documents")

    bm25_index_path = Config.DATA_DIR / "embeddings" / "bm25_index.pkl"
    if not bm25_index_path.exists():
        raise RuntimeError(
            f"BM25 index not found at {bm25_index_path}. "
            "Run scripts/02_build_vector_store.py first."
        )
    sparse_retriever = SparseRetriever(index_path=str(bm25_index_path))
    dense_retriever = DenseRetriever(vector_store, embedder)
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, k=Config.RRF_K)

    classifier = None
    if need_llm:
        if provider == "groq":
            classifier = SAPModuleClassifier(
                llm_provider="groq",
                api_key=Config.GROQ_API_KEY,
                groq_api_keys=Config.get_groq_backup_keys(),
                hybrid_retriever=hybrid_retriever,
                sap_modules=Config.SAP_MODULES,
                model_name=Config.GROQ_MODEL,
                temperature=Config.TEMPERATURE,
                use_raft=True,
            )
        else:
            classifier = SAPModuleClassifier(
                llm_provider="gemini",
                api_key=Config.GOOGLE_API_KEY,
                hybrid_retriever=hybrid_retriever,
                sap_modules=Config.SAP_MODULES,
                model_name=Config.LLM_MODEL,
                temperature=Config.TEMPERATURE,
                use_raft=True,
            )

    return {
        "embedder": embedder,
        "vector_store": vector_store,
        "sparse": sparse_retriever,
        "dense": dense_retriever,
        "hybrid": hybrid_retriever,
        "classifier": classifier,
    }


def load_test_set(sample: int, slice_name: str, seed: int) -> pd.DataFrame:
    """Load the held-out test set, optionally restricted to one data slice."""
    paths = Config.get_paths()
    test_path = paths["processed"] / "train_test_split" / "test.csv"
    if not test_path.exists():
        raise RuntimeError(
            f"Test set not found at {test_path}. Run scripts/01_prepare_data.py first."
        )

    df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(df)} test rows")

    if slice_name != "all":
        wanted = {"original": "Original", "paraphrase": "Paraphrase",
                  "synthetic": "Synthetic"}[slice_name]
        if "Augmented" not in df.columns:
            raise RuntimeError("Test set has no 'Augmented' column; cannot slice.")
        df = df[df["Augmented"] == wanted]
        logger.info(f"Slice '{slice_name}': {len(df)} rows")

    if sample and sample < len(df):
        df = df.sample(sample, random_state=seed)
        logger.info(f"Sampled {len(df)} rows (seed {seed})")

    return df.reset_index(drop=True)


def query_text_for(row) -> str:
    """Build the query exactly the way the serving pipeline does."""
    summary = "" if pd.isna(row.get("Summary")) else str(row.get("Summary"))
    description = "" if pd.isna(row.get("Description")) else str(row.get("Description"))
    return f"{summary} [SEP] {description}"


# ---------------------------------------------------------------------------
# Suite 1 — Retrieval
# ---------------------------------------------------------------------------

def ndcg_at_k(relevances, k: int) -> float:
    """Binary-gain nDCG. Ideal ranking puts every relevant document first."""
    rel = relevances[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def run_retrieval_suite(stack, df: pd.DataFrame, top_k: int, strategy: str) -> dict:
    """
    A retrieved document is relevant when its module equals the query's true module.

    Reported:
      hit_rate@k          at least one same-module precedent in the top k
      precision@k         mean fraction of the top k that are same-module
      mrr                 mean reciprocal rank of the first same-module precedent
      ndcg@k              rank-weighted relevance
      majority_vote_acc   accuracy if we skipped the LLM and took the modal module
                          of the top k -- the retrieval-only baseline the LLM must beat
    """
    logger.info(f"Retrieval suite | strategy={strategy} | k={top_k} | n={len(df)}")

    hits, precisions, rrs, ndcgs, majority_correct = [], [], [], [], []
    per_module = defaultdict(lambda: {"n": 0, "hits": 0, "prec": 0.0, "rr": 0.0})
    empty_results = 0
    latencies = []

    for i, row in df.iterrows():
        gold = row["Module"]
        query = query_text_for(row)

        started = time.perf_counter()
        try:
            docs = stack["hybrid"].retrieve(query, k=top_k, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 - eval must survive one bad row
            logger.error(f"Retrieval failed on row {i}: {exc}")
            docs = []
        latencies.append(time.perf_counter() - started)

        if not docs:
            empty_results += 1
            docs = []

        rel = [1 if d["metadata"].get("Module") == gold else 0 for d in docs]

        hit = 1 if any(rel) else 0
        prec = (sum(rel) / len(rel)) if rel else 0.0
        rr = 0.0
        for rank, r in enumerate(rel, start=1):
            if r:
                rr = 1.0 / rank
                break

        hits.append(hit)
        precisions.append(prec)
        rrs.append(rr)
        ndcgs.append(ndcg_at_k(rel, top_k))

        modules = [d["metadata"].get("Module") for d in docs]
        modal = Counter(modules).most_common(1)[0][0] if modules else None
        majority_correct.append(1 if modal == gold else 0)

        pm = per_module[gold]
        pm["n"] += 1
        pm["hits"] += hit
        pm["prec"] += prec
        pm["rr"] += rr

        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(df)} queries")

    n = max(len(df), 1)
    result = {
        "strategy": strategy,
        "top_k": top_k,
        "n": int(len(df)),
        f"hit_rate_at_{top_k}": sum(hits) / n,
        f"precision_at_{top_k}": sum(precisions) / n,
        "mrr": sum(rrs) / n,
        f"ndcg_at_{top_k}": sum(ndcgs) / n,
        "majority_vote_accuracy": sum(majority_correct) / n,
        "empty_result_count": int(empty_results),
        "mean_latency_s": sum(latencies) / n,
        "per_module": {
            m: {
                "n": v["n"],
                "hit_rate": v["hits"] / v["n"] if v["n"] else 0.0,
                "precision": v["prec"] / v["n"] if v["n"] else 0.0,
                "mrr": v["rr"] / v["n"] if v["n"] else 0.0,
            }
            for m, v in sorted(per_module.items())
        },
    }
    # Gate keys are fixed at @10 regardless of the k actually used.
    result["hit_rate_at_10"] = result[f"hit_rate_at_{top_k}"]
    return result


# ---------------------------------------------------------------------------
# Suite 2 — Classification
# ---------------------------------------------------------------------------

def run_classification_suite(stack, df: pd.DataFrame, top_k: int) -> dict:
    """
    End-to-end classification with per-slice and per-module breakdown.

    The headline number over the whole test set is optimistic: roughly half of the
    test set is LLM-generated, and a paraphrase can sit in test while its source sits
    in train. The 'Original' slice is the honest figure. See 05_EVALUATION.md section 4.
    """
    logger.info(f"Classification suite | n={len(df)}")

    records = []
    for i, row in df.iterrows():
        gold = row["Module"]
        started = time.perf_counter()
        try:
            result = stack["classifier"].predict(
                summary="" if pd.isna(row.get("Summary")) else str(row.get("Summary")),
                description="" if pd.isna(row.get("Description")) else str(row.get("Description")),
                incident_number=str(row.get("Incident")),
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Prediction failed on row {i}: {exc}")
            result = {"module": "Unknown", "confidence": 0.0, "reasoning": "",
                      "needs_review": True, "similar_tickets": []}
        elapsed = time.perf_counter() - started

        retrieved_modules = [t.get("module") for t in result.get("similar_tickets", [])]
        records.append({
            "incident": str(row.get("Incident")),
            "gold": gold,
            "pred": result.get("module", "Unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "confidence_level": result.get("confidence_level", ""),
            "needs_review": bool(result.get("needs_review", False)),
            "has_reasoning": bool(str(result.get("reasoning", "")).strip()),
            "slice": row.get("Augmented", "Unknown"),
            "latency_s": elapsed,
            "retrieved_modules": retrieved_modules,
            "gold_in_context": gold in retrieved_modules,
        })

        if (i + 1) % 25 == 0:
            done = sum(1 for r in records if r["pred"] == r["gold"])
            logger.info(f"  {i + 1}/{len(df)} | running accuracy {done / len(records):.3f}")

    y_true = [r["gold"] for r in records]
    y_pred = [r["pred"] for r in records]
    confidences = [r["confidence"] for r in records]
    correct = [t == p for t, p in zip(y_true, y_pred)]

    metrics = calculate_metrics(y_true, y_pred, Config.SAP_MODULES)
    calibration = calculate_confidence_calibration(confidences, correct)

    n = max(len(records), 1)
    parse_failures = sum(1 for r in records if r["pred"] == "Unknown")
    no_reasoning = sum(1 for r in records if not r["has_reasoning"])
    flagged = sum(1 for r in records if r["needs_review"])
    flagged_and_wrong = sum(1 for r in records if r["needs_review"] and r["pred"] != r["gold"])
    ctx_hit = sum(1 for r in records if r["gold_in_context"])

    # Slice breakdown -- the reason this suite exists in this form.
    slices = {}
    by_slice = defaultdict(list)
    for r in records:
        by_slice[r["slice"]].append(r)
    for name, rows in sorted(by_slice.items()):
        s_true = [r["gold"] for r in rows]
        s_pred = [r["pred"] for r in rows]
        s_metrics = calculate_metrics(s_true, s_pred, Config.SAP_MODULES)
        slices[name] = {
            "n": len(rows),
            "accuracy": s_metrics["overall_accuracy"],
            "macro_f1": s_metrics["macro_avg"]["f1_score"],
        }

    return {
        "n": int(len(records)),
        "accuracy": metrics["overall_accuracy"],
        "macro_f1": metrics["macro_avg"]["f1_score"],
        "weighted_f1": metrics["weighted_avg"]["f1_score"],
        "ece": calibration["expected_calibration_error"],
        "parse_failure_rate": parse_failures / n,
        "missing_reasoning_rate": no_reasoning / n,
        "needs_review_rate": flagged / n,
        "review_precision": (flagged_and_wrong / flagged) if flagged else None,
        "context_hit_rate": ctx_hit / n,
        "mean_latency_s": sum(r["latency_s"] for r in records) / n,
        "per_module": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_labels": Config.SAP_MODULES,
        "calibration": calibration,
        "slices": slices,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Suite 3 — Robustness
# ---------------------------------------------------------------------------

class PoisonedRetriever:
    """
    Stands in for the hybrid retriever and returns only documents from a module
    that is deliberately NOT the true one.

    This is a direct test of the RAFT premise: the prompt tells the model that some
    context may be misleading and that it must justify from the ticket's own content.
    If that instruction works, the model resists unanimous wrong context. If it does
    not, the model is parroting retrieval, and the RAFT framing is decorative.
    """

    def __init__(self, sparse_retriever, wrong_module: str, k: int = 10):
        self.wrong_module = wrong_module
        self.k = k
        self.docs = []
        for idx, meta in enumerate(sparse_retriever.metadata):
            if meta.get("Module") == wrong_module:
                self.docs.append({
                    "id": f"poison_{idx}",
                    "text": sparse_retriever.corpus[idx],
                    "metadata": meta,
                    "similarity": 0.75,
                    "rank": len(self.docs) + 1,
                    "retriever": "poison",
                })
                if len(self.docs) >= k:
                    break

    def retrieve(self, query, k=10, strategy="rrf"):
        return self.docs[:k]


def perturb(summary: str, description: str, kind: str):
    """Meaning-preserving perturbations a real ticket could plausibly arrive with."""
    if kind == "summary_only":
        return summary, ""
    if kind == "description_only":
        return "", description
    if kind == "lowercased":
        return summary.lower(), description.lower()
    if kind == "noise_prefix":
        return (summary,
                "Logged via service desk. Priority to be confirmed. " + description)
    if kind == "whitespace_mangled":
        return "  ".join(summary.split()), "\n\n".join(description.split(". "))
    raise ValueError(kind)


PERTURBATIONS = ["summary_only", "description_only", "lowercased",
                 "noise_prefix", "whitespace_mangled"]


def run_robustness_suite(stack, df: pd.DataFrame, top_k: int) -> dict:
    """
    Two independent probes:

      prediction_stability   fraction of perturbed variants that keep the baseline answer
      distractor_resistance  fraction of poisoned-context runs that still answer correctly

    Deliberately small by default: this is 1 + 5 + 1 LLM calls per ticket.
    """
    logger.info(f"Robustness suite | n={len(df)} tickets "
                f"({len(df) * (len(PERTURBATIONS) + 2)} LLM calls)")

    classifier = stack["classifier"]
    original_retriever = classifier.hybrid_retriever

    stability_hits, stability_total = 0, 0
    per_perturbation = defaultdict(lambda: {"same": 0, "n": 0})
    resisted, poison_total = 0, 0
    baseline_correct = 0
    empty_input_safe = 0

    for i, row in df.iterrows():
        summary = "" if pd.isna(row.get("Summary")) else str(row.get("Summary"))
        description = "" if pd.isna(row.get("Description")) else str(row.get("Description"))
        gold = row["Module"]

        base = classifier.predict(summary, description, str(row.get("Incident")), top_k=top_k)
        base_module = base.get("module", "Unknown")
        if base_module == gold:
            baseline_correct += 1

        for kind in PERTURBATIONS:
            p_summary, p_description = perturb(summary, description, kind)
            try:
                out = classifier.predict(p_summary, p_description, None, top_k=top_k)
                same = out.get("module") == base_module
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Perturbation {kind} failed: {exc}")
                same = False
            stability_total += 1
            stability_hits += int(same)
            per_perturbation[kind]["n"] += 1
            per_perturbation[kind]["same"] += int(same)

        # Poisoned context: force every retrieved document to the wrong module.
        wrong_module = next(m for m in Config.SAP_MODULES if m != gold)
        try:
            classifier.hybrid_retriever = PoisonedRetriever(
                stack["sparse"], wrong_module, k=top_k
            )
            poisoned = classifier.predict(summary, description, None, top_k=top_k)
            poison_total += 1
            resisted += int(poisoned.get("module") == gold)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Poisoned-context probe failed: {exc}")
        finally:
            classifier.hybrid_retriever = original_retriever

        logger.info(f"  ticket {i + 1}/{len(df)} done")

    # Degenerate input must not crash or fabricate.
    for summary, description in [("", ""), ("?", ""), ("test", "test")]:
        try:
            out = classifier.predict(summary, description, None, top_k=top_k)
            if out.get("module") in Config.SAP_MODULES + ["Unknown"]:
                empty_input_safe += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Degenerate input crashed: {exc}")

    return {
        "n_tickets": int(len(df)),
        "baseline_accuracy": baseline_correct / max(len(df), 1),
        "prediction_stability": stability_hits / max(stability_total, 1),
        "per_perturbation": {
            k: {"n": v["n"], "stability": v["same"] / v["n"] if v["n"] else 0.0}
            for k, v in sorted(per_perturbation.items())
        },
        "distractor_resistance": resisted / max(poison_total, 1),
        "poison_probes": int(poison_total),
        "degenerate_input_handled": f"{empty_input_safe}/3",
    }


# ---------------------------------------------------------------------------
# Suite 4 — LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are auditing an automated SAP ticket classifier. Score its output.

TICKET:
{ticket}

CLASSIFIER OUTPUT:
Module: {module}
Reasoning: {reasoning}
Key indicators: {indicators}

Score each dimension 1-5 (5 is best):

1. GROUNDEDNESS - is every claim in the reasoning supported by text actually present in
   the ticket? Penalise invented details, assumed systems, or facts not in the ticket.
2. RELEVANCE - does the reasoning explain THIS module choice, rather than restating the
   ticket or giving generic filler?
3. SPECIFICITY - are the key indicators concrete terms from the ticket, rather than vague
   category words?

Reply in exactly this format and nothing else:
GROUNDEDNESS: <1-5>
RELEVANCE: <1-5>
SPECIFICITY: <1-5>
NOTE: <one short sentence>"""


def run_judge_suite(stack, df: pd.DataFrame, top_k: int) -> dict:
    """
    Score reasoning quality with a second LLM. This measures explanation quality, which
    the accuracy metrics say nothing about: a right answer with fabricated reasoning is
    still a trust problem.

    Judge scores are soft signals. Treat a drop as a prompt for inspection, not proof.
    """
    import re

    logger.info(f"Judge suite | n={len(df)}")
    classifier = stack["classifier"]
    scores = {"groundedness": [], "relevance": [], "specificity": []}
    notes, failures = [], 0

    for i, row in df.iterrows():
        summary = "" if pd.isna(row.get("Summary")) else str(row.get("Summary"))
        description = "" if pd.isna(row.get("Description")) else str(row.get("Description"))
        result = classifier.predict(summary, description, str(row.get("Incident")), top_k=top_k)

        prompt = JUDGE_PROMPT.format(
            ticket=f"Summary: {summary}\nDescription: {description}"[:3000],
            module=result.get("module", "Unknown"),
            reasoning=str(result.get("reasoning", ""))[:2000],
            indicators=str(result.get("key_indicators", ""))[:1000],
        )
        verdict = classifier._call_llm(prompt)
        if not verdict:
            failures += 1
            continue

        parsed = {}
        for dim, key in [("GROUNDEDNESS", "groundedness"),
                         ("RELEVANCE", "relevance"),
                         ("SPECIFICITY", "specificity")]:
            match = re.search(rf"{dim}:\s*([1-5])", verdict)
            if match:
                parsed[key] = int(match.group(1))
        if len(parsed) < 3:
            failures += 1
            continue

        for key, value in parsed.items():
            scores[key].append(value)
        note = re.search(r"NOTE:\s*(.+)", verdict)
        if note:
            notes.append({"incident": str(row.get("Incident")),
                          "predicted": result.get("module"),
                          "gold": row["Module"],
                          "note": note.group(1).strip()[:200]})

        logger.info(f"  judged {i + 1}/{len(df)}")

    def mean(values):
        return sum(values) / len(values) if values else None

    return {
        "n": int(len(df)),
        "judge_failures": failures,
        "mean_groundedness": mean(scores["groundedness"]),
        "mean_relevance": mean(scores["relevance"]),
        "mean_specificity": mean(scores["specificity"]),
        "low_groundedness_rate": (
            sum(1 for s in scores["groundedness"] if s <= 2) / len(scores["groundedness"])
            if scores["groundedness"] else None
        ),
        "notes": notes[:25],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def flatten_for_gates(results: dict) -> dict:
    flat = {}
    for suite, payload in results.items():
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"{suite}.{key}"] = value
    return flat


def check_gates(results: dict) -> tuple:
    flat = flatten_for_gates(results)
    rows, passed = [], True
    for name, threshold in GATES.items():
        if name not in flat:
            continue
        value = flat[name]
        upper = name in UPPER_BOUND_GATES
        ok = value <= threshold if upper else value >= threshold
        passed = passed and ok
        rows.append({
            "gate": name,
            "value": round(value, 4),
            "threshold": threshold,
            "direction": "<=" if upper else ">=",
            "pass": ok,
        })
    return rows, passed


def render_summary(results: dict, gate_rows) -> str:
    out = ["=" * 78, "RAG EVALUATION SUMMARY", "=" * 78, ""]

    for name, payload in results.items():
        if name == "meta" or not isinstance(payload, dict):
            continue
        out.append(f"--- {name.upper()} " + "-" * (72 - len(name)))
        for key, value in payload.items():
            if key in ("records", "notes", "per_module", "calibration",
                       "confusion_matrix", "confusion_labels", "slices",
                       "per_perturbation"):
                continue
            if isinstance(value, float):
                out.append(f"  {key:<32} {value:.4f}")
            else:
                out.append(f"  {key:<32} {value}")

        if payload.get("slices"):
            out.append("  slices:")
            for slice_name, slice_data in payload["slices"].items():
                out.append(f"    {slice_name:<14} n={slice_data['n']:<5} "
                           f"acc={slice_data['accuracy']:.3f}  "
                           f"macroF1={slice_data['macro_f1']:.3f}")

        if payload.get("per_module"):
            out.append("  per module:")
            for module, m in payload["per_module"].items():
                if "f1_score" in m:
                    out.append(f"    {module:<14} P={m['precision']:.3f} "
                               f"R={m['recall']:.3f} F1={m['f1_score']:.3f} "
                               f"n={m['support']}")
                else:
                    out.append(f"    {module:<14} hit={m['hit_rate']:.3f} "
                               f"prec={m['precision']:.3f} mrr={m['mrr']:.3f} "
                               f"n={m['n']}")

        if payload.get("per_perturbation"):
            out.append("  per perturbation:")
            for kind, p in payload["per_perturbation"].items():
                out.append(f"    {kind:<20} stability={p['stability']:.3f} n={p['n']}")
        out.append("")

    if gate_rows:
        out.append("--- GATES " + "-" * 68)
        for row in gate_rows:
            mark = "PASS" if row["pass"] else "FAIL"
            out.append(f"  [{mark}] {row['gate']:<38} "
                       f"{row['value']} {row['direction']} {row['threshold']}")
        out.append("")

    out.append("=" * 78)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation suite")
    parser.add_argument("--suite", default="retrieval",
                        choices=["retrieval", "classification", "robustness", "judge", "all"])
    parser.add_argument("--sample", type=int, default=200,
                        help="tickets to evaluate (robustness/judge default to fewer)")
    parser.add_argument("--slice", default="all",
                        choices=["all", "original", "paraphrase", "synthetic"],
                        help="'original' is the honest slice: genuine tickets only")
    parser.add_argument("--strategy", default="rrf",
                        choices=["rrf", "dense_only", "sparse_only"])
    parser.add_argument("--compare-strategies", action="store_true",
                        help="run the retrieval suite over all three strategies")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq"])
    parser.add_argument("--top-k", type=int, default=Config.TOP_K_RETRIEVAL)
    parser.add_argument("--seed", type=int, default=Config.RANDOM_SEED)
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 if any threshold fails")
    parser.add_argument("--out", default="outputs/evals")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO", format="<level>{message}</level>")
    logger.add(out_dir / "eval.log", level="DEBUG")

    logger.info("=" * 78)
    logger.info(f"RAG EVALUATION | suite={args.suite} | slice={args.slice} | run={run_id}")
    logger.info("=" * 78)

    suites = ["retrieval", "classification", "robustness", "judge"] \
        if args.suite == "all" else [args.suite]
    need_llm = any(s in suites for s in ("classification", "robustness", "judge"))

    try:
        stack = build_stack(args.provider, need_llm=need_llm)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Could not build the stack: {exc}")
        return 2

    df = load_test_set(args.sample, args.slice, args.seed)
    if df.empty:
        logger.error("No test rows selected.")
        return 2

    results = {"meta": {
        "run_id": run_id,
        "suites": suites,
        "slice": args.slice,
        "strategy": args.strategy,
        "provider": args.provider,
        "top_k": args.top_k,
        "sample_requested": args.sample,
        "sample_actual": int(len(df)),
        "seed": args.seed,
        "embedding_model": Config.EMBEDDING_MODEL,
        "llm_model": Config.LLM_MODEL if args.provider == "gemini" else Config.GROQ_MODEL,
        "rrf_k": Config.RRF_K,
    }}

    if "retrieval" in suites:
        if args.compare_strategies:
            for strategy in ["rrf", "dense_only", "sparse_only"]:
                results[f"retrieval_{strategy}"] = run_retrieval_suite(
                    stack, df, args.top_k, strategy)
            results["retrieval"] = results["retrieval_rrf"]
        else:
            results["retrieval"] = run_retrieval_suite(
                stack, df, args.top_k, args.strategy)

    if "classification" in suites:
        results["classification"] = run_classification_suite(stack, df, args.top_k)

    if "robustness" in suites:
        n = min(len(df), 30 if args.suite == "all" else args.sample)
        results["robustness"] = run_robustness_suite(
            stack, df.head(n), args.top_k)

    if "judge" in suites:
        n = min(len(df), 25 if args.suite == "all" else args.sample)
        results["judge"] = run_judge_suite(stack, df.head(n), args.top_k)

    gate_rows, gates_passed = check_gates(results) if args.gate else ([], True)

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    summary = render_summary(results, gate_rows)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    logger.info(f"Results written to {out_dir}")

    if args.gate and not gates_passed:
        logger.error("One or more gates FAILED")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
