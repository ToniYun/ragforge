"""Score a retrieval mode against the eval set.

    python -m scripts.evaluate --mode vector
    python -m scripts.evaluate --mode hybrid --k 10
    python -m scripts.evaluate --mode rerank --json results/rerank.json

Modes light up as you build them. A mode whose code does not exist yet
exits with the step you still need to complete rather than a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

from scripts import eval_metrics as metrics
from scripts import eval_set as eval_set_module

# mode -> (module path, attribute, step description)
MODES = {
    "vector":   ("app.services.retrieval_service", "search_chunks",   "Step 0 (already built)"),
    "keyword":  ("app.services.retrieval_service", "keyword_search",  "Step 3: add keyword_search()"),
    "hybrid":   ("app.services.retrieval_service", "hybrid_search",   "Step 4: add hybrid_search()"),
    "rerank":   ("app.services.research_service",  "retrieve_and_rerank", "Step 5/6: add retrieve_and_rerank()"),
    "research": ("app.services.research_service",  "research_retrieval",  "Step 9: add research_retrieval()"),
}


class ModeUnavailable(Exception):
    pass


def resolve_mode(mode: str):
    """Import the retrieval callable for ``mode``, or explain what's missing."""
    module_path, attribute, step = MODES[mode]

    try:
        module = __import__(module_path, fromlist=[attribute])
    except ImportError as e:
        raise ModeUnavailable(f"--mode {mode} needs {module_path} ({step}). {e}")

    function = getattr(module, attribute, None)
    if function is None:
        raise ModeUnavailable(
            f"--mode {mode} needs {module_path}.{attribute}, which does not exist yet.\n"
            f"  -> {step}"
        )
    return function


def retrieved_ids(results) -> list[str]:
    """Normalise any retrieval return type to an ordered list of chunk ids.

    Every mode returns objects exposing ``.chunk``, so this works for
    SearchResult, KeywordResult and Candidate alike.
    """
    ids = []
    for result in results:
        chunk = getattr(result, "chunk", result)
        ids.append(str(chunk.id))
    return ids


def evaluate(mode: str, k: int, path: Path | None, retrieve_k: int | None) -> dict:
    questions = eval_set_module.load(path or eval_set_module.DEFAULT_PATH)

    # Resolve the mode first: "you haven't built this yet" is cheaper and more
    # actionable than "your eval set isn't labeled", and it lets you smoke-test
    # a newly written retriever before the labeling is finished.
    search = resolve_mode(mode)

    pending = eval_set_module.unlabeled(questions)
    if pending:
        print(
            f"error: {len(pending)} of {len(questions)} questions have no "
            f"relevant_chunk_ids yet.\n"
            f"  Label them first:  python -m scripts.label\n",
            file=sys.stderr,
        )
        for q in pending[:5]:
            print(f"    unlabeled: {q.query}", file=sys.stderr)
        raise SystemExit(1)

    # Import lazily so `--help` and validation errors don't pay for torch.
    from app.database import SessionLocal

    rows = []
    latencies = []
    db = SessionLocal()
    try:
        for question in questions:
            started = time.perf_counter()
            results = search(question.query, db, top_k=retrieve_k or k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies.append(elapsed_ms)

            ids = retrieved_ids(results)
            rows.append(
                {
                    "query": question.query,
                    "kind": question.kind,
                    "recall_at_k": metrics.recall_at_k(ids, question.relevant_chunk_ids, k),
                    "reciprocal_rank": metrics.reciprocal_rank(ids, question.relevant_chunk_ids, k),
                    "hit_rate": metrics.hit_rate(ids, question.relevant_chunk_ids, k),
                    "latency_ms": elapsed_ms,
                    "retrieved": ids[:k],
                }
            )
    finally:
        db.close()

    return {
        "mode": mode,
        "k": k,
        "questions": len(questions),
        "rows": rows,
        "overall": summarise(rows),
        "by_kind": {kind: summarise(group) for kind, group in group_by_kind(rows).items()},
        "latency_p50_ms": metrics.p50(latencies),
    }


def group_by_kind(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row)
    return dict(grouped)


def summarise(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "recall_at_k": metrics.mean(r["recall_at_k"] for r in rows),
        "mrr": metrics.mean(r["reciprocal_rank"] for r in rows),
        "hit_rate": metrics.mean(r["hit_rate"] for r in rows),
    }


def fmt(value: float | None) -> str:
    return "  --  " if value is None else f"{value:.3f}"


def report(result: dict) -> None:
    k = result["k"]
    print(f"\nmode={result['mode']}  k={k}  questions={result['questions']}")
    print(f"{'bucket':<14}{'n':>4}{'recall@k':>11}{'MRR':>9}{'hit':>9}")
    print("-" * 47)

    overall = result["overall"]
    print(
        f"{'OVERALL':<14}{overall['n']:>4}"
        f"{fmt(overall['recall_at_k']):>11}{fmt(overall['mrr']):>9}{fmt(overall['hit_rate']):>9}"
    )
    for kind in sorted(result["by_kind"]):
        row = result["by_kind"][kind]
        print(
            f"{kind:<14}{row['n']:>4}"
            f"{fmt(row['recall_at_k']):>11}{fmt(row['mrr']):>9}{fmt(row['hit_rate']):>9}"
        )

    print(f"\nlatency p50: {result['latency_p50_ms']:.0f} ms")
    print("\n'unanswerable' shows -- because retrieval metrics are undefined")
    print("without relevant chunks; those are scored at the generation stage.\n")

    misses = [r for r in result["rows"] if r["hit_rate"] == 0.0]
    if misses:
        print(f"complete misses ({len(misses)}):")
        for row in misses:
            print(f"  [{row['kind']}] {row['query']}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), default="vector")
    parser.add_argument("--k", type=int, default=10, help="cutoff for recall@k / MRR")
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=None,
        help="how many to ask the retriever for (defaults to --k; raise it to "
             "give fusion and reranking room to work)",
    )
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="also write raw results here")
    args = parser.parse_args(argv)

    try:
        result = evaluate(args.mode, args.k, args.eval_set, args.retrieve_k)
    except ModeUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except eval_set_module.EvalSetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    report(result)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
