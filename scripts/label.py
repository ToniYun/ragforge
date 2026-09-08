"""Interactive labeller for the eval set.

    python -m scripts.label            # label every unlabeled question
    python -m scripts.label --all      # revisit questions already labeled

For each question it runs vector search, prints the candidates with a
snippet, and asks which ones actually answer the question. Answers are
written straight back to scripts/eval_set.json.

You are the ground truth here. Judge whether the chunk *contains the
answer*, not whether it looks topically related -- that distinction is
the whole value of the eval set.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from scripts import eval_set as eval_set_module


def snippet(text: str, width: int = 300) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > width:
        collapsed = collapsed[:width] + "..."
    return textwrap.fill(collapsed, width=76, initial_indent="      ", subsequent_indent="      ")


def parse_selection(raw: str, count: int) -> list[int] | None:
    """'1,3 5' -> [0, 2, 4]. Returns None if the input is unusable."""
    tokens = raw.replace(",", " ").split()
    picked = []
    for token in tokens:
        if not token.isdigit():
            return None
        index = int(token) - 1
        if not 0 <= index < count:
            return None
        picked.append(index)
    return sorted(set(picked))


def label_question(question, db, candidate_k: int) -> bool:
    """Returns False if the user chose to quit."""
    from app.services.retrieval_service import search_chunks

    print("\n" + "=" * 78)
    print(f"[{question.kind}] {question.query}")
    if question.note:
        print(f"note: {question.note}")
    print("=" * 78)

    if question.kind == "unanswerable":
        print("(unanswerable -- no labeling needed, skipping)")
        return True

    results = search_chunks(question.query, db, top_k=candidate_k)
    if not results:
        print("no candidates returned. Is anything ingested and embedded?")
        return True

    for position, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"\n  {position:>2}. chunk={chunk.id}  doc={chunk.document.filename}  p{chunk.page_number}")
        print(snippet(chunk.content))

    print(
        "\n  Enter the numbers that ANSWER the question (e.g. '1,4'),"
        "\n  blank for none, 's' to skip, 'q' to save and quit."
    )

    while True:
        raw = input("  > ").strip().lower()

        if raw == "q":
            return False
        if raw == "s":
            return True
        if raw == "":
            question.relevant_chunk_ids = []
            print("  recorded: none relevant")
            return True

        picked = parse_selection(raw, len(results))
        if picked is None:
            print(f"  didn't understand that. Use numbers 1-{len(results)}, or s/q.")
            continue

        question.relevant_chunk_ids = [str(results[i].chunk.id) for i in picked]
        print(f"  recorded {len(picked)} relevant chunk(s)")
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=eval_set_module.DEFAULT_PATH)
    parser.add_argument("--all", action="store_true", help="revisit already-labeled questions")
    parser.add_argument("--candidates", type=int, default=15, help="how many chunks to show")
    args = parser.parse_args(argv)

    questions = eval_set_module.load(args.eval_set)
    todo = questions if args.all else eval_set_module.unlabeled(questions)

    if not todo:
        print("Everything is labeled. Run: python -m scripts.evaluate --mode vector")
        return 0

    print(f"{len(todo)} question(s) to label. Progress is saved as you go.")

    from app.database import SessionLocal

    db = SessionLocal()
    try:
        for question in todo:
            keep_going = label_question(question, db, args.candidates)
            eval_set_module.save(questions, args.eval_set)
            if not keep_going:
                print("\nsaved. Rerun to continue.")
                return 0
    finally:
        db.close()

    remaining = len(eval_set_module.unlabeled(questions))
    print(f"\nsaved to {args.eval_set}. {remaining} still unlabeled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
