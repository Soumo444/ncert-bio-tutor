"""Draft an evaluation set straight from your own chunks.

For each sampled chunk the LLM writes one question whose answer is contained
in that chunk, so `expected_sources` is known automatically. Review and edit
the draft before trusting the numbers - auto-generated questions tend to be
easier than real user questions, so add a few of your own too.

Usage:
    python eval/generate_questions.py --count 20
    python eval/generate_questions.py --count 20 --out eval/questions_draft.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import config, llm, prompts, store  # noqa: E402


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate draft evaluation questions")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--out", default="eval/questions_draft.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if store.count() == 0:
        print("Vector store is empty. Run `python ingest.py` first.")
        return 1

    collection = store.get_collection()
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        print("No chunks found.")
        return 1

    random.seed(args.seed)
    indexes = random.sample(range(len(documents)), min(args.count, len(documents)))

    questions: list[dict] = []
    for position, index in enumerate(indexes, start=1):
        text = documents[index]
        metadata = metadatas[index] or {}
        source = metadata.get("source", "unknown")
        print(f"[{position}/{len(indexes)}] generating from {source} ...")

        raw, _provider = llm.complete(
            "You write concise, realistic evaluation questions for a document chatbot. Return only JSON.",
            [
                {
                    "role": "user",
                    "content": prompts.build_question_generation_prompt(text[:2500], source),
                }
            ],
        )
        parsed = parse_json(raw)
        if not parsed or not parsed.get("question"):
            print("   skipped (could not parse model output)")
            continue
        questions.append(
            {
                "question": parsed["question"],
                "expected_sources": [source],
                "expected_answer": parsed.get("answer", ""),
                "must_contain": [],
                "expect_refusal": False,
            }
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(questions)} draft questions to {out_path}")
    print("Review them, delete weak ones, and add 3-5 questions of your own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
