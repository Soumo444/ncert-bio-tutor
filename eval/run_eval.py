"""Retrieval + answer evaluation for the RAG chatbot.

This is a lightweight, dependency-free take on RAGAS-style evaluation:
no heavy framework, just the four numbers that actually tell you whether
your pipeline works.

Retrieval metrics (computed against `expected_sources`):
  hit@k        did the correct document appear anywhere in the top-k?  (1/0)
  MRR          mean reciprocal rank of the first correct chunk
  precision@k  share of retrieved chunks that came from an expected source
  recall       share of expected sources that were retrieved

Answer metrics (LLM-as-judge, 0..1):
  faithfulness     is every claim supported by the retrieved context?
  answer_relevance does the answer actually address the question?
  keyword recall   did the answer contain the expected facts (if provided)

Abstention:
  questions with "expect_refusal": true test whether the bot refuses to
  answer when the documents do not contain the answer (anti-hallucination).

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --k 5 --limit 10 --no-judge
    python eval/run_eval.py --retrieval-only          # instant, no LLM calls
    python eval/run_eval.py --pause 30                # stay under 8k tokens/min
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import config, llm, pipeline, prompts, retriever, store  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Question file not found: {path}\n"
            "Create eval/questions.json (see eval/questions.example.json) or run:\n"
            "  python eval/generate_questions.py --count 20"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("questions.json must be a non-empty JSON list")
    return data


def retrieval_metrics(
    chunks: list[retriever.RetrievedChunk], expected_sources: list[str]
) -> dict:
    if not expected_sources:
        return {"hit": None, "mrr": None, "precision": None, "recall": None}
    expected = {s.lower() for s in expected_sources}
    found_ranks = [
        rank
        for rank, chunk in enumerate(chunks, start=1)
        if str(chunk.metadata.get("source", "")).lower() in expected
    ]
    retrieved_sources = {
        str(c.metadata.get("source", "")).lower() for c in chunks
    }
    return {
        "hit": 1.0 if found_ranks else 0.0,
        "mrr": (1.0 / found_ranks[0]) if found_ranks else 0.0,
        "precision": len(found_ranks) / len(chunks) if chunks else 0.0,
        "recall": len(expected & retrieved_sources) / len(expected),
    }


def keyword_recall(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    lowered = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


def run(
    k: int,
    limit: int | None,
    judge: bool,
    pause: float = 0.0,
    answers: bool = True,
    questions_path: Path = QUESTIONS_PATH,
) -> dict:
    questions = load_questions(questions_path)
    if limit:
        questions = questions[:limit]

    config.ensure_dirs()
    if store.count() == 0:
        raise SystemExit("The vector store is empty. Run `python ingest.py` first.")

    rows = []
    started = time.time()

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_sources = item.get("expected_sources") or []
        print(f"[{index}/{len(questions)}] {question[:80]}")

        stage = time.time()
        chunks = retriever.retrieve(question, k=k)
        metrics = retrieval_metrics(chunks, expected_sources)
        retrieve_s = time.time() - stage

        context = prompts.build_context(chunks)
        answer_parts: list[str] = []
        provider = ""
        judge_result = None
        stage = time.time()
        if answers:
            messages = [
                {"role": "user", "content": prompts.build_user_prompt(question, chunks)}
            ]
            for event in llm.stream_chat(prompts.SYSTEM_PROMPT, messages):
                if event["type"] == "provider":
                    provider = event.get("label", "")
                elif event["type"] == "delta":
                    answer_parts.append(event["text"])
        answer = "".join(answer_parts).strip()
        answer_s = time.time() - stage

        judge_s = 0.0
        if answers and judge:
            stage = time.time()
            judge_result = pipeline.judge_answer(question, context, answer)
            judge_s = time.time() - stage

        if answers:
            print(
                f"    {len(chunks)} chunks · retrieve {retrieve_s:.1f}s · "
                f"answer {answer_s:.1f}s{(' (' + provider + ')') if provider else ''} · "
                f"judge {judge_s:.1f}s"
            )

        refused = prompts.is_refusal(answer) if answers else None
        expect_refusal = bool(item.get("expect_refusal"))
        abstention_ok = None
        if expect_refusal and answers:
            abstention_ok = refused

        rows.append(
            {
                "question": question,
                "expected_sources": expected_sources,
                "retrieved": [
                    {
                        "source": c.metadata.get("source"),
                        "page": c.metadata.get("page"),
                        "score": round(c.score, 3),
                    }
                    for c in chunks
                ],
                **metrics,
                "faithfulness": (judge_result or {}).get("faithfulness"),
                "answer_relevance": (judge_result or {}).get("answer_relevance"),
                "context_support": (judge_result or {}).get("context_support"),
                "unsupported_claims": (judge_result or {}).get("unsupported_claims", []),
                "keyword_recall": (
                    keyword_recall(answer, item.get("must_contain") or [])
                    if answers
                    else None
                ),
                "expect_refusal": expect_refusal,
                "abstention_ok": abstention_ok,
                "refused": refused,
                "answer": answer,
                "provider": provider,
            }
        )

        if pause and index < len(questions):
            # Free tiers cap tokens per minute; spacing questions out avoids
            # a mid-run 429 that would silently switch models.
            time.sleep(pause)

    def mean(values: list[float]) -> float | None:
        values = [v for v in values if v is not None]
        return round(statistics.fmean(values), 3) if values else None

    summary = {
        "questions": len(rows),
        "k": k,
        "hit_rate": mean([r["hit"] for r in rows]),
        "mrr": mean([r["mrr"] for r in rows]),
        "precision_at_k": mean([r["precision"] for r in rows]),
        "context_recall": mean([r["recall"] for r in rows]),
        "faithfulness": mean([r["faithfulness"] for r in rows]),
        "answer_relevance": mean([r["answer_relevance"] for r in rows]),
        "keyword_recall": mean([r["keyword_recall"] for r in rows]),
        "abstention_accuracy": mean(
            [float(r["abstention_ok"]) for r in rows if r["abstention_ok"] is not None]
        ),
        "elapsed_s": round(time.time() - started, 1),
        "provider": rows[0]["provider"] if rows else None,
    }
    return {"summary": summary, "rows": rows}


def write_report(result: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = REPORTS_DIR / f"report-{stamp}.json"
    md_path = REPORTS_DIR / f"report-{stamp}.md"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = result["summary"]
    lines = [
        "# RAG evaluation report",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Questions: {summary['questions']} · top-k: {summary['k']}",
        f"- Provider: {summary['provider']}",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| hit@k | {summary['hit_rate']} |",
        f"| MRR | {summary['mrr']} |",
        f"| precision@k | {summary['precision_at_k']} |",
        f"| context recall | {summary['context_recall']} |",
        f"| faithfulness | {summary['faithfulness']} |",
        f"| answer relevance | {summary['answer_relevance']} |",
        f"| keyword recall | {summary['keyword_recall']} |",
        f"| abstention accuracy | {summary['abstention_accuracy']} |",
        "",
        "## Per question",
        "",
        "| # | Question | hit | MRR | faith | relevance |",
        "|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(result["rows"], start=1):
        question = row["question"].replace("|", "\\|")[:70]
        lines.append(
            f"| {index} | {question} | {row['hit']} | {row['mrr']} | "
            f"{row['faithfulness']} | {row['answer_relevance']} |"
        )

    weak = [r for r in result["rows"] if (r["hit"] == 0.0) or (r["faithfulness"] is not None and r["faithfulness"] < 0.6)]
    if weak:
        lines += ["", "## Needs attention", ""]
        for row in weak:
            reason = "retrieval miss" if row["hit"] == 0.0 else "low faithfulness"
            lines.append(f"- **{row['question'][:90]}** — {reason}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only first N questions")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM answer scoring")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="only measure hit@k/MRR/precision/recall (no LLM calls at all)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="seconds to wait between questions (use ~30 on Groq's free 8k TPM)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_PATH,
        help="question file (default: eval/questions.json)",
    )
    args = parser.parse_args()

    result = run(
        k=args.k,
        limit=args.limit,
        judge=not args.no_judge,
        pause=args.pause,
        answers=not args.retrieval_only,
        questions_path=args.questions,
    )
    report_path = write_report(result)

    summary = result["summary"]
    print("\n" + "=" * 62)
    print("RAG EVALUATION SUMMARY")
    print("=" * 62)
    for key in (
        "questions", "k", "hit_rate", "mrr", "precision_at_k", "context_recall",
        "faithfulness", "answer_relevance", "keyword_recall", "abstention_accuracy",
    ):
        print(f"  {key:<22} {summary[key]}")
    print(f"\nReport saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
