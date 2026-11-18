"""The RAG pipeline: retrieve -> build context -> stream answer -> verify.

The generator `answer_stream` emits small events so the web UI can render
sources first, then stream tokens, then show a groundedness badge.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

from . import config, llm, prompts, retriever, store

_CITATION_RE = re.compile(r"[\[【](\d{1,2})[\]】]")


def _normalise_history(history: list[dict] | None, max_turns: int = 6) -> list[dict]:
    if not history:
        return []
    cleaned: list[dict] = []
    for message in history[-max_turns * 2 :]:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:4000]})
    return cleaned


def _public_sources(chunks: list[retriever.RetrievedChunk]) -> list[dict[str, Any]]:
    sources = []
    for number, chunk in enumerate(chunks, start=1):
        preview = chunk.text.strip().replace("\n", " ")
        sources.append(
            {
                "n": number,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page"),
                "section": chunk.metadata.get("section"),
                "row_start": chunk.metadata.get("row_start"),
                "row_end": chunk.metadata.get("row_end"),
                "score": round(chunk.score, 4),
                "preview": preview[:280] + ("..." if len(preview) > 280 else ""),
            }
        )
    return sources


def _chapter_source(number: int) -> str | None:
    """Find the indexed file that matches an NCERT chapter number."""
    suffix = f"lebo1{number:02d}.pdf"
    try:
        for document in store.list_documents():
            if document["source"].lower().endswith(suffix):
                return document["source"]
    except Exception:  # noqa: BLE001 - index may be empty or unavailable
        pass
    return None


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_answer(question: str, context: str, answer: str) -> dict | None:
    """LLM-as-judge groundedness check. Returns None if the judge fails."""
    if not config.GROUNDEDNESS_CHECK or not answer.strip():
        return None
    try:
        raw, _provider = llm.complete(
            prompts.JUDGE_SYSTEM,
            [
                {
                    "role": "user",
                    "content": prompts.build_judge_prompt(question, context[:12000], answer),
                }
            ],
            preferred=config.JUDGE_PROVIDER,
        )
    except Exception:
        return None
    parsed = _parse_json_object(raw)
    if not parsed:
        return None
    try:
        parsed["faithfulness"] = float(parsed.get("faithfulness", 0))
        parsed["answer_relevance"] = float(parsed.get("answer_relevance", 0))
    except (TypeError, ValueError):
        return None
    parsed["unsupported_claims"] = parsed.get("unsupported_claims") or []
    parsed["context_support"] = str(parsed.get("context_support", "unknown"))
    parsed["notes"] = str(parsed.get("notes", ""))
    return parsed


def answer_stream(
    question: str,
    history: list[dict] | None = None,
    top_k: int | None = None,
    verify: bool | None = None,
    quality: str | None = None,
    mode: str | None = None,
) -> Iterator[dict]:
    started = time.perf_counter()
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "message": "Empty question."}
        return

    mode = (mode or config.ANSWER_MODE or "auto").strip().lower()
    if mode not in {"auto", "docs"}:
        mode = "auto"

    if prompts.is_smalltalk(question):
        yield {"type": "sources", "sources": [], "retrieval_ms": 0}
        yield {"type": "delta", "text": prompts.smalltalk_reply(question)}
        yield {
            "type": "done",
            "citations": [],
            "refused": False,
            "provider": None,
            "model": None,
            "groundedness": None,
            "smalltalk": True,
            "mode": mode,
            "general_knowledge": False,
            "total_ms": int((time.perf_counter() - started) * 1000),
        }
        return

    # Practice-question requests and answers to an earlier practice question.
    quiz = prompts.is_quiz_request(question)
    grading_question = (
        prompts.previous_practice_question(history)
        if prompts.is_quiz_answer(question, history)
        else None
    )

    # "Give me a question from chapter 4" filters retrieval to that file, so the
    # quiz really comes from the chapter the user named.
    chapter = prompts.detect_chapter_number(question, history, quiz=quiz)
    where = None
    if chapter:
        source = _chapter_source(chapter)
        if source:
            where = {"source": source}

    retrieval_query = question
    if grading_question:
        retrieval_query = f"{grading_question[:800]}\n{question}"
    elif chapter and not where:
        retrieval_query = f"chapter {chapter} {retrieval_query}"

    try:
        chunks = retriever.retrieve(retrieval_query, k=top_k, where=where)
        if not chunks and where:
            # Chapter file missing or filter returned nothing - search everything.
            chunks = retriever.retrieve(retrieval_query, k=top_k)
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"Retrieval failed: {exc}"}
        return

    sources = _public_sources(chunks)
    yield {
        "type": "sources",
        "sources": sources,
        "retrieval_ms": int((time.perf_counter() - started) * 1000),
    }

    # Strict documents-only mode keeps the original refusal behaviour.
    if not chunks and mode == "docs":
        yield {"type": "delta", "text": prompts.REFUSAL}
        yield {
            "type": "done",
            "citations": [],
            "refused": True,
            "provider": None,
            "groundedness": None,
            "mode": mode,
            "general_knowledge": False,
            "total_ms": int((time.perf_counter() - started) * 1000),
        }
        return

    context = prompts.build_context(chunks) if chunks else ""
    messages = _normalise_history(history)

    if chunks:
        system_prompt = (
            prompts.ASSISTANT_PROMPT if mode == "auto" else prompts.SYSTEM_PROMPT
        )
        user_prompt = prompts.build_user_prompt(
            question,
            chunks,
            mode=mode,
            quiz=quiz,
            grading_question=grading_question,
        )
    else:
        system_prompt = prompts.ASSISTANT_PROMPT
        user_prompt = prompts.build_general_prompt(
            question,
            quiz=quiz,
            grading_question=grading_question,
        )
    messages.append({"role": "user", "content": user_prompt})

    parts: list[str] = []
    provider_label = None
    provider_model = None
    try:
        for event in llm.stream_chat(
            system_prompt,
            messages,
            quality=quality,
        ):
            if event["type"] == "provider":
                provider_label = event.get("label")
                provider_model = event.get("model")
                yield {
                    "type": "provider",
                    "label": provider_label,
                    "model": provider_model,
                }
            elif event["type"] == "delta":
                parts.append(event["text"])
                yield {"type": "delta", "text": event["text"]}
            elif event["type"] == "done":
                provider_model = event.get("model") or provider_model
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
        return

    answer = "".join(parts).strip()

    cited_numbers: list[int] = sorted(
        {int(n) for n in _CITATION_RE.findall(answer) if 1 <= int(n) <= len(chunks)}
    )
    citations = [
        {
            "n": number,
            "source": chunks[number - 1].metadata.get("source", "unknown"),
            "page": chunks[number - 1].metadata.get("page"),
            "section": chunks[number - 1].metadata.get("section"),
        }
        for number in cited_numbers
    ]

    # Auto mode: an answer with no citations did not rely on the library. The UI
    # uses this to hide unused snippets and show a "general knowledge" badge.
    general_knowledge = mode == "auto" and not citations
    if mode == "auto" and re.search(
        r"(?im)^\s*(?:\*\*)?general knowledge(?:\*\*)?\s*:", answer
    ):
        general_knowledge = True

    # Groundedness judges only make sense in strict document mode; in auto mode
    # a legitimate general-knowledge section would be scored as unsupported.
    should_verify = (
        config.GROUNDEDNESS_CHECK if verify is None else verify
    ) and mode == "docs"
    refused = mode == "docs" and prompts.is_refusal(answer)
    groundedness = None
    if should_verify and not refused:
        yield {"type": "verifying"}
        groundedness = judge_answer(question, context, answer)

    yield {
        "type": "done",
        "citations": citations,
        "refused": refused,
        "provider": provider_label,
        "model": provider_model,
        "groundedness": groundedness,
        "mode": mode,
        "general_knowledge": general_knowledge,
        "quiz": quiz,
        "grading": bool(grading_question),
        "total_ms": int((time.perf_counter() - started) * 1000),
    }


def answer(question: str, history: list[dict] | None = None, top_k: int | None = None) -> dict:
    """Convenience wrapper used by the evaluation script."""
    text = []
    result: dict[str, Any] = {"answer": "", "sources": [], "groundedness": None}
    for event in answer_stream(question, history=history, top_k=top_k, verify=True):
        if event["type"] == "delta":
            text.append(event["text"])
        elif event["type"] == "sources":
            result["sources"] = event["sources"]
        elif event["type"] == "done":
            result["groundedness"] = event.get("groundedness")
            result["citations"] = event.get("citations", [])
        elif event["type"] == "error":
            result["error"] = event["message"]
    result["answer"] = "".join(text).strip()
    return result
