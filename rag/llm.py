"""LLM access with automatic provider fallback.

One interface, several free providers:

* groq       - Groq free tier, very fast open models (primary answerer)
* gemini     - Google AI Studio free tier (embeddings, judge, backup answers)
* openrouter - OpenRouter ":free" models
* ollama     - fully local fallback (optional, no key)

If the primary provider is rate-limited or down, the next configured provider
answers instead, so a quota error never breaks the chat.
"""

from __future__ import annotations

import json
import time
from typing import Iterator

import httpx

from . import config

GEMINI_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Provider implementations - each yields plain text deltas
# --------------------------------------------------------------------------- #
def _gemini_stream(
    system: str, messages: list[dict], model: str | None = None
) -> Iterator[str]:
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY not configured")

    model_id = model or config.GEMINI_MODEL
    url = f"{GEMINI_ROOT}/models/{model_id}:streamGenerateContent"
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
    ]
    generation: dict = {
        "temperature": config.TEMPERATURE,
        "maxOutputTokens": config.MAX_OUTPUT_TOKENS,
    }
    if "2.5" in model_id:
        budget = config.GEMINI_THINKING_BUDGET
        if "pro" in model_id:  # Pro cannot disable thinking
            budget = max(128, budget)
        generation["thinkingConfig"] = {"thinkingBudget": budget}

    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": generation,
    }

    with httpx.Client(timeout=config.REQUEST_TIMEOUT) as client:
        with client.stream(
            "POST",
            url,
            params={"alt": "sse", "key": config.GEMINI_API_KEY},
            json=payload,
        ) as response:
            if response.status_code != 200:
                detail = response.read().decode("utf-8", "ignore")
                raise LLMError(f"Gemini {response.status_code}: {detail[:300]}")
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for candidate in obj.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        text = part.get("text")
                        if text:
                            yield text


def _retry_after(response: httpx.Response, default: float) -> float:
    """Seconds to wait from a Retry-After header ('7.2s' or '7')."""
    raw = (response.headers.get("retry-after") or "").strip().rstrip("sS")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _openai_compatible_stream(
    url: str,
    api_key: str,
    model: str,
    system: str,
    messages: list[dict],
    label: str,
    extra: dict | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    if not api_key:
        raise LLMError(f"{label} API key not configured")

    payload = {
        "model": model,
        "stream": True,
        "temperature": config.TEMPERATURE,
        "max_tokens": max_tokens or config.MAX_OUTPUT_TOKENS,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    if extra:
        payload.update(extra)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if label == "openrouter":
        headers["HTTP-Referer"] = "http://localhost"
        headers["X-Title"] = "Local RAG Chatbot"

    last_error = ""
    for attempt in range(3):
        with httpx.Client(timeout=config.REQUEST_TIMEOUT) as client:
            with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content")
                        if text:
                            yield text
                    return

                detail = response.read().decode("utf-8", "ignore")
                last_error = f"{label} {response.status_code}: {detail[:300]}"
                if response.status_code in {429, 503} and attempt < 2:
                    # Free-tier minute windows reset in seconds. Waiting briefly
                    # keeps this fast model instead of falling back to a slow one.
                    wait = _retry_after(response, default=6.0 * (attempt + 1))
                    time.sleep(min(wait, 20.0))
                    continue
                raise LLMError(last_error)

    raise LLMError(last_error)


def _groq_stream(
    system: str, messages: list[dict], quality: str | None = None
) -> Iterator[str]:
    effort = config.groq_reasoning_for(quality)
    # reasoning models spend max_tokens on hidden reasoning first, so "deep"
    # answers need a larger budget or the visible content comes back empty.
    max_tokens = (
        config.GROQ_MAX_TOKENS_STRONG
        if effort in {"medium", "high"}
        else config.MAX_OUTPUT_TOKENS
    )
    yield from _openai_compatible_stream(
        "https://api.groq.com/openai/v1/chat/completions",
        config.GROQ_API_KEY,
        config.GROQ_MODEL,
        system,
        messages,
        "groq",
        extra={"reasoning_effort": effort},
        max_tokens=max_tokens,
    )


def _openrouter_stream(system: str, messages: list[dict]) -> Iterator[str]:
    yield from _openai_compatible_stream(
        "https://openrouter.ai/api/v1/chat/completions",
        config.OPENROUTER_API_KEY,
        config.OPENROUTER_MODEL,
        system,
        messages,
        "openrouter",
    )


def _ollama_stream(system: str, messages: list[dict]) -> Iterator[str]:
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": True,
        "options": {"temperature": config.TEMPERATURE},
        "messages": [{"role": "system", "content": system}] + messages,
    }
    with httpx.Client(timeout=config.REQUEST_TIMEOUT) as client:
        with client.stream(
            "POST", f"{config.OLLAMA_BASE_URL}/api/chat", json=payload
        ) as response:
            if response.status_code != 200:
                detail = response.read().decode("utf-8", "ignore")
                raise LLMError(f"ollama {response.status_code}: {detail[:300]}")
            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = (obj.get("message") or {}).get("content")
                if text:
                    yield text
                if obj.get("done"):
                    break


_PROVIDERS = {
    "gemini": _gemini_stream,
    "groq": _groq_stream,
    "openrouter": _openrouter_stream,
    "ollama": _ollama_stream,
}

_LABELS = {
    "gemini": "Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "ollama": "Ollama",
}


def _available(provider: str) -> bool:
    if provider == "gemini":
        return bool(config.GEMINI_API_KEY)
    if provider == "groq":
        return bool(config.GROQ_API_KEY)
    if provider == "openrouter":
        return bool(config.OPENROUTER_API_KEY)
    if provider == "ollama":
        return config.LLM_PROVIDER == "ollama"
    return False


def provider_order() -> list[str]:
    order = [config.LLM_PROVIDER] + config.LLM_FALLBACKS
    seen: list[str] = []
    for name in order:
        if name in _PROVIDERS and name not in seen:
            seen.append(name)
    return seen


def _attempts(
    name: str, system: str, messages: list[dict], quality: str | None
):
    """Yield (model_name, generator) attempts for one provider."""
    if name == "gemini":
        for model in config.gemini_models_for(quality):
            yield model, _gemini_stream(system, messages, model)
    elif name == "groq":
        yield config.GROQ_MODEL, _groq_stream(system, messages, quality)
    else:
        yield name, _PROVIDERS[name](system, messages)


def stream_chat(
    system: str,
    messages: list[dict],
    quality: str | None = None,
    model: str | None = None,
    preferred: str | None = None,
) -> Iterator[dict]:
    """Yield {"type": "provider"|"delta"|"done", ...} with automatic fallback.

    `preferred` moves one provider to the front without removing the others,
    so a judge call can use a separate free quota from the main answer.
    """
    errors: list[str] = []

    order = provider_order()
    if preferred and preferred in _PROVIDERS:
        order = [preferred] + [name for name in order if name != preferred]

    for name in order:
        if not _available(name):
            errors.append(f"{name}: no API key")
            continue

        for attempt_model, generator in _attempts(name, system, messages, quality):
            try:
                first = next(generator)
            except StopIteration:
                errors.append(f"{name}/{attempt_model}: empty response")
                continue
            except Exception as exc:  # noqa: BLE001 - any failure means try next
                errors.append(f"{name}/{attempt_model}: {exc}")
                continue

            yield {
                "type": "provider",
                "provider": name,
                "label": _LABELS.get(name, name),
                "model": attempt_model,
            }
            yield {"type": "delta", "text": first}
            try:
                for token in generator:
                    yield {"type": "delta", "text": token}
            except Exception as exc:  # noqa: BLE001
                yield {"type": "delta", "text": f"\n\n_(stream interrupted: {exc})_"}
            yield {"type": "done", "provider": name, "model": attempt_model}
            return

    raise LLMError(
        "No LLM provider is configured or reachable. Add a free GEMINI_API_KEY "
        "to .env (https://aistudio.google.com/apikey). Details: " + " | ".join(errors)
    )


def complete(
    system: str, messages: list[dict], preferred: str | None = None
) -> tuple[str, str]:
    """Non-streaming helper used by the evaluation and groundedness judge."""
    parts: list[str] = []
    provider = "unknown"
    for event in stream_chat(system, messages, preferred=preferred):
        if event["type"] == "provider":
            provider = event["label"]
        elif event["type"] == "delta":
            parts.append(event["text"])
    return "".join(parts), provider
