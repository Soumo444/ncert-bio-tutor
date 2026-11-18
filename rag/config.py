"""Central configuration for the RAG chatbot.

Everything is read from `.env` (see `.env.example`). Nothing here downloads
models - the LLM and the embeddings both live in the cloud.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"
MANIFEST_PATH = DATA_DIR / "manifest.json"
EVAL_DIR = ROOT_DIR / "eval"

COLLECTION_NAME = "documents"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xlsm"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #
LLM_PROVIDER = (os.getenv("LLM_PROVIDER", "groq") or "groq").strip().lower()
LLM_FALLBACKS = [
    p.strip().lower()
    for p in os.getenv("LLM_FALLBACKS", "gemini").split(",")
    if p.strip()
]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
# "Strong" mode - Pro has zero free quota on new keys, 3.5 Flash is the
# strongest Gemini model that actually answers on the free tier.
GEMINI_MODEL_STRONG = os.getenv("GEMINI_MODEL_STRONG", "gemini-3.5-flash").strip()
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash"
    ).split(",")
    if m.strip()
]
GEMINI_THINKING_BUDGET = _int("GEMINI_THINKING_BUDGET", 0)


def gemini_models_for(quality: str | None) -> list[str]:
    """Ordered Gemini models to try for this quality level."""
    first = GEMINI_MODEL_STRONG if _is_strong(quality) else GEMINI_MODEL
    ordered = [first]
    for model in GEMINI_FALLBACK_MODELS:
        if model not in ordered:
            ordered.append(model)
    return ordered


def _is_strong(quality: str | None) -> bool:
    return (quality or "").strip().lower() in {"strong", "pro", "max", "best", "deep"}


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_REASONING_FAST = os.getenv("GROQ_REASONING_FAST", "low").strip()
GROQ_REASONING_STRONG = os.getenv("GROQ_REASONING_STRONG", "medium").strip()
GROQ_MAX_TOKENS_STRONG = _int("GROQ_MAX_TOKENS_STRONG", 3000)


def groq_reasoning_for(quality: str | None) -> str:
    return GROQ_REASONING_STRONG if _is_strong(quality) else GROQ_REASONING_FAST


def answer_model_label(quality: str | None) -> str:
    """Human label of the model that will most likely answer."""
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return GROQ_MODEL
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        return GEMINI_MODEL_STRONG if _is_strong(quality) else GEMINI_MODEL
    return f"{LLM_PROVIDER}"


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
).strip()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()

TEMPERATURE = _float("TEMPERATURE", 0.2)
MAX_OUTPUT_TOKENS = _int("MAX_OUTPUT_TOKENS", 1200)
REQUEST_TIMEOUT = _float("REQUEST_TIMEOUT", 120.0)
GROUNDEDNESS_CHECK = _bool("GROUNDEDNESS_CHECK", True)
# The judge is a second LLM call per answer. Run it on a different free quota
# (Gemini) so it does not eat the answer provider's tokens-per-minute budget.
JUDGE_PROVIDER = (os.getenv("JUDGE_PROVIDER", "gemini") or "").strip().lower()

# "auto"  = ChatGPT-style: use the documents when they help, otherwise answer
#           from general knowledge (the UI labels which one happened).
# "docs"  = strict: refuse when the documents do not contain the answer.
ANSWER_MODE = (os.getenv("ANSWER_MODE", "auto") or "auto").strip().lower()

# --------------------------------------------------------------------------- #
# Embeddings (cloud, free tier - no files on disk)
# --------------------------------------------------------------------------- #
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "gemini").strip().lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001").strip()
EMBED_DIM = _int("EMBED_DIM", 768)
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 30)
# Free tier counts every text as one request, capped at 100/minute.
# 30 texts per batch + this pause keeps us around 85/minute.
EMBED_SLEEP_SECONDS = _float("EMBED_SLEEP_SECONDS", 21.0)

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE = _int("CHUNK_SIZE", 900)          # characters per chunk
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)    # characters of overlap
CSV_ROWS_PER_CHUNK = _int("CSV_ROWS_PER_CHUNK", 15)

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
TOP_K = _int("TOP_K", 6)
MIN_SCORE = _float("MIN_SCORE", 0.20)   # cosine similarity floor
USE_MMR = _bool("USE_MMR", True)
MMR_LAMBDA = _float("MMR_LAMBDA", 0.6)

# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
HOST = os.getenv("HOST", "127.0.0.1").strip()
PORT = _int("PORT", 8000)

# Public deployments set this so visitors can chat but cannot upload, reindex
# or delete documents. The index is baked into the image.
READ_ONLY_MODE = _bool("READ_ONLY_MODE", False)

# Per-IP cap on chat requests. 0 = unlimited (good for local use).
RATE_LIMIT_PER_HOUR = _int("RATE_LIMIT_PER_HOUR", 0)


def ensure_dirs() -> None:
    for path in (DATA_DIR, DOCS_DIR, CHROMA_DIR, EVAL_DIR / "reports"):
        path.mkdir(parents=True, exist_ok=True)


def provider_status() -> dict:
    """Which providers are configured (used by the health endpoint)."""
    return {
        "gemini": bool(GEMINI_API_KEY),
        "groq": bool(GROQ_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        # Ollama needs no key but does need a local server; only claim it when
        # the user actually selected it as a provider.
        "ollama": LLM_PROVIDER == "ollama" or "ollama" in LLM_FALLBACKS,
    }
