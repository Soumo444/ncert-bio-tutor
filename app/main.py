"""FastAPI application: chat API, document management, and the web UI.

Run with:
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from rag import config, llm, pipeline, store
from rag.loaders import file_hash

BASE_DIR = Path(__file__).resolve().parent
config.ensure_dirs()

app = FastAPI(title="RAG Document Chat", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_ingest_lock = threading.Lock()
MAX_UPLOAD_MB = 60

_rate_lock = threading.Lock()
_rate_hits: dict[str, deque[float]] = defaultdict(deque)


def _require_writable() -> None:
    if config.READ_ONLY_MODE:
        raise HTTPException(
            status_code=403,
            detail=(
                "This is a read-only public demo. Uploading, reindexing and "
                "deleting documents are disabled."
            ),
        )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    limit = config.RATE_LIMIT_PER_HOUR
    if limit <= 0:
        return
    now = time.monotonic()
    ip = _client_ip(request)
    with _rate_lock:
        hits = _rate_hits[ip]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Hourly limit of {limit} questions reached. "
                    "Please try again later."
                ),
            )
        hits.append(now)
        # Keep the in-memory map from growing forever on a public deployment.
        if len(_rate_hits) > 5000:
            stale = [
                key
                for key, value in _rate_hits.items()
                if not value or now - value[-1] > 3600
            ]
            for key in stale:
                del _rate_hits[key]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)
    verify: bool | None = None
    quality: str | None = Field(default="fast", pattern="^(fast|strong)$")
    mode: str | None = Field(default=None, pattern="^(auto|docs)$")


class ReindexRequest(BaseModel):
    reset: bool = False
    force: bool = True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _docs_summary() -> dict[str, Any]:
    documents = store.list_documents()
    return {
        "documents": documents,
        "chunk_count": store.count(),
        "document_count": len(documents),
    }


def _safe_docs_path(relative: str) -> Path:
    candidate = (config.DOCS_DIR / relative).resolve()
    root = config.DOCS_DIR.resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate


def _unique_path(name: str) -> Path:
    name = Path(name).name  # strip any directory traversal
    target = config.DOCS_DIR / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for index in range(1, 1000):
        candidate = config.DOCS_DIR / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=409, detail="Too many files with that name")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_top_k": config.TOP_K,
            "default_verify": config.GROUNDEDNESS_CHECK,
            "default_mode": config.ANSWER_MODE,
            "read_only": config.READ_ONLY_MODE,
        },
    )


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(BASE_DIR / "static" / "favicon.svg", media_type="image/svg+xml")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    statuses = config.provider_status()
    active = config.LLM_PROVIDER
    documents = store.list_documents()
    return {
        "ok": True,
        "llm_provider": active,
        "llm_model": config.answer_model_label("fast"),
        "answer_models": {
            "fast": config.answer_model_label("fast"),
            "strong": config.answer_model_label("strong"),
        },
        "providers": statuses,
        "any_provider_ready": any(
            statuses[name] for name in llm.provider_order()
        ),
        "embedding_model": config.EMBED_MODEL,
        "embedding_ready": bool(config.GEMINI_API_KEY),
        "top_k": config.TOP_K,
        "min_score": config.MIN_SCORE,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "groundedness_check": config.GROUNDEDNESS_CHECK,
        "answer_mode": config.ANSWER_MODE,
        "read_only": config.READ_ONLY_MODE,
        "chunk_count": store.count(),
        "document_count": len(documents),
    }


@app.get("/api/documents")
async def documents():
    return _docs_summary()


@app.get("/health")
async def health_alias():
    """Alias for simple uptime checks and container platforms."""
    return await health()


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    _check_rate_limit(request)
    history = [message.model_dump() for message in payload.history]

    def event_stream():
        try:
            for event in pipeline.answer_stream(
                payload.question,
                history=history,
                top_k=payload.top_k,
                verify=payload.verify,
                quality=payload.quality,
                mode=payload.mode,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            payload_error = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(payload_error)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/upload")
async def upload(request: Request, files: list[UploadFile]):
    _require_writable()
    config.ensure_dirs()
    saved: list[Path] = []
    rejected: list[dict[str, str]] = []

    for upload_file in files:
        name = Path(upload_file.filename or "file").name
        suffix = Path(name).suffix.lower()
        if suffix not in config.SUPPORTED_EXTENSIONS:
            rejected.append({"file": name, "reason": f"unsupported type {suffix}"})
            continue
        target = _unique_path(name)
        size = 0
        try:
            with target.open("wb") as out:
                while chunk := await upload_file.read(1024 * 512):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1024 * 1024:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{name} is larger than {MAX_UPLOAD_MB} MB",
                        )
                    out.write(chunk)
            saved.append(target)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            rejected.append({"file": name, "reason": str(exc)})
        finally:
            await upload_file.close()

    if not saved:
        return {"ingested": [], "rejected": rejected, "message": "No valid files uploaded"}

    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another ingest is already running")
    try:
        from ingest import run as ingest_run

        result = await asyncio.to_thread(ingest_run, force=True, only=saved)
    finally:
        _ingest_lock.release()

    return {
        "ingested": [r["source"] for r in result.get("results", []) if r.get("status") == "ingested"],
        "rejected": rejected,
        "result": {k: v for k, v in result.items() if k != "results"},
        **_docs_summary(),
    }


@app.post("/api/reindex")
async def reindex(payload: ReindexRequest):
    _require_writable()
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another ingest is already running")
    try:
        from ingest import run as ingest_run

        result = await asyncio.to_thread(ingest_run, force=payload.force, reset=payload.reset)
    finally:
        _ingest_lock.release()
    return {
        "result": {k: v for k, v in result.items() if k != "results"},
        **_docs_summary(),
    }


@app.delete("/api/documents/{relative_path:path}")
async def delete_document(relative_path: str):
    _require_writable()
    target = _safe_docs_path(relative_path)
    source = relative_path
    if target.exists():
        if not target.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        target.unlink()

    store.delete_source(source)
    try:
        from ingest import load_manifest, save_manifest

        manifest = load_manifest()
        manifest.pop(source, None)
        save_manifest(manifest)
    except Exception:
        pass

    return {"deleted": source, **_docs_summary()}


@app.get("/api/documents/{relative_path:path}/raw")
async def raw_document(relative_path: str):
    target = _safe_docs_path(relative_path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(target, filename=target.name)
