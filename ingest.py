"""Ingest documents from data/docs into the vector database.

Usage:
    python ingest.py                 # add/update everything, skip unchanged files
    python ingest.py --force         # re-embed everything
    python ingest.py --reset         # wipe the collection first
    python ingest.py --file x.pdf    # one file (path inside data/docs or absolute)
    python ingest.py --chunk-size 700 --overlap 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rag import chunking, config, embeddings, loaders, store


# --------------------------------------------------------------------------- #
# Manifest: remembers which file version is already embedded
# --------------------------------------------------------------------------- #
def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        try:
            return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.DOCS_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


# --------------------------------------------------------------------------- #
# Core ingest
# --------------------------------------------------------------------------- #
def ingest_file(
    path: Path,
    manifest: dict,
    force: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    source = _relative(path)
    digest = loaders.file_hash(path)
    size_kb = round(path.stat().st_size / 1024, 1)

    previous = manifest.get(source)
    if not force and previous and previous.get("hash") == digest:
        print(f"  = {source}: unchanged ({previous.get('chunks', 0)} chunks), skipped")
        return {"source": source, "status": "skipped", "chunks": previous.get("chunks", 0)}

    print(f"  > {source} ({size_kb} KB) parsing...")
    pages = loaders.load_file(path)
    if not pages:
        print(f"  ! {source}: no extractable text (scanned PDF?)")
        return {"source": source, "status": "empty", "chunks": 0}

    # use the relative path as the display/citation name
    for page in pages:
        page.source = source

    chunks = chunking.chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print(f"  ! {source}: produced no chunks")
        return {"source": source, "status": "empty", "chunks": 0}

    print(f"  > {source}: {len(pages)} page(s) -> {len(chunks)} chunks, embedding...")

    texts = [c.embed_text for c in chunks]
    vectors: list[list[float]] = []
    batch_size = config.EMBED_BATCH_SIZE
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embeddings.embed_texts(batch))
        done = min(start + batch_size, len(texts))
        print(f"    embedded {done}/{len(texts)}", end="\r", flush=True)
        if done < len(texts):
            time.sleep(config.EMBED_SLEEP_SECONDS)  # stay under 100 texts/minute
    print(" " * 40, end="\r")

    store.delete_source(source)
    store.add_chunks(chunks, vectors)

    manifest[source] = {
        "hash": digest,
        "chunks": len(chunks),
        "pages": len(pages),
        "bytes": path.stat().st_size,
        "chunk_size": chunk_size or config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"  + {source}: stored {len(chunks)} chunks")
    return {"source": source, "status": "ingested", "chunks": len(chunks)}


def run(
    force: bool = False,
    reset: bool = False,
    only: list[Path] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    config.ensure_dirs()

    if reset:
        print("Resetting collection...")
        store.reset()

    manifest = load_manifest()
    if reset:
        manifest = {}

    if only:
        paths = only
    else:
        paths = loaders.iter_documents()

    if not paths:
        print(
            "No documents found.\n"
            f"Put your files in: {config.DOCS_DIR}\n"
            "Supported: PDF, DOCX, TXT, MD, CSV, XLSX"
        )
        return {"files": 0, "chunks": 0, "results": []}

    print(f"Ingesting {len(paths)} file(s) from {config.DOCS_DIR}\n")
    results = []
    started = time.time()
    for path in paths:
        try:
            results.append(
                ingest_file(
                    path,
                    manifest,
                    force=force,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
            save_manifest(manifest)
        except Exception as exc:  # noqa: BLE001
            print(f"  x {path.name}: FAILED - {exc}")
            results.append({"source": path.name, "status": "error", "error": str(exc)})

    # drop manifest entries whose files disappeared
    if not only:
        known = {_relative(p) for p in paths}
        for name in list(manifest.keys()):
            if name not in known:
                print(f"  - {name}: removed from docs folder, deleting from index")
                store.delete_source(name)
                manifest.pop(name, None)
        save_manifest(manifest)

    total_chunks = store.count()
    elapsed = round(time.time() - started, 1)
    ingested = sum(1 for r in results if r["status"] == "ingested")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = [r for r in results if r["status"] in {"error", "empty"}]

    print(f"\nDone in {elapsed}s - {ingested} ingested, {skipped} skipped, {len(errors)} failed")
    print(f"Collection now holds {total_chunks} chunks from {len(store.list_documents())} document(s)")
    return {
        "files": len(paths),
        "chunks": total_chunks,
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
        "results": results,
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    parser.add_argument("--force", action="store_true", help="re-embed even unchanged files")
    parser.add_argument("--file", action="append", help="only ingest this file (can repeat)")
    parser.add_argument("--chunk-size", type=int, default=None, help="characters per chunk")
    parser.add_argument("--overlap", type=int, default=None, help="overlap characters")
    args = parser.parse_args()

    only = None
    if args.file:
        only = []
        for raw in args.file:
            path = Path(raw)
            if not path.is_absolute():
                candidate = config.DOCS_DIR / raw
                path = candidate if candidate.exists() else path
            if not path.exists():
                print(f"File not found: {raw}")
                return 1
            only.append(path)

    try:
        result = run(
            force=args.force,
            reset=args.reset,
            only=only,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )
    except embeddings.EmbeddingError as exc:
        print(f"\nEmbedding failed: {exc}")
        return 2

    if result.get("errors"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
