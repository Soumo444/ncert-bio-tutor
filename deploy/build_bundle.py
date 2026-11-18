"""Build the deployment bundle for the Render Docker image.

The bundle contains the application code, the 706-chunk Chroma index and the
manifest - but no PDFs, no `.env` and no virtual environment.  The Render
Dockerfile downloads this tarball during the build, which keeps the GitHub
repository small and avoids Hugging Face LFS pointer files.

Usage:
    .\\.venv\\Scripts\\python.exe deploy\\build_bundle.py
    .\\.venv\\Scripts\\python.exe deploy\\build_bundle.py --output D:\\bundle.tar.gz

After it finishes, upload the new bundle to the public dataset used by the
Dockerfile:

    https://huggingface.co/datasets/Soumojit444/ncert-bio-tutor-deploy

(replace `bundle.tar.gz` there), then trigger a deploy in Render with the build
cache cleared: **Manual Deploy -> Clear build cache & deploy**. A normal deploy
reuses the cached download layer and would keep serving the old code.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "bundle.tar.gz"


def skip_cache(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Keep bytecode caches out of the bundle."""
    if "__pycache__" in Path(info.name).parts or info.name.endswith(".pyc"):
        return None
    return info


def build(output: Path) -> tuple[int, int]:
    required = [
        ROOT / "app",
        ROOT / "rag",
        ROOT / "ingest.py",
        ROOT / "data" / "chroma",
        ROOT / "data" / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("ERROR: missing required path(s): " + ", ".join(missing))

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with tarfile.open(output, "w:gz") as tar:
        for name in ("app", "rag"):
            tar.add(ROOT / name, arcname=name, filter=skip_cache)
        tar.add(ROOT / "ingest.py", arcname="ingest.py")
        tar.add(ROOT / "data" / "chroma", arcname="data/chroma", filter=skip_cache)
        tar.add(ROOT / "data" / "manifest.json", arcname="data/manifest.json")
        # Pinned runtime dependencies; kept inside the bundle for completeness.
        requirements = HERE / "render_repo" / "requirements.txt"
        if requirements.is_file():
            tar.add(requirements, arcname="requirements.txt")

    with tarfile.open(output) as tar:
        members = tar.getmembers()
    return len(members), output.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the tarball (default: deploy/bundle.tar.gz)",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    files, size = build(output)
    print(f"Bundle: {output}")
    print(f"Files:  {files}")
    print(f"Size:   {size / (1024 * 1024):.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
