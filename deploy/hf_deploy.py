"""Deploy the pre-built RAG index to a free Hugging Face Docker Space.

Usage:
    .\\.venv\\Scripts\\hf.exe auth login      # once, paste a Write token
    .\\.venv\\Scripts\\python.exe deploy\\hf_deploy.py
    .\\.venv\\Scripts\\python.exe deploy\\hf_deploy.py --token hf_xxx

The script stages a clean copy (app + index, no PDFs, no .env), creates the
Space, sets the two API keys as Space secrets, uploads everything and waits
for the container to come online.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, LocalTokenNotFoundError

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
SPACE_SRC = HERE / "hf_space"
STAGING = HERE / ".staging"
DEFAULT_SPACE = "ncert-bio-tutor"
TOKEN_FILE = HERE / "hf_token.txt"
REQUIRED_SECRETS = ("GROQ_API_KEY", "GEMINI_API_KEY")


def log(message: str) -> None:
    print(message, flush=True)


def stage_folder() -> Path:
    """Build a fresh, minimal deployment folder and return its path."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "data").mkdir(parents=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".env")
    shutil.copytree(ROOT / "app", STAGING / "app", ignore=ignore)
    shutil.copytree(ROOT / "rag", STAGING / "rag", ignore=ignore)
    shutil.copy2(ROOT / "ingest.py", STAGING / "ingest.py")
    shutil.copytree(ROOT / "data" / "chroma", STAGING / "data" / "chroma")
    shutil.copy2(ROOT / "data" / "manifest.json", STAGING / "data" / "manifest.json")

    for name in ("Dockerfile", "README.md", "requirements.txt"):
        shutil.copy2(SPACE_SRC / name, STAGING / name)

    files = [path for path in STAGING.rglob("*") if path.is_file()]
    size_mb = sum(path.stat().st_size for path in files) / (1024 * 1024)
    log(f"Staged {len(files)} files ({size_mb:.1f} MB) in {STAGING}")
    return STAGING


def app_url(username: str, space: str) -> str:
    owner = username.lower().replace("_", "-")
    repo = space.lower().replace("_", "-")
    return f"https://{owner}-{repo}.hf.space"


def wait_for_space(api: HfApi, repo_id: str, timeout: int) -> bool:
    log("Waiting for the Space to build (first build can take 5-10 minutes)...")
    deadline = time.time() + timeout
    last_stage = None
    while time.time() < deadline:
        try:
            info = api.space_info(repo_id)
            stage = info.runtime.stage if info.runtime else "UNKNOWN"
        except HfHubHTTPError as exc:
            log(f"  could not read Space status: {exc}")
            stage = "UNKNOWN"

        if stage != last_stage:
            log(f"  stage: {stage}")
            last_stage = stage

        if stage == "RUNNING":
            return True
        if stage in {"BUILD_ERROR", "CONFIG_ERROR", "APP_START_ERROR", "STOPPED"}:
            log(f"Space stopped with stage {stage}. Check the build logs in the browser.")
            return False
        time.sleep(15)
    log("Timed out waiting for the Space to start.")
    return False


def check_health(url: str, attempts: int = 12) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=30) as response:
                body = response.read().decode("utf-8", errors="replace")
            if response.status == 200:
                log(f"Health check OK: {body[:220]}")
                return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == attempts:
                log(f"Health check failed after {attempts} tries: {exc}")
        if attempt < attempts:
            time.sleep(10)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default="", help="Hugging Face write token")
    parser.add_argument("--token-file", default=str(TOKEN_FILE), help="File containing the token")
    parser.add_argument("--name", default=DEFAULT_SPACE, help="Space name (default: ncert-bio-tutor)")
    parser.add_argument("--private", action="store_true", help="Create a private Space")
    parser.add_argument("--skip-secrets", action="store_true", help="Do not upload API keys as secrets")
    parser.add_argument("--skip-wait", action="store_true", help="Upload and exit without waiting for the build")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds to wait for the build")
    parser.add_argument("--stage-only", action="store_true", help="Only prepare the deployment folder")
    args = parser.parse_args()

    folder = stage_folder()
    if args.stage_only:
        return 0

    token = args.token.strip()
    if not token:
        token = os.getenv("HF_TOKEN", "").strip()
    if not token and Path(args.token_file).is_file():
        token = Path(args.token_file).read_text(encoding="utf-8").strip()

    # No token given: huggingface_hub can use a saved `hf auth login` session.
    api = HfApi(token=token) if token else HfApi()
    try:
        who = api.whoami()
    except (HfHubHTTPError, LocalTokenNotFoundError) as exc:
        log(f"ERROR: not logged in / token rejected: {exc}")
        log("  1. Create a free account: https://huggingface.co/join")
        log("  2. Create a Write token: https://huggingface.co/settings/tokens")
        log("  3. Either save it in deploy/hf_token.txt, or run:")
        log("       .venv\\Scripts\\hf.exe auth login")
        log("     and paste the token there.")
        return 2

    username = who.get("name") or ""
    if not username:
        log("ERROR: could not read the account name from the token.")
        return 2

    repo_id = f"{username}/{args.name}"
    url = f"https://huggingface.co/spaces/{repo_id}"
    live = app_url(username, args.name)
    log(f"Account: {username}")
    log(f"Space:   {url}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    if not args.skip_secrets:
        env = dotenv_values(ROOT / ".env")
        for key in REQUIRED_SECRETS:
            value = (env.get(key) or os.getenv(key) or "").strip()
            if not value:
                log(f"ERROR: {key} is missing from .env; refusing to deploy a broken Space.")
                return 2
            try:
                api.add_space_secret(repo_id=repo_id, key=key, value=value)
                log(f"  secret set: {key}")
            except HfHubHTTPError as exc:
                log(f"  could not set secret {key}: {exc}")
                log("  Add it manually under Settings -> Variables and secrets.")
                return 2

    log("Uploading files...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(folder),
        commit_message="Deploy NCERT Bio Tutor",
        ignore_patterns=["**/__pycache__/**", "**/*.pyc", ".env"],
    )
    log("Upload complete.")

    if args.skip_wait:
        log(f"Live URL (after the build finishes): {live}")
        return 0

    if not wait_for_space(api, repo_id, args.timeout):
        log(f"Check the build logs at {url}")
        return 1

    log(f"Verifying {live} ...")
    if check_health(live):
        log("")
        log(f"LIVE: {live}")
        log(f"Space page: {url}")
        return 0

    log(f"The Space is running but the health check did not answer yet: {live}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
