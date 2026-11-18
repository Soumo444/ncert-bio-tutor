"""Create the public GitHub repo that Render builds from.

The repo only contains the Dockerfile, requirements.txt and README.  The app
and the 706-chunk Chroma index are downloaded from the public Hugging Face
bundle during the Render build, so no secrets or PDFs are stored here.

Usage:
    .\\.venv\\Scripts\\python.exe deploy\\github_upload.py
    .\\.venv\\Scripts\\python.exe deploy\\github_upload.py --token ghp_xxx
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "render_repo"
TOKEN_FILE = HERE / "github_token.txt"
REPO_NAME = "ncert-bio-tutor-render"
API = "https://api.github.com"


def log(message: str) -> None:
    print(message, flush=True)


def headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ncert-bio-tutor-deploy/1.0",
    }


def error_text(response: requests.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("message") or body)[:1000]
        return str(body)[:1000]
    except ValueError:
        return (response.text or "")[:1000]


def read_token(args: argparse.Namespace) -> str:
    token = (args.token or "").strip()
    if not token:
        token = (os.getenv("GITHUB_TOKEN") or "").strip()
    if not token and Path(args.token_file).is_file():
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default="", help="GitHub personal access token (ghp_...)")
    parser.add_argument("--token-file", default=str(TOKEN_FILE), help="File containing the token")
    parser.add_argument("--repo", default=REPO_NAME, help="Repository name")
    args = parser.parse_args()

    token = read_token(args)
    if not token:
        log("ERROR: no GitHub token.")
        log("  Open https://github.com/settings/tokens/new?scopes=public_repo")
        log("  Create a token with the public_repo scope, then save it in:")
        log(f"    {TOKEN_FILE}")
        log("  or run: .venv\\Scripts\\python.exe deploy\\github_upload.py --token ghp_xxx")
        return 2

    h = headers(token)
    user = requests.get(f"{API}/user", headers=h, timeout=60)
    if user.status_code != 200:
        log(f"ERROR: GitHub rejected the token: HTTP {user.status_code} {error_text(user)}")
        return 2
    login = user.json().get("login")
    if not login:
        log("ERROR: could not read the GitHub username.")
        return 2
    log(f"GitHub account: {login}")

    repo_url = f"https://github.com/{login}/{args.repo}"
    created = requests.post(
        f"{API}/user/repos",
        headers=h,
        json={
            "name": args.repo,
            "private": False,
            "auto_init": True,
            "description": "Dockerfile for the NCERT Bio Tutor Render deployment",
        },
        timeout=60,
    )
    if created.status_code == 201:
        log(f"Created repository: {repo_url}")
        default_branch = created.json().get("default_branch") or "main"
    elif created.status_code == 422:
        log(f"Repository already exists, updating: {repo_url}")
        info = requests.get(f"{API}/repos/{login}/{args.repo}", headers=h, timeout=60)
        if info.status_code != 200:
            log(f"ERROR: could not read the repository: HTTP {info.status_code} {error_text(info)}")
            return 2
        default_branch = info.json().get("default_branch") or "main"
    else:
        log(f"ERROR: could not create the repository: HTTP {created.status_code} {error_text(created)}")
        return 2

    files = ["Dockerfile", "requirements.txt", "README.md"]
    for name in files:
        content = (SOURCE / name).read_bytes()
        encoded = base64.b64encode(content).decode("ascii")
        url = f"{API}/repos/{login}/{args.repo}/contents/{name}"
        existing = requests.get(url, headers=h, timeout=60)
        payload: dict[str, str] = {
            "message": f"Add {name}",
            "content": encoded,
            "branch": default_branch,
        }
        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]
        response = requests.put(url, headers=h, json=payload, timeout=60)
        if response.status_code not in {200, 201}:
            log(f"ERROR uploading {name}: HTTP {response.status_code} {error_text(response)}")
            return 2
        log(f"  uploaded: {name}")

    log("")
    log(f"GitHub repo ready: {repo_url}")
    log(f"Render repo URL: {repo_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
