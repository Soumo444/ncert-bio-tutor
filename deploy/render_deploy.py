"""Deploy the pre-built RAG index to a free Render web service.

Render's free plan needs no credit card and gives a stable
``https://<name>.onrender.com`` URL.  The service builds the Dockerfile from a
tiny public Hugging Face dataset repo and downloads the 5 MB application bundle
(app + 706-chunk Chroma index) during the build.

Usage:
    .\\.venv\\Scripts\\python.exe deploy\\render_deploy.py --token rnd_xxx
    .\\.venv\\Scripts\\python.exe deploy\\render_deploy.py   # reads deploy/render_token.txt

Create a free account at https://dashboard.render.com/register, then create an
API key at https://dashboard.render.com/u/settings#api-keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
TOKEN_FILE = HERE / "render_token.txt"
DEFAULT_NAME = "ncert-bio-tutor"
REPO_URL = "https://huggingface.co/datasets/Soumojit444/ncert-bio-tutor-render"
API = "https://api.render.com/v1"
REQUIRED_SECRETS = ("GROQ_API_KEY", "GEMINI_API_KEY")
FAILED_STATUSES = {
    "build_failed",
    "update_failed",
    "pre_deploy_failed",
    "canceled",
    "deactivated",
}


def log(message: str) -> None:
    print(message, flush=True)


def api_request(
    method: str,
    path: str,
    token: str,
    payload: dict | None = None,
    timeout: int = 60,
) -> requests.Response:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ncert-bio-tutor-deploy/1.0",
    }
    return requests.request(
        method,
        f"{API}{path}",
        headers=headers,
        json=payload,
        timeout=timeout,
    )


def error_text(response: requests.Response) -> str:
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("message") or body.get("error") or body)[:1200]
        return str(body)[:1200]
    except ValueError:
        return (response.text or "")[:1200]


def read_token(args: argparse.Namespace) -> str:
    token = (args.token or "").strip()
    if not token:
        token = (os.getenv("RENDER_API_KEY") or "").strip()
    if not token and Path(args.token_file).is_file():
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    return token


def read_secrets() -> dict[str, str]:
    values = dotenv_values(ROOT / ".env")
    secrets = {}
    for key in REQUIRED_SECRETS:
        value = (values.get(key) or os.getenv(key) or "").strip()
        if not value:
            raise SystemExit(f"ERROR: {key} is missing from .env")
        secrets[key] = value
    return secrets


def create_service(
    token: str,
    owner_id: str,
    name: str,
    region: str,
    plan: str,
    secrets: dict[str, str],
    repo_url: str = REPO_URL,
) -> requests.Response:
    env_vars = [
        {"key": "GROQ_API_KEY", "value": secrets["GROQ_API_KEY"]},
        {"key": "GEMINI_API_KEY", "value": secrets["GEMINI_API_KEY"]},
        {"key": "READ_ONLY_MODE", "value": "true"},
        {"key": "RATE_LIMIT_PER_HOUR", "value": "60"},
    ]
    payload = {
        "type": "web_service",
        "name": name,
        "ownerId": owner_id,
        "repo": repo_url,
        "branch": "main",
        "autoDeployTrigger": "off",
        "envVars": env_vars,
        "serviceDetails": {
            "runtime": "docker",
            "plan": plan,
            "region": region,
            "numInstances": 1,
            "envSpecificDetails": {"dockerfilePath": "./Dockerfile"},
            "healthCheckPath": "/api/health",
        },
    }
    return api_request("POST", "/services", token, payload)


def latest_deploy(token: str, service_id: str) -> dict | None:
    response = api_request("GET", f"/services/{service_id}/deploys?limit=1", token)
    if response.status_code != 200:
        log(f"  could not read deploy status: HTTP {response.status_code} {error_text(response)}")
        return None
    body = response.json()
    if not body:
        return None
    item = body[0]
    return item.get("deploy", item)


def wait_for_deploy(token: str, service_id: str, dashboard: str, timeout: int) -> bool:
    log("Waiting for Render to build and start the service (usually 5-15 minutes)...")
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        deploy = latest_deploy(token, service_id)
        if deploy:
            status = str(deploy.get("status") or "unknown")
            if status != last_status:
                log(f"  deploy status: {status}")
                last_status = status
            if status == "live":
                return True
            if status in FAILED_STATUSES:
                log(f"Deploy failed with status {status}.")
                log(f"Build logs: {dashboard}/logs")
                return False
        time.sleep(15)
    log(f"Timed out waiting for the deploy. Check {dashboard}")
    return False


def check_health(url: str, attempts: int = 20) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(f"{url}/api/health", timeout=30)
            if response.status_code == 200:
                body = response.json()
                log(
                    "Health OK: "
                    f"chunks={body.get('chunk_count')} "
                    f"docs={body.get('document_count')} "
                    f"mode={body.get('answer_mode')} "
                    f"read_only={body.get('read_only')}"
                )
                return True
        except (requests.RequestException, ValueError) as exc:
            if attempt == attempts:
                log(f"Health check failed after {attempts} tries: {exc}")
        if attempt < attempts:
            time.sleep(10)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default="", help="Render API key (rnd_...)")
    parser.add_argument("--token-file", default=str(TOKEN_FILE), help="File containing the API key")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Service name")
    parser.add_argument("--repo", default=REPO_URL, help="Public GitHub/GitLab repository URL")
    parser.add_argument("--region", default="singapore", choices=["singapore", "oregon", "frankfurt", "ohio", "virginia"])
    parser.add_argument("--plan", default="free", help="Render compute plan (default: free)")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds to wait for the build")
    parser.add_argument("--skip-wait", action="store_true", help="Create the service and exit")
    args = parser.parse_args()

    token = read_token(args)
    if not token:
        log("ERROR: no Render API key.")
        log("  1. Create a free account (no credit card): https://dashboard.render.com/register")
        log("  2. Create an API key: https://dashboard.render.com/u/settings?add-api-key")
        log(f"  3. Save it in {TOKEN_FILE}")
        log("     or run: .venv\\Scripts\\python.exe deploy\\render_deploy.py --token rnd_xxx")
        return 2

    try:
        secrets = read_secrets()
    except SystemExit as exc:
        log(str(exc))
        return 2

    owners = api_request("GET", "/owners?limit=20", token)
    if owners.status_code != 200:
        log(f"ERROR: Render rejected the API key: HTTP {owners.status_code} {error_text(owners)}")
        return 2
    body = owners.json()
    owner = (body[0].get("owner") if body and isinstance(body[0], dict) else None) or {}
    owner_id = owner.get("id")
    owner_name = owner.get("name") or owner.get("email") or "workspace"
    if not owner_id:
        log("ERROR: could not find a Render workspace for this API key.")
        return 2
    log(f"Workspace: {owner_name} ({owner_id})")

    plans = [args.plan]
    if args.plan == "free":
        plans.append("0.5c-512mb")
    regions = [args.region]
    if args.region != "oregon":
        regions.append("oregon")

    response = None
    for region in regions:
        for plan in plans:
            response = create_service(token, owner_id, args.name, region, plan, secrets, args.repo)
            if response.status_code in {200, 201}:
                log(f"Created with region={region}, plan={plan}")
                break
            text = error_text(response)
            log(f"Create attempt with plan={plan}, region={region} failed: HTTP {response.status_code} {text}")
            if response.status_code == 402:
                log("Render is asking for a payment method. The free plan should not require one;")
                log("check the workspace billing page or create the service from the dashboard.")
                return 2
        if response is not None and response.status_code in {200, 201}:
            break
    if response is None or response.status_code not in {200, 201}:
        log("ERROR: could not create the Render service.")
        return 1

    created = response.json()
    service = created.get("service") or {}
    service_id = service.get("id")
    details = service.get("serviceDetails") or {}
    url = details.get("url") or ""
    if url and not url.startswith("http"):
        url = "https://" + url
    dashboard = f"https://dashboard.render.com/web/{service_id}" if service_id else "https://dashboard.render.com"
    log(f"Service created: {service_id}")
    log(f"URL: {url or '(pending)'}")

    if args.skip_wait:
        log(f"Dashboard: {dashboard}")
        return 0

    if not wait_for_deploy(token, service_id, dashboard, args.timeout):
        return 1

    if not url:
        service_resp = api_request("GET", f"/services/{service_id}", token)
        if service_resp.status_code == 200:
            details = (service_resp.json().get("service") or {}).get("serviceDetails") or {}
            url = details.get("url") or ""
            if url and not url.startswith("http"):
                url = "https://" + url

    if not url:
        log(f"Service is live but the URL could not be read. Open {dashboard}")
        return 1

    log(f"Verifying {url} ...")
    if not check_health(url):
        log(f"The service is live but /api/health did not answer yet: {url}")
        return 1

    log("")
    log(f"LIVE: {url}")
    log(f"Dashboard: {dashboard}")
    log("Free plan note: the service sleeps after 15 minutes without traffic and")
    log("wakes on the next visit (~1 minute). Add a free uptime pinger to keep it warm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
