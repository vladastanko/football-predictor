"""Small, polite API-Football client with an on-disk cache and a daily request budget.

- Every response is cached as JSON under data/api/, so re-running never re-spends requests.
- A request counter (data/api/_usage.json) stops collection before the daily plan limit.
- Requests are spaced out to respect the per-minute limit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class BudgetExhausted(RuntimeError):
    """Daily request budget used up; rerun tomorrow to continue."""


class PlanRestriction(RuntimeError):
    """The API refused the request because of the subscription plan."""


def load_api_key(project_root: Path) -> str:
    key = os.environ.get("API_FOOTBALL_KEY")
    env = project_root / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("API_FOOTBALL_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key or key == "put-your-key-here":
        raise RuntimeError(
            "API key not found. Create a file named .env in the project folder with the line\n"
            "    API_FOOTBALL_KEY=your-key\n"
            "(see .env.example). The key is never stored anywhere else.")
    return key


class ApiFootball:
    def __init__(self, key: str, cfg: dict, cache_dir: Path):
        self.base = cfg["base_url"].rstrip("/")
        self.budget = cfg["daily_request_budget"]
        self.gap = cfg["min_seconds_between_requests"]
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["x-apisports-key"] = key
        self._last = 0.0
        self._usage_file = self.cache / "_usage.json"

    # ---------------------------------------------------------------- usage
    def _usage(self) -> dict:
        today = date.today().isoformat()
        if self._usage_file.exists():
            u = json.loads(self._usage_file.read_text())
            if u.get("date") == today:
                return u
        return {"date": today, "requests": 0}

    def requests_left(self) -> int:
        return self.budget - self._usage()["requests"]

    def _count(self, remaining_header: str | None) -> None:
        u = self._usage()
        u["requests"] += 1
        if remaining_header is not None:
            u["api_reported_remaining"] = int(remaining_header)
        self._usage_file.write_text(json.dumps(u))

    # ---------------------------------------------------------------- calls
    def _cache_path(self, endpoint: str, params: dict) -> Path:
        key = json.dumps(params, sort_keys=True)
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache / endpoint.strip("/").replace("/", "_") / f"{h}.json"

    def _send(self, endpoint: str, params: dict, retries: int = 4) -> requests.Response:
        """GET with retries for network hiccups (DNS, Wi-Fi drop, timeouts, 5xx)."""
        url = f"{self.base}/{endpoint.strip('/')}"
        for attempt in range(retries + 1):
            wait = self.gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self.session.get(url, params=params, timeout=30)
                self._last = time.time()
                if resp.status_code < 500 or attempt == retries:
                    return resp
                problem = f"server error {resp.status_code}"
            except (requests.ConnectionError, requests.Timeout) as exc:
                self._last = time.time()
                if attempt == retries:
                    raise
                problem = type(exc).__name__
            pause = (10, 30, 60, 120)[min(attempt, 3)]
            log.warning("Network problem (%s); retrying in %ds (%d/%d)",
                        problem, pause, attempt + 1, retries)
            time.sleep(pause)
        raise RuntimeError("unreachable")

    def get(self, endpoint: str, params: dict, use_cache: bool = True) -> dict:
        path = self._cache_path(endpoint, params)
        if use_cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.requests_left() <= 0:
            raise BudgetExhausted(f"Daily budget of {self.budget} requests used.")

        resp = self._send(endpoint, params)
        self._count(resp.headers.get("x-ratelimit-requests-remaining"))
        resp.raise_for_status()
        data = resp.json()

        errors = data.get("errors")
        if errors:  # API returns 200 with an "errors" object for plan/limit problems
            msg = json.dumps(errors)
            if any(w in msg.lower() for w in ("plan", "subscription", "access")):
                raise PlanRestriction(msg)
            if "requests" in msg.lower() or "limit" in msg.lower():
                raise BudgetExhausted(msg)
            raise RuntimeError(f"API error for {endpoint} {params}: {msg}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data
