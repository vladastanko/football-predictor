"""Download raw CSV files from football-data.co.uk.

Files are cached in data/raw/{league}/{season_code}.csv. Completed seasons are
downloaded once; the current season is always refreshed.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from .config import Paths, season_code

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "football-predictor (portfolio project; contact via GitHub)"}


def _download(url: str, dest: Path, timeout: int) -> bool:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    if resp.status_code == 404:
        log.warning("Not available: %s", url)
        return False
    resp.raise_for_status()
    if len(resp.content) < 200:  # empty placeholder files exist for some seasons
        log.warning("Empty file skipped: %s", url)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True


def download_history(cfg: dict, paths: Paths, refresh_current: bool = True) -> list[Path]:
    """Download every configured league/season. Returns the list of local files."""
    src = cfg["source"]
    first, last = cfg["seasons"]["first"], cfg["seasons"]["last"]
    files: list[Path] = []

    for league in cfg["leagues"]:
        for year in range(first, last + 1):
            code = season_code(year)
            dest = paths.raw / league / f"{code}.csv"
            is_current = year == last
            if dest.exists() and not (is_current and refresh_current):
                files.append(dest)
                continue
            url = f"{src['base_url']}/{code}/{league}.csv"
            log.info("Downloading %s", url)
            try:
                if _download(url, dest, src["request_timeout_s"]):
                    files.append(dest)
            except requests.RequestException as exc:
                log.error("Failed %s: %s", url, exc)
            time.sleep(src["polite_delay_s"])
    return files


def download_fixtures(cfg: dict, paths: Paths) -> Path | None:
    """Download the upcoming-fixtures file (all main leagues in one CSV)."""
    dest = paths.raw / "fixtures.csv"
    try:
        ok = _download(cfg["source"]["fixtures_url"], dest, cfg["source"]["request_timeout_s"])
    except requests.RequestException as exc:
        log.error("Failed to download fixtures: %s", exc)
        return None
    return dest if ok else None
