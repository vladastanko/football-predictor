"""Resumable collection of fixtures, lineups, player statistics and injuries.

Request plan per league-season (free plan friendly):
    1   /leagues          coverage flags (does the API have lineups/players/injuries?)
    1   /fixtures         list of all matches in the season
    ~19 /fixtures?ids=    full details for 20 matches per request (lineups + player stats)
    1+  /injuries         all injury / suspension records for the season
About 22-25 requests per season on paid plans.

Free plans may not allow the `ids` parameter. The collector then switches
automatically to one request per match (/fixtures/players), which still gives every
player's minutes, rating, goals and whether he started. That costs ~380 requests per
Premier League season, i.e. about 4 days on the free plan.
Each match is stored as its own JSON file, so an interrupted run simply resumes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from .api import ApiFootball, BudgetExhausted, PlanRestriction

log = logging.getLogger(__name__)

FINISHED = {"FT", "AET", "PEN"}


def _all_pages(api: ApiFootball, endpoint: str, params: dict) -> list:
    data = api.get(endpoint, params)
    out = list(data.get("response", []))
    total = (data.get("paging") or {}).get("total", 1) or 1
    for page in range(2, total + 1):
        out += api.get(endpoint, {**params, "page": page}).get("response", [])
    return out


def _save(detail_dir: Path, d: dict, league_code: str) -> None:
    d["_league_code"] = league_code
    (detail_dir / f"{d['fixture']['id']}.json").write_text(json.dumps(d), encoding="utf-8")


def collect_season(api: ApiFootball, league_code: str, league_id: int, season: int,
                   store: Path, batch: int) -> dict:
    status = {"league": league_code, "season": season}

    lg = api.get("leagues", {"id": league_id, "season": season}).get("response", [])
    if lg:
        cov = (lg[0].get("seasons") or [{}])[0].get("coverage", {})
        status["league_name"] = lg[0].get("league", {}).get("name")
        status["coverage"] = {
            "lineups": cov.get("fixtures", {}).get("lineups"),
            "players": cov.get("fixtures", {}).get("statistics_players"),
            "injuries": cov.get("injuries"),
        }

    fixtures = _all_pages(api, "fixtures", {"league": league_id, "season": season})
    finished = [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") in FINISHED]
    status["fixtures_total"] = len(fixtures)
    status["fixtures_finished"] = len(finished)

    detail_dir = store / "fixture_detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    todo = [f["fixture"]["id"] for f in finished
            if not (detail_dir / f"{f['fixture']['id']}.json").exists()]
    log.info("%s %s: %d finished matches, %d still to fetch",
             league_code, season, len(finished), len(todo))

    inj_file = store / "injuries" / f"{league_code}_{season}.json"
    if not inj_file.exists():
        injuries = [r for r in _all_pages(api, "injuries", {"league": league_id, "season": season})
                    if isinstance(r, dict)]
        inj_file.parent.mkdir(parents=True, exist_ok=True)
        inj_file.write_text(json.dumps(injuries), encoding="utf-8")
    status["injury_records"] = len(json.loads(inj_file.read_text(encoding="utf-8")))

    mode_file = store / "_fetch_mode.txt"
    mode = mode_file.read_text().strip() if mode_file.exists() else "batch"
    by_id = {f["fixture"]["id"]: f for f in finished}

    if mode == "batch":
        try:
            for i in range(0, len(todo), batch):
                ids = todo[i:i + batch]
                details = api.get("fixtures", {"ids": "-".join(map(str, ids))}).get("response", [])
                for d in details:
                    _save(detail_dir, d, league_code)
        except PlanRestriction as exc:
            if "ids" not in str(exc).lower():
                raise
            log.warning("Plan does not allow batched requests; switching to one request "
                        "per match (slower, same data).")
            mode = "single"
            mode_file.write_text(mode)

    if mode == "single":
        todo = [i for i in todo if not (detail_dir / f"{i}.json").exists()]
        for n, fid in enumerate(todo, 1):
            players = api.get("fixtures/players", {"fixture": fid}).get("response", [])
            d = dict(by_id[fid])
            d["players"] = players
            d["lineups"] = []          # starters are taken from each player's substitute flag
            _save(detail_dir, d, league_code)
            if n % 20 == 0:
                log.info("%s %s: %d/%d matches fetched", league_code, season, n, len(todo))

    done = sum((detail_dir / f"{f['fixture']['id']}.json").exists() for f in finished)
    status["fixtures_collected"] = done
    return status


def collect(cfg: dict, api: ApiFootball, store: Path) -> list[dict]:
    p = cfg["players"]
    results = []
    try:
        for league, seasons in p["collect"].items():
            for season in seasons:
                results.append(collect_season(api, league, p["league_ids"][league], season,
                                              store, p["fixtures_per_batch"]))
    except BudgetExhausted as exc:
        log.warning("Stopped: %s Run `fpred players-collect` again tomorrow to continue.", exc)
    except requests.RequestException as exc:
        log.error("Stopped because the API could not be reached (%s). Everything fetched so far "
                  "is saved; check the internet connection and run `fpred players-collect` "
                  "again to continue.", type(exc).__name__)
    except PlanRestriction as exc:
        log.error("The API plan does not allow this request: %s\n"
                  "If it names a season, the free plan does not cover it: "
                  "change players.collect in config/config.yaml.", exc)
    (store / "_collection_status.json").write_text(json.dumps(results, indent=2))
    for r in results:
        log.info("%s %s: %s/%s matches collected, %s injury records, coverage %s",
                 r["league"], r["season"], r.get("fixtures_collected"),
                 r.get("fixtures_finished"), r.get("injury_records"), r.get("coverage"))
    log.info("Requests left today (own counter): %d", api.requests_left())
    return results
