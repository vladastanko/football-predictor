"""Turn cached API-Football JSON into flat, typed tables.

Tables (one row per ...):
    api_fixtures   match                      fixture_id, league, season, date, teams, score
    appearances    player per match           starter flag, minutes, rating, goals, assists, shots...
    injuries       player missing a match     type (Missing Fixture / Questionable), reason
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def parse_details(store: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures, apps = [], []
    for path in sorted((store / "fixture_detail").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        fx, teams, goals = d.get("fixture", {}), d.get("teams", {}), d.get("goals", {})
        fid = fx.get("id")
        fixtures.append({
            "fixture_id": fid, "league": d.get("_league_code"),
            "season": d.get("league", {}).get("season"),
            "date": pd.to_datetime(fx.get("date"), utc=True).tz_convert(None),
            "api_home_id": teams.get("home", {}).get("id"),
            "api_home_name": teams.get("home", {}).get("name"),
            "api_away_id": teams.get("away", {}).get("id"),
            "api_away_name": teams.get("away", {}).get("name"),
            "home_goals": goals.get("home"), "away_goals": goals.get("away"),
            "has_lineups": bool(d.get("lineups")), "has_players": bool(d.get("players")),
        })
        starters = {}
        for lu in d.get("lineups") or []:
            tid = lu.get("team", {}).get("id")
            for s in lu.get("startXI") or []:
                starters[(tid, s.get("player", {}).get("id"))] = True
            for s in lu.get("substitutes") or []:
                starters.setdefault((tid, s.get("player", {}).get("id")), False)
        for team in d.get("players") or []:
            tid = team.get("team", {}).get("id")
            for p in team.get("players") or []:
                pl, st = p.get("player", {}), (p.get("statistics") or [{}])[0]
                g = st.get("games", {}) or {}
                pid = pl.get("id")
                apps.append({
                    "fixture_id": fid, "team_id": tid, "player_id": pid,
                    "player_name": pl.get("name"), "position": g.get("position"),
                    "starter": starters.get((tid, pid), not g.get("substitute", True)),
                    "minutes": _num(g.get("minutes")), "rating": _num(g.get("rating")),
                    "goals": _num((st.get("goals") or {}).get("total")),
                    "assists": _num((st.get("goals") or {}).get("assists")),
                    "shots": _num((st.get("shots") or {}).get("total")),
                    "shots_on": _num((st.get("shots") or {}).get("on")),
                    "key_passes": _num((st.get("passes") or {}).get("key")),
                    "yellow": _num((st.get("cards") or {}).get("yellow")),
                    "red": _num((st.get("cards") or {}).get("red")),
                })
    fx_df = pd.DataFrame(fixtures)
    app_df = pd.DataFrame(apps)
    if not app_df.empty:
        for c in ("goals", "assists", "shots", "shots_on", "key_passes", "yellow", "red", "minutes"):
            app_df[c] = app_df[c].fillna(0.0)
    return fx_df, app_df


def _injury_records(path: Path) -> list[dict]:
    """Accept both the stored list and a raw API envelope; skip anything malformed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("response", [])
    if not isinstance(data, list):
        data = []
    good = [r for r in data if isinstance(r, dict)]
    if len(good) < len(data):
        sample = next((r for r in data if not isinstance(r, dict)), None)
        log.warning("%s: skipped %d malformed injury records (example: %r)",
                    path.name, len(data) - len(good), str(sample)[:120])
    return good


def parse_injuries(store: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((store / "injuries").glob("*.json")):
        league = path.stem.rsplit("_", 1)[0]
        for r in _injury_records(path):
            pl, fx = r.get("player") or {}, r.get("fixture") or {}
            rows.append({
                "league": league, "fixture_id": fx.get("id"),
                "date": pd.to_datetime(fx.get("date"), utc=True).tz_convert(None)
                if fx.get("date") else pd.NaT,
                "team_id": (r.get("team") or {}).get("id"), "player_id": pl.get("id"),
                "player_name": pl.get("name"), "type": pl.get("type"), "reason": pl.get("reason"),
            })
    df = pd.DataFrame(rows, columns=["league", "fixture_id", "date", "team_id", "player_id",
                                     "player_name", "type", "reason"])
    return df.drop_duplicates(["fixture_id", "player_id"])
