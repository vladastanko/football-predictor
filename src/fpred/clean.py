"""Parse raw football-data.co.uk CSVs into one clean, typed match table.

Output schema (one row per match):
    match_id, league, season, date, home_team, away_team,
    home_goals, away_goals, result (H/D/A), ht_home_goals, ht_away_goals,
    home_shots, away_shots, home_sot, away_sot, home_corners, away_corners,
    home_fouls, away_fouls, home_yellow, away_yellow, home_red, away_red,
    mkt_p_home, mkt_p_draw, mkt_p_away, mkt_source, mkt_overround
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RENAME = {
    "Div": "league", "Date": "date", "Time": "kickoff",
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HTHG": "ht_home_goals", "HTAG": "ht_away_goals",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners", "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow", "AY": "away_yellow", "HR": "home_red", "AR": "away_red",
}
INT_COLS = [c for c in RENAME.values() if c not in ("league", "date", "kickoff",
                                                    "home_team", "away_team", "result")]

# Bookmaker odds in order of preference. Pinnacle closing odds are the sharpest
# public benchmark; older seasons only have averages or single bookmakers.
ODDS_PRIORITY = [
    ("PSC", "pinnacle_closing"), ("PS", "pinnacle"),
    ("AvgC", "market_avg_closing"), ("Avg", "market_avg"),
    ("BbAv", "betbrain_avg"), ("B365", "bet365"),
]


def _read_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), index_col=False, on_bad_lines="skip", low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df.dropna(how="all")


def _parse_dates(s: pd.Series) -> pd.Series:
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return long.fillna(short)


def _market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds to probabilities and remove the bookmaker margin."""
    out = pd.DataFrame(index=df.index, columns=["mkt_p_home", "mkt_p_draw", "mkt_p_away",
                                                "mkt_source", "mkt_overround"], dtype=object)
    remaining = pd.Series(True, index=df.index)
    for prefix, name in ODDS_PRIORITY:
        cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
        if not all(c in df.columns for c in cols):
            continue
        odds = df[cols].apply(pd.to_numeric, errors="coerce")
        valid = remaining & odds.notna().all(axis=1) & (odds > 1.0).all(axis=1)
        if not valid.any():
            continue
        inv = 1.0 / odds[valid]
        total = inv.sum(axis=1)
        probs = inv.div(total, axis=0)
        out.loc[valid, ["mkt_p_home", "mkt_p_draw", "mkt_p_away"]] = probs.values
        out.loc[valid, "mkt_source"] = name
        out.loc[valid, "mkt_overround"] = (total - 1.0).values
        remaining &= ~valid
    for c in ("mkt_p_home", "mkt_p_draw", "mkt_p_away", "mkt_overround"):
        out[c] = pd.to_numeric(out[c])
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def make_match_id(league: str, date: pd.Timestamp, home: str, away: str) -> str:
    return f"{league}_{date:%Y%m%d}_{_slug(home)}_{_slug(away)}"


def standardize(df: pd.DataFrame, season: int | None = None,
                league_fallback: str | None = None) -> pd.DataFrame:
    """Rename, type-cast and enrich one raw season (or the fixtures file)."""
    out = df.rename(columns=RENAME).copy()
    if "league" not in out.columns and league_fallback:
        out["league"] = league_fallback
    out = out.dropna(subset=["home_team", "away_team", "date"])
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["date"] = _parse_dates(out["date"].astype(str).str.strip())
    for c in INT_COLS:
        # float64 with NaN for every count column: some leagues/seasons lack columns
        # (e.g. shots), and a uniform dtype keeps comparisons safe after concatenation
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float) if c in out else np.nan
    if "result" not in out:
        out["result"] = np.nan
    if season is not None:
        out["season"] = season
    out = pd.concat([out, _market_probs(df.loc[out.index])], axis=1)
    out["match_id"] = [make_match_id(l, d, h, a) if pd.notna(d) else None for l, d, h, a in
                       zip(out["league"], out["date"], out["home_team"], out["away_team"])]
    keep = (["match_id", "league", "season", "date", "home_team", "away_team"] +
            [c for c in INT_COLS] + ["result", "mkt_p_home", "mkt_p_draw", "mkt_p_away",
                                     "mkt_source", "mkt_overround"])
    return out[[c for c in keep if c in out.columns]]


def build_matches(raw_dir: Path, leagues: list[str]) -> pd.DataFrame:
    """Read all cached raw files into a single match table."""
    frames = []
    for league in leagues:
        for path in sorted((raw_dir / league).glob("*.csv")):
            code = path.stem                      # e.g. '2425'
            season = 2000 + int(code[:2])
            try:
                df = standardize(_read_csv(path), season=season, league_fallback=league)
            except Exception as exc:              # keep going, but make it visible
                log.error("Could not parse %s: %s", path, exc)
                continue
            df["league"] = league                 # trust the folder over the file
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No raw files found under {raw_dir}. Run `fpred download` first.")
    matches = pd.concat(frames, ignore_index=True)
    matches[INT_COLS] = matches[INT_COLS].astype(float)
    matches = matches.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    return matches


def result_from_score(home_goals: pd.Series, away_goals: pd.Series) -> np.ndarray:
    """'H'/'D'/'A' from the score; works with nullable integers (missing scores give 'D'
    here, so callers must mask unplayed matches themselves)."""
    h = pd.to_numeric(home_goals, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    a = pd.to_numeric(away_goals, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return np.select([h > a, h < a], ["H", "A"], "D")


def played(matches: pd.DataFrame) -> pd.DataFrame:
    """Matches with a final result."""
    m = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    m["home_goals"] = m["home_goals"].astype(int)
    m["away_goals"] = m["away_goals"].astype(int)
    m["result"] = result_from_score(m.home_goals, m.away_goals)
    return m


def load_fixtures(path: Path, leagues: list[str]) -> pd.DataFrame:
    df = _read_csv(path)
    fx = standardize(df)
    return fx[fx["league"].isin(leagues)].reset_index(drop=True)
