"""Pre-match features. Every feature uses only information available before kick-off.

Two families:
1. Elo ratings per league, updated match by match (goal-difference weighted).
2. Rolling form over each team's last N matches: goals, shots on target, points, rest days.

Leakage guard: rolling statistics are shifted by one match, and Elo values are stored
before the match result is applied. Tests in tests/test_features.py enforce this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORM_STATS = ["goals_for", "goals_against", "sot_for", "sot_against",
              "shots_for", "shots_against", "points"]


# --------------------------------------------------------------------------- Elo
def _goal_multiplier(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def elo_expected(r_home: float, r_away: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home + home_adv - r_away) / 400))


def compute_elo(played: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return (pre-match Elo per match_id, final ratings per (league, team))."""
    e = cfg["elo"]
    rows, final = [], {}
    for league, df in played.sort_values("date").groupby("league", sort=False):
        ratings: dict[str, float] = {}
        prev_season_teams: set[str] = set()
        season = None
        for r in df.itertuples(index=False):
            if r.season != season:                         # season boundary
                if ratings:
                    mean = np.mean(list(ratings.values()))
                    for t in ratings:
                        ratings[t] += e["season_regression"] * (mean - ratings[t])
                prev_season_teams = set(df.loc[df.season == season, "home_team"]) if season else set()
                season = r.season
            for team in (r.home_team, r.away_team):
                if team not in ratings:
                    base = np.mean(list(ratings.values())) if ratings else e["initial_rating"]
                    # newcomers (promoted) start below average, except in the very first season
                    offset = e["promoted_rating_offset"] if prev_season_teams else 0.0
                    ratings[team] = base + offset
            rh, ra = ratings[r.home_team], ratings[r.away_team]
            rows.append((r.match_id, rh, ra))
            exp_h = elo_expected(rh, ra, e["home_advantage"])
            gd = r.home_goals - r.away_goals
            actual = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            delta = e["k_factor"] * _goal_multiplier(gd) * (actual - exp_h)
            ratings[r.home_team] = rh + delta
            ratings[r.away_team] = ra - delta
        final.update({(league, t): v for t, v in ratings.items()})
    elo = pd.DataFrame(rows, columns=["match_id", "elo_home", "elo_away"])
    return elo, final


# ------------------------------------------------------------------ rolling form
def _team_long(m: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home and away perspectives)."""
    def side(us: str, them: str, is_home: int) -> pd.DataFrame:
        gf, ga = m[f"{us}_goals"], m[f"{them}_goals"]
        pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0)).astype(float)
        pts[gf.isna().to_numpy()] = np.nan
        return pd.DataFrame({
            "match_id": m["match_id"], "league": m["league"], "date": m["date"],
            "team": m[f"{us}_team"], "is_home": is_home,
            "goals_for": gf, "goals_against": ga,
            "sot_for": m[f"{us}_sot"], "sot_against": m[f"{them}_sot"],
            "shots_for": m[f"{us}_shots"], "shots_against": m[f"{them}_shots"],
            "points": pts,
        })
    long = pd.concat([side("home", "away", 1), side("away", "home", 0)], ignore_index=True)
    for c in FORM_STATS:
        long[c] = pd.to_numeric(long[c], errors="coerce").astype(float)
    return long.sort_values(["team", "date", "is_home"]).reset_index(drop=True)


def rolling_form(m: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    long = _team_long(m)
    g = long.groupby(["league", "team"], sort=False)
    for c in FORM_STATS:
        long[f"form_{c}"] = g[c].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    long["rest_days"] = g["date"].diff().dt.days.clip(upper=30)
    long["matches_in_league"] = g.cumcount()
    feat_cols = [f"form_{c}" for c in FORM_STATS] + ["rest_days", "matches_in_league"]
    home = long[long.is_home == 1].set_index("match_id")[feat_cols].add_prefix("h_")
    away = long[long.is_home == 0].set_index("match_id")[feat_cols].add_prefix("a_")
    return home.join(away)


# ------------------------------------------------------------------ public API
def build_features(played: pd.DataFrame, cfg: dict,
                   fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature table for played matches and (optionally) upcoming fixtures."""
    f = cfg["features"]
    elo, final = compute_elo(played, cfg)

    frames = [played.assign(is_fixture=False)]
    if fixtures is not None and len(fixtures):
        frames.append(fixtures.assign(is_fixture=True))
    allm = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    out = allm.merge(elo, on="match_id", how="left")
    if fixtures is not None and len(fixtures):
        fx = out["is_fixture"]
        e = cfg["elo"]
        for side in ("home", "away"):
            out.loc[fx, f"elo_{side}"] = [
                final.get((lg, t), e["initial_rating"] + e["promoted_rating_offset"])
                for lg, t in zip(out.loc[fx, "league"], out.loc[fx, f"{side}_team"])]

    out = out.join(rolling_form(allm, f["rolling_window"], f["min_matches_for_form"]),
                   on="match_id")
    out["elo_diff"] = out["elo_home"] + cfg["elo"]["home_advantage"] - out["elo_away"]
    out["elo_p_home"] = 1 / (1 + 10 ** (-out["elo_diff"] / 400))
    for c in ("goals_for", "goals_against", "sot_for", "sot_against", "points"):
        out[f"diff_form_{c}"] = out[f"h_form_{c}"] - out[f"a_form_{c}"]
    out["rest_diff"] = out["h_rest_days"] - out["a_rest_days"]
    return out


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns fed to the ML model (no odds, no outcomes, no identifiers).

    Dixon-Coles columns are out-of-sample predictions (walk-forward), so using
    them as inputs is a leak-free form of stacking."""
    prefixes = ("elo_", "dc_", "h_form_", "a_form_", "diff_form_", "h_rest", "a_rest", "rest_diff",
                "h_matches", "a_matches")
    return [c for c in df.columns if c.startswith(prefixes)]
