"""Synthetic API-Football JSON (same shape as the real API) for tests and the offline demo.

Built on top of football-data style matches, but with different team names, so the
automatic team-mapping step is exercised exactly as with real data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

POS = ["G"] + ["D"] * 8 + ["M"] * 9 + ["F"] * 6


def generate_api_store(played: pd.DataFrame, store: Path, leagues: list[str],
                       seasons: list[int], seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    m = played[played.league.isin(leagues) & played.season.isin(seasons)].sort_values("date")
    teams = sorted(set(m.home_team) | set(m.away_team))
    team_id = {t: 1000 + i for i, t in enumerate(teams)}
    api_name = {t: t.split(" ", 1)[1].replace("Club", "FC") for t in teams}   # 'E0 Club 03' -> 'FC 03'
    squad = {t: [(team_id[t] * 100 + k, f"P{team_id[t]}-{k}", POS[k], rng.gamma(2.0, 0.5))
                 for k in range(24)] for t in teams}

    detail = store / "fixture_detail"
    detail.mkdir(parents=True, exist_ok=True)
    injuries = {}
    for fid, r in enumerate(m.itertuples(), start=500000):
        lineups, players = [], []
        for side, team, gf in (("home", r.home_team, r.home_goals), ("away", r.away_team, r.away_goals)):
            sq = squad[team]
            injured = [p for p in sq if rng.random() < 0.06]
            avail = [p for p in sq if p not in injured]
            gk = [p for p in avail if p[2] == "G"][:1] or [sq[0]]
            outfield = sorted([p for p in avail if p[2] != "G"], key=lambda p: -p[3] - rng.random())
            xi = gk + outfield[:10]
            subs = outfield[10:17]
            lineups.append({"team": {"id": team_id[team]}, "formation": "4-3-3",
                            "startXI": [{"player": {"id": p[0], "name": p[1], "pos": p[2]}} for p in xi],
                            "substitutes": [{"player": {"id": p[0], "name": p[1], "pos": p[2]}} for p in subs]})
            scorers = rng.choice(len(xi) - 1, size=int(gf), replace=True) + 1 if gf else []
            plist = []
            for k, p in enumerate(xi + subs[:3]):
                start = k < 11
                plist.append({"player": {"id": p[0], "name": p[1]}, "statistics": [{
                    "games": {"minutes": 90 if start else int(rng.integers(5, 30)),
                              "rating": f"{6.3 + 0.4 * p[3] + rng.normal(0, 0.4):.1f}",
                              "position": p[2], "substitute": not start},
                    "goals": {"total": int(np.sum(np.array(scorers) == k)) if start else 0, "assists": 0},
                    "shots": {"total": int(rng.poisson(1)), "on": int(rng.poisson(0.4))},
                    "passes": {"key": int(rng.poisson(0.8))}, "cards": {"yellow": 0, "red": 0}}]})
            players.append({"team": {"id": team_id[team]}, "players": plist})
            for p in injured:
                injuries.setdefault((r.league, r.season), []).append({
                    "player": {"id": p[0], "name": p[1], "type": "Missing Fixture", "reason": "Knock"},
                    "team": {"id": team_id[team]}, "fixture": {"id": fid, "date": f"{r.date:%Y-%m-%d}T15:00:00+00:00"}})
        d = {"fixture": {"id": fid, "date": f"{r.date:%Y-%m-%d}T15:00:00+00:00", "status": {"short": "FT"}},
             "league": {"id": 39, "season": int(r.season)},
             "teams": {"home": {"id": team_id[r.home_team], "name": api_name[r.home_team]},
                       "away": {"id": team_id[r.away_team], "name": api_name[r.away_team]}},
             "goals": {"home": int(r.home_goals), "away": int(r.away_goals)},
             "lineups": lineups, "players": players, "_league_code": r.league}
        (detail / f"{fid}.json").write_text(json.dumps(d))
    (store / "injuries").mkdir(exist_ok=True)
    for (lg, s), rows in injuries.items():
        (store / "injuries" / f"{lg}_{s}.json").write_text(json.dumps(rows))
