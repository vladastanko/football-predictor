"""Synthetic data in the exact football-data.co.uk CSV format.

Used by the test suite and by `fpred demo`, so the whole pipeline can run offline.
Team strengths drift over time, three clubs are relegated each season, goals are
Poisson, and bookmaker odds are derived from the true probabilities plus noise
and a margin. A few deliberate data errors can be injected to exercise the
data-quality checks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .config import season_code


def _true_probs(lh: float, la: float, g: int = 10) -> tuple[float, float, float]:
    ph, pa = poisson.pmf(np.arange(g), lh), poisson.pmf(np.arange(g), la)
    grid = np.outer(ph, pa)
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()


def _round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    t = list(teams)
    n = len(t)
    rounds = []
    for r in range(n - 1):
        pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(a, b) if r % 2 == 0 else (b, a) for a, b in pairs])
        t = [t[0]] + [t[-1]] + t[1:-1]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def generate(raw_dir: Path, leagues: list[str], first: int, last: int,
             rounds_played_current: int = 8, n_teams: int = 18,
             inject_errors: bool = False, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    fixtures = []
    for li, league in enumerate(leagues):
        pool = [f"{league} Club {i:02d}" for i in range(60)]
        teams = pool[:n_teams]
        nxt = n_teams
        strength = {t: rng.normal(0, 0.28) for t in pool}
        for year in range(first, last + 1):
            schedule = _round_robin(teams)
            start = pd.Timestamp(year, 8, 10) + pd.Timedelta(days=li % 3)
            rows = []
            for r, rnd in enumerate(schedule):
                date = start + pd.Timedelta(days=7 * r + (3 if r >= 17 else 0))
                for home, away in rnd:
                    lh = np.exp(0.15 + 0.25 + strength[home] - 0.8 * strength[away])
                    la = np.exp(0.15 + strength[away] - 0.8 * strength[home])
                    ph, pd_, pa = _true_probs(lh, la)
                    noisy = np.array([ph, pd_, pa]) * np.exp(rng.normal(0, 0.04, 3))
                    noisy = np.clip(noisy / noisy.sum(), 0.02, 0.94)
                    noisy /= noisy.sum()
                    odds = np.round(1 / (noisy * 1.04), 2)
                    if year == last and r >= rounds_played_current:
                        if r < rounds_played_current + 2:
                            fixtures.append({"Div": league, "Date": date.strftime("%d/%m/%Y"),
                                             "Time": "18:00", "HomeTeam": home, "AwayTeam": away,
                                             "PSH": odds[0], "PSD": odds[1], "PSA": odds[2]})
                        continue
                    hg, ag = rng.poisson(lh), rng.poisson(la)
                    hst, ast = hg + rng.poisson(3), ag + rng.poisson(2.5)
                    rows.append({
                        "Div": league, "Date": date.strftime("%d/%m/%Y"), "Time": "18:00",
                        "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag,
                        "FTR": "H" if hg > ag else "A" if hg < ag else "D",
                        "HTHG": rng.binomial(hg, 0.45), "HTAG": rng.binomial(ag, 0.45),
                        "HS": hst + rng.poisson(8), "AS": ast + rng.poisson(6),
                        "HST": hst, "AST": ast, "HC": rng.poisson(5), "AC": rng.poisson(4),
                        "HF": rng.poisson(12), "AF": rng.poisson(12),
                        "HY": rng.poisson(1.8), "AY": rng.poisson(2.0),
                        "HR": rng.binomial(1, 0.05), "AR": rng.binomial(1, 0.06),
                        "PSCH": odds[0], "PSCD": odds[1], "PSCA": odds[2],
                    })
            df = pd.DataFrame(rows)
            if inject_errors and league == leagues[0] and year == first + 1:
                df.loc[3, "FTR"] = "A" if df.loc[3, "FTR"] != "A" else "H"   # wrong result letter
                df = pd.concat([df, df.iloc[[10]]], ignore_index=True)       # duplicate row
                df.loc[20, "HST"] = int(df.loc[20, "HS"]) + 2                # SOT > shots
            dest = raw_dir / league / f"{season_code(year)}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest, index=False)

            # strengths drift; bottom three relegated, three new clubs promoted
            for t in pool:
                strength[t] = 0.9 * strength[t] + rng.normal(0, 0.12)
            pts = {t: 0 for t in teams}
            for r in rows:
                pts[r["HomeTeam"]] += 3 if r["FTR"] == "H" else 1 if r["FTR"] == "D" else 0
                pts[r["AwayTeam"]] += 3 if r["FTR"] == "A" else 1 if r["FTR"] == "D" else 0
            if year < last:
                down = sorted(teams, key=pts.get)[:3]
                new = pool[nxt:nxt + 3]
                nxt += 3
                for t in new:
                    strength[t] = rng.normal(-0.25, 0.15)
                teams = [t for t in teams if t not in down] + new
    pd.DataFrame(fixtures).to_csv(raw_dir / "fixtures.csv", index=False)
