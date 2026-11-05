"""Data quality checks on the clean match table.

Every check returns a row: name, severity, number of failing rows, and examples.
Severity:
    error   -> data is wrong and must be fixed or excluded before modelling
    warning -> suspicious, needs a human look (often explained by the league format)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    severity: str
    failed: int
    description: str
    examples: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.failed == 0


def _examples(df: pd.DataFrame, n: int = 5) -> list[str]:
    cols = [c for c in ("league", "season", "date", "home_team", "away_team") if c in df]
    return [" | ".join(str(v)[:10] if k == "date" else str(v) for k, v in row.items())
            for row in df[cols].head(n).to_dict("records")]


def run_checks(m: pd.DataFrame, current_season: int) -> list[CheckResult]:
    res: list[CheckResult] = []

    def add(name, severity, bad, description):
        if not isinstance(bad, pd.DataFrame):
            # nullable comparisons can contain <NA> (e.g. unplayed matches): treat as "not failing"
            mask = pd.Series(bad, index=m.index).astype("boolean").fillna(False).astype(bool)
            bad = m.loc[mask]
        res.append(CheckResult(name, severity, len(bad), description, _examples(bad)))

    played = m["home_goals"].notna() & m["away_goals"].notna()
    hg, ag = m["home_goals"], m["away_goals"]

    # --- structural -----------------------------------------------------------
    add("duplicate_match", "error", m["match_id"].duplicated(keep=False),
        "Same league, date and teams appear more than once")
    add("missing_date", "error", m["date"].isna(), "Match without a parseable date")
    add("same_team_both_sides", "error", m["home_team"] == m["away_team"],
        "Home and away team are identical")
    long = pd.concat([m[["league", "season", "date", "home_team"]].rename(columns={"home_team": "team"}),
                      m[["league", "season", "date", "away_team"]].rename(columns={"away_team": "team"})])
    dup_keys = pd.MultiIndex.from_frame(
        long[long.duplicated(["league", "date", "team"], keep=False)][["league", "date", "team"]])
    home_key = pd.MultiIndex.from_frame(m[["league", "date", "home_team"]])
    away_key = pd.MultiIndex.from_frame(m[["league", "date", "away_team"]])
    add("team_twice_same_day", "error", home_key.isin(dup_keys) | away_key.isin(dup_keys),
        "A team is listed in two matches on the same day")

    # --- completeness ---------------------------------------------------------
    add("missing_result_past_season", "error", (~played) & (m["season"] < current_season),
        "Completed season contains matches without a final score")

    # --- internal consistency -------------------------------------------------
    from .clean import result_from_score
    calc = result_from_score(hg, ag)
    add("result_vs_score", "error", played & m["result"].notna() & (m["result"] != calc),
        "Full-time result letter (FTR) disagrees with the score")
    for side in ("home", "away"):
        ht, ft = m[f"ht_{side}_goals"], m[f"{side}_goals"]
        add(f"ht_gt_ft_{side}", "error", ht.notna() & ft.notna() & (ht > ft),
            f"Half-time {side} goals greater than full-time goals")
        sot, sh = m[f"{side}_sot"], m[f"{side}_shots"]
        add(f"sot_gt_shots_{side}", "error", sot.notna() & sh.notna() & (sot > sh),
            f"{side} shots on target greater than total shots")
        add(f"goals_gt_sot_{side}", "warning", sot.notna() & (m[f"{side}_goals"] > sot),
            f"{side} goals exceed shots on target (possible with own goals; check if frequent)")
        add(f"red_cards_implausible_{side}", "error", m[f"{side}_red"].fillna(0) > 5,
            "More than 5 red cards for one team (match would be abandoned)")

    # --- plausibility / outliers ----------------------------------------------
    add("high_scoring_outlier", "warning", (hg + ag) >= 10,
        "Ten or more goals in a match: verify against an official source")
    neg = m[[c for c in m.columns if c.startswith(("home_", "away_", "ht_")) and
             pd.api.types.is_numeric_dtype(m[c])]].lt(0).any(axis=1)
    add("negative_values", "error", neg, "Negative counts in match statistics")

    # --- season structure -----------------------------------------------------
    done = m[played & (m["season"] < current_season)]
    home_n = done.groupby(["league", "season", "home_team"]).size()
    away_n = done.groupby(["league", "season", "away_team"]).size()
    bal = pd.concat([home_n.rename("home"), away_n.rename("away")], axis=1).fillna(0)
    unbalanced = bal[(bal.home - bal.away).abs() > 1].reset_index()
    unbalanced.columns = ["league", "season", "team", "home", "away"]
    res.append(CheckResult(
        "home_away_balance", "warning", len(unbalanced),
        "Team's home and away match counts differ by >1 in a completed season "
        "(expected only in leagues with play-off rounds)",
        [f"{r.league} {r.season} {r.team}: {int(r.home)}H/{int(r.away)}A"
         for r in unbalanced.head(5).itertuples()]))

    rare = (long.groupby(["league", "season", "team"]).size().reset_index(name="n")
            .query("n < 5"))
    res.append(CheckResult(
        "rare_team_name", "warning", len(rare),
        "Team appears in fewer than 5 matches in a season (possible typo or renamed club)",
        [f"{r.league} {r.season} {r.team} ({r.n})" for r in rare.head(5).itertuples()]))

    # --- market data ----------------------------------------------------------
    has_mkt = m["mkt_p_home"].notna()
    psum = m[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].sum(axis=1)
    add("market_probs_not_normalised", "error", has_mkt & ((psum - 1).abs() > 1e-6),
        "Implied probabilities do not sum to 1 after removing the margin")
    add("market_overround_out_of_range", "warning",
        has_mkt & ((m["mkt_overround"] < 0) | (m["mkt_overround"] > 0.20)),
        "Bookmaker margin outside 0-20% (likely a data entry error in the odds)")
    add("missing_market_odds", "warning", played & ~has_mkt,
        "Played match without usable bookmaker odds (excluded from market benchmark)")
    return res


def coverage_table(m: pd.DataFrame) -> pd.DataFrame:
    """Matches, teams and goal averages per league and season."""
    p = m.dropna(subset=["home_goals"])
    g = p.groupby(["league", "season"])
    return pd.DataFrame({
        "matches": g.size(),
        "teams": g["home_team"].nunique(),
        "goals_per_match": g.apply(lambda d: (d.home_goals + d.away_goals).mean()).round(2),
        "home_win_pct": g.apply(lambda d: (d.home_goals > d.away_goals).mean() * 100).round(1),
        "draw_pct": g.apply(lambda d: (d.home_goals == d.away_goals).mean() * 100).round(1),
        "odds_coverage_pct": g["mkt_p_home"].apply(lambda s: s.notna().mean() * 100).round(1),
    }).reset_index()


def to_markdown(results: list[CheckResult], coverage: pd.DataFrame) -> str:
    lines = ["# Data quality report", "",
             "| Check | Severity | Failed rows | Status | Description |",
             "|---|---|---:|---|---|"]
    for r in results:
        status = "PASS" if r.passed else ("FAIL" if r.severity == "error" else "REVIEW")
        lines.append(f"| `{r.name}` | {r.severity} | {r.failed} | {status} | {r.description} |")
    flagged = [r for r in results if not r.passed]
    if flagged:
        lines += ["", "## Examples of flagged rows", ""]
        for r in flagged:
            lines.append(f"**{r.name}**")
            lines += [f"- {e}" for e in r.examples] + [""]
    lines += ["", "## Coverage by league and season", "", coverage.to_markdown(index=False)]
    return "\n".join(lines)
