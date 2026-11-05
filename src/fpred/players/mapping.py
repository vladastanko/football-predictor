"""Link API-Football fixtures to football-data.co.uk matches.

The two sources name teams differently ("Man United" vs "Manchester United",
"Nott'm Forest" vs "Nottingham Forest"). Instead of a hand-written dictionary,
the team map is learned from the data:

1. Candidate pairs: same league, kick-off within one day, identical final score.
2. Every candidate pair is a vote "API team id X = football-data name Y".
3. Each API team takes the name with most votes, if it clearly beats the runner-up
   (other matches played the same day with the same score add some noise votes).
4. Matches are then joined on league + date (+-1 day) + mapped team names.

Everything that cannot be matched is reported, never silently dropped.
"""
from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd


def learn_team_map(api_fx: pd.DataFrame, fd: pd.DataFrame,
                   min_votes: int = 3, min_margin: float = 3.0) -> pd.DataFrame:
    a = api_fx.dropna(subset=["home_goals", "away_goals"]).copy()
    a["day"] = a.date.dt.normalize()
    f = fd[["league", "date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    cands = []
    for shift in (-1, 0, 1):
        f2 = f.assign(day=f.date + pd.Timedelta(days=shift))
        cands.append(a.merge(f2, on=["league", "day", "home_goals", "away_goals"]))
    c = pd.concat(cands, ignore_index=True)
    votes = pd.concat([
        c[["league", "api_home_id", "api_home_name", "home_team"]].set_axis(
            ["league", "api_id", "api_name", "fd_name"], axis=1),
        c[["league", "api_away_id", "api_away_name", "away_team"]].set_axis(
            ["league", "api_id", "api_name", "fd_name"], axis=1),
    ])
    counts = votes.groupby(["league", "api_id", "api_name", "fd_name"]).size().reset_index(name="votes")
    counts["share"] = counts.votes / counts.groupby(["league", "api_id"]).votes.transform("sum")
    counts = counts.sort_values("votes", ascending=False)
    counts["rank"] = counts.groupby(["league", "api_id"]).cumcount()
    best = counts[counts["rank"] == 0].drop(columns="rank")
    second = counts[counts["rank"] == 1][["league", "api_id", "votes"]] \
        .rename(columns={"votes": "runner_up_votes"})
    best = best.merge(second, on=["league", "api_id"], how="left").fillna({"runner_up_votes": 0})
    best["accepted"] = (best.votes >= min_votes) & \
        (best.votes >= min_margin * best.runner_up_votes)
    # one football-data name must not be claimed by two API teams
    dup = best[best.accepted].duplicated(["league", "fd_name"], keep=False)
    best.loc[dup[dup].index, "accepted"] = False
    best["name_similarity"] = [round(SequenceMatcher(None, x.lower(), y.lower()).ratio(), 2)
                               for x, y in zip(best.api_name, best.fd_name)]
    return best.sort_values(["league", "fd_name"]).reset_index(drop=True)


def link_fixtures(api_fx: pd.DataFrame, fd: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Return api_fx with a match_id column (NaN where no match was found)."""
    tm = team_map[team_map.accepted].set_index(["league", "api_id"]).fd_name
    a = api_fx.copy()
    a["home_team"] = [tm.get((l, i)) for l, i in zip(a.league, a.api_home_id)]
    a["away_team"] = [tm.get((l, i)) for l, i in zip(a.league, a.api_away_id)]
    a["day"] = a.date.dt.normalize()
    f = fd[["match_id", "league", "date", "home_team", "away_team"]]
    linked = []
    for shift in (0, -1, 1):          # exact day first
        f2 = f.assign(day=f.date + pd.Timedelta(days=shift)).drop(columns="date")
        linked.append(a.merge(f2, on=["league", "day", "home_team", "away_team"], how="inner"))
    m = pd.concat(linked).drop_duplicates("fixture_id")[["fixture_id", "match_id"]]
    return api_fx.merge(m, on="fixture_id", how="left")


def mapping_report(team_map: pd.DataFrame, linked: pd.DataFrame) -> str:
    ok = linked.match_id.notna()
    lines = ["# Player data: source linking and coverage", "",
             f"- API fixtures: {len(linked):,}; linked to football-data matches: "
             f"{ok.sum():,} ({ok.mean() * 100:.1f}%)",
             f"- Teams mapped automatically: {team_map.accepted.sum()} of {len(team_map)}", ""]
    rej = team_map[~team_map.accepted]
    if len(rej):
        lines += ["## Teams that could not be mapped (need manual review)", "",
                  rej[["league", "api_name", "fd_name", "votes", "runner_up_votes"]].to_markdown(index=False), ""]
    low = team_map[team_map.accepted & (team_map.name_similarity < 0.5)]
    if len(low):
        lines += ["## Mapped teams with very different names (spot-check these)", "",
                  low[["league", "api_name", "fd_name", "votes", "runner_up_votes"]].to_markdown(index=False), ""]
    cov = linked.groupby(["league", "season"]).agg(
        fixtures=("fixture_id", "size"), linked=("match_id", lambda s: s.notna().sum()),
        with_lineups=("has_lineups", "sum"), with_player_stats=("has_players", "sum")).reset_index()
    lines += ["## Coverage by league and season", "", cov.to_markdown(index=False), ""]
    if (~ok).any():
        lines += ["## Unlinked fixtures (first 15)", "",
                  linked[~ok][["league", "date", "api_home_name", "api_away_name", "home_goals",
                               "away_goals"]].head(15).to_markdown(index=False)]
    lines += ["", "## Full team map", "",
              team_map[["league", "api_name", "fd_name", "votes", "runner_up_votes", "accepted"]]
              .to_markdown(index=False)]
    return "\n".join(lines)
