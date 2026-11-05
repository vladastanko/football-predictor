"""Player-based team features, using only information available before kick-off.

For every player we track, at each of his team's matches, what he had done in the
team's *previous* matches (shifted, never including the match being predicted):
    min_share   share of available minutes played in the last N team matches
    rating      minutes-weighted average match rating over the last 2N team matches
    ga90        goals + assists per 90 minutes over the last 2N team matches

These are aggregated per team and match into two feature groups:

  Absences (known ~1 day before, from injury/suspension lists)
    inj_count      players listed as missing
    inj_min_share  sum of their minute shares   (1.0 ~ one full-time regular missing)
    inj_ga_share   share of the team's recent goals+assists that is missing
    inj_key        missing players who were regulars (min_share >= 0.7)

  Confirmed line-up (known ~30 minutes before kick-off)
    xi_rating      average recent rating of the starting XI
    xi_min_share   average minute share of the starters (1.0 = usual XI)
    xi_ga90        summed goals+assists per 90 of the starters
    xi_rotation    starters who did not start the team's previous match
    xi_new         starters with no previous appearance for the team
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INJURY_FEATURES = ["inj_count", "inj_min_share", "inj_ga_share", "inj_key"]
LINEUP_FEATURES = ["xi_rating", "xi_min_share", "xi_ga90", "xi_rotation", "xi_new"]


def _team_matches(api_fx: pd.DataFrame) -> pd.DataFrame:
    side = lambda s: api_fx[["fixture_id", "date", f"api_{s}_id"]].rename(  # noqa: E731
        columns={f"api_{s}_id": "team_id"}).assign(side=s)
    tm = pd.concat([side("home"), side("away")], ignore_index=True)
    tm = tm.sort_values(["team_id", "date"]).reset_index(drop=True)
    tm["team_match_no"] = tm.groupby("team_id").cumcount()
    return tm


def player_form(api_fx: pd.DataFrame, apps: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pre-match form of every player of a team, at every one of the team's matches."""
    tm = _team_matches(api_fx)
    a = apps.merge(tm[["fixture_id", "team_id", "team_match_no"]], on=["fixture_id", "team_id"])
    a["ga"] = a.goals + a.assists
    a["rating_x_min"] = a.rating.fillna(0) * a.minutes
    a["rated_min"] = np.where(a.rating.notna(), a.minutes, 0.0)
    out = []
    for team, g in a.groupby("team_id"):
        n_matches = tm.loc[tm.team_id == team, "team_match_no"].max() + 1
        idx = pd.RangeIndex(n_matches, name="team_match_no")

        def wide(col):
            return g.pivot_table(index="team_match_no", columns="player_id", values=col,
                                 aggfunc="sum").reindex(idx).fillna(0.0)

        mins, ga, rxm, rmin = wide("minutes"), wide("ga"), wide("rating_x_min"), wide("rated_min")
        roll = lambda w, k: w.rolling(k, min_periods=1).sum().shift(1)  # noqa: E731
        played_n = pd.Series(np.minimum(np.arange(n_matches), n), index=idx)
        min_share = roll(mins, n).div(90 * played_n.replace(0, np.nan), axis=0)
        long_min = roll(mins, 2 * n)
        ga90 = roll(ga, 2 * n) / long_min.where(long_min >= 90) * 90
        rating = roll(rxm, 2 * n) / roll(rmin, 2 * n).where(lambda x: x >= 45)
        ever = (mins.cumsum().shift(1) > 0)
        started_prev = g[g.starter].pivot_table(index="team_match_no", columns="player_id",
                                                values="minutes", aggfunc="size")
        started_prev = started_prev.reindex(idx).notna().shift(1, fill_value=False)
        f = pd.concat({"min_share": min_share, "ga90": ga90, "rating": rating,
                       "has_history": ever, "started_prev": started_prev.reindex(
                           columns=mins.columns, fill_value=False)}, axis=1)
        f = f.stack(level="player_id", future_stack=True).reset_index()
        f["team_id"] = team
        out.append(f)
    form = pd.concat(out, ignore_index=True)
    return form.merge(tm[["fixture_id", "team_id", "team_match_no"]], on=["team_id", "team_match_no"])


def team_player_features(api_fx: pd.DataFrame, apps: pd.DataFrame, injuries: pd.DataFrame,
                         n: int = 5) -> pd.DataFrame:
    form = player_form(api_fx, apps, n)
    key = ["fixture_id", "team_id"]

    # team-level recent goal involvement, to express missing G+A as a share
    team_ga = form.assign(ga_recent=form.ga90.fillna(0) * form.min_share.fillna(0)) \
        .groupby(key).ga_recent.sum().rename("team_ga_recent")

    # ---- absences
    miss = injuries[injuries.type.fillna("").str.contains("Missing", case=False)]
    m = miss[["fixture_id", "team_id", "player_id"]].merge(form, on=["fixture_id", "team_id", "player_id"], how="left")
    m["ga_recent"] = m.ga90.fillna(0) * m.min_share.fillna(0)
    inj = m.groupby(key).agg(inj_count=("player_id", "size"),
                             inj_min_share=("min_share", lambda s: s.fillna(0).sum()),
                             inj_ga=("ga_recent", "sum"),
                             inj_key=("min_share", lambda s: (s >= 0.7).sum()))
    inj = inj.join(team_ga)
    inj["inj_ga_share"] = (inj.inj_ga / inj.team_ga_recent.replace(0, np.nan)).fillna(0)
    inj = inj[INJURY_FEATURES]

    # ---- confirmed line-up
    xi = apps.loc[apps.starter, ["fixture_id", "team_id", "player_id"]].merge(form, on=["fixture_id", "team_id", "player_id"], how="left")
    lu = xi.groupby(key).agg(
        xi_rating=("rating", "mean"),
        xi_min_share=("min_share", lambda s: s.fillna(0).mean()),
        xi_ga90=("ga90", lambda s: s.fillna(0).sum()),
        xi_rotation=("started_prev", lambda s: (~s.fillna(False).astype(bool)).sum()),
        xi_new=("has_history", lambda s: (~s.fillna(False).astype(bool)).sum()),
    )

    tm = _team_matches(api_fx)[["fixture_id", "team_id", "side", "team_match_no"]]
    t = tm.join(inj, on=key).join(lu, on=key)
    t[INJURY_FEATURES] = t[INJURY_FEATURES].fillna(0)      # no record = nobody missing
    t.loc[t.team_match_no == 0, LINEUP_FEATURES] = np.nan  # no history yet for the team

    cols = INJURY_FEATURES + LINEUP_FEATURES
    h = t[t.side == "home"].set_index("fixture_id")[cols].add_prefix("pl_h_")
    a = t[t.side == "away"].set_index("fixture_id")[cols].add_prefix("pl_a_")
    out = h.join(a)
    for c in cols:
        out[f"pl_diff_{c}"] = out[f"pl_h_{c}"] - out[f"pl_a_{c}"]
    return out.reset_index()


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    inj = [c for c in df.columns if c.startswith("pl_") and any(f in c for f in INJURY_FEATURES)]
    lu = [c for c in df.columns if c.startswith("pl_") and any(f in c for f in LINEUP_FEATURES)]
    return {"injuries": inj, "lineup": lu}
