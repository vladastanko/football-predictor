"""End-to-end pipeline steps used by the CLI."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean, validate
from .config import Paths, season_label
from .evaluate import _probs, _put, summary, walk_forward
from .features import build_features, model_feature_columns
from .models import GBMModel, dixon_coles_walk_forward

log = logging.getLogger(__name__)


def fix_known_issues(m: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Resolve error-level findings in a transparent, logged way."""
    notes = []
    dup = m.duplicated("match_id")
    if dup.any():
        notes.append(f"Removed {dup.sum()} duplicate match rows (kept first occurrence).")
        m = m[~dup]
    for side in ("home", "away"):
        bad = m[f"{side}_sot"] > m[f"{side}_shots"]
        bad = bad.fillna(False)
        if bad.any():
            m.loc[bad, [f"{side}_sot", f"{side}_shots"]] = np.nan
            notes.append(f"Set {side} shots/shots-on-target to missing in {bad.sum()} rows "
                         "where shots on target exceeded total shots.")
    calc = clean.result_from_score(m.home_goals, m.away_goals)
    wrong = (m.home_goals.notna() & m.result.notna() & (m.result != calc)).astype(bool)
    if wrong.any():
        notes.append(f"Recomputed the result letter from the score in {wrong.sum()} rows.")
    return m.reset_index(drop=True), notes


def build(cfg: dict, paths: Paths) -> dict:
    leagues = list(cfg["leagues"])
    current = cfg["seasons"]["last"]

    matches = clean.build_matches(paths.raw, leagues)
    checks = validate.run_checks(matches, current)
    coverage = validate.coverage_table(matches)
    dq_md = validate.to_markdown(checks, coverage)
    matches, notes = fix_known_issues(matches)
    if notes:
        dq_md += "\n\n## Fixes applied before modelling\n\n" + "\n".join(f"- {n}" for n in notes)
    (paths.reports / "data_quality.md").write_text(dq_md, encoding="utf-8")
    errors = [c for c in checks if c.severity == "error" and not c.passed]
    log.info("Data quality: %d checks, %d error checks failed (fixed or excluded)",
             len(checks), len(errors))

    played = clean.played(matches)
    fx_path = paths.raw / "fixtures.csv"
    fixtures = clean.load_fixtures(fx_path, leagues) if fx_path.exists() else None
    if fixtures is not None:
        last_played = played.date.max()
        fixtures = fixtures[fixtures.date > last_played - pd.Timedelta(days=3)]
        fixtures = fixtures[~fixtures.match_id.isin(played.match_id)]
        fixtures["season"] = current

    feat = build_features(played, cfg, fixtures)
    log.info("Fitting Dixon-Coles walk-forward (refit every %s days)...",
             cfg["dixon_coles"]["refit_every_days"])
    dc = dixon_coles_walk_forward(played, cfg, fixtures)
    feat = feat.merge(dc, on="match_id", how="left")

    matches.to_parquet(paths.processed / "matches.parquet", index=False)
    matches.to_csv(paths.processed / "all_matches.csv", index=False)  # one merged file, Excel-friendly
    feat.to_parquet(paths.processed / "features.parquet", index=False)
    summary_lines = [f"- {len(played):,} played matches, {played.league.nunique()} leagues, "
                     f"seasons {season_label(played.season.min())} to {season_label(played.season.max())}",
                     f"- {sum(c.passed for c in checks)}/{len(checks)} checks passed; "
                     f"see `data_quality.md` for details"] + [f"- {n}" for n in notes]
    return {"feat": feat, "dq_summary": "\n".join(summary_lines),
            "n_fixtures": 0 if fixtures is None else len(fixtures)}


def run_evaluation(cfg: dict, paths: Paths, feat: pd.DataFrame, dq_summary: str) -> Path:
    from .report import write_report
    preds, imp = walk_forward(feat, cfg)
    preds.to_parquet(paths.processed / "backtest_predictions.parquet", index=False)
    overall = summary(preds)
    by_league = summary(preds, ["league"])
    by_season = summary(preds, ["season"])
    overall.to_csv(paths.reports / "scores_overall.csv", index=False)
    by_league.to_csv(paths.reports / "scores_by_league.csv", index=False)
    names = {k: v["name"] for k, v in cfg["leagues"].items()}
    return write_report(paths.reports, overall, by_league, by_season, preds, imp, names, dq_summary)


def predict_fixtures(cfg: dict, paths: Paths, feat: pd.DataFrame) -> Path | None:
    """Train on all history and output probabilities for upcoming fixtures."""
    fx = feat[feat.is_fixture].copy()
    if fx.empty:
        log.warning("No upcoming fixtures found for the configured leagues.")
        return None
    hist = feat[~feat.is_fixture & feat.result.notna() & feat.dc_p_home.notna()]
    gbm = GBMModel(cfg["lightgbm"], model_feature_columns(hist)).fit(hist)
    _put(fx, "lightgbm", gbm.predict_proba(fx))
    p = (_probs(fx, "dc_p") + _probs(fx, "lightgbm_p")) / 2
    _put(fx, "model", p)

    names = {k: v["name"] for k, v in cfg["leagues"].items()}
    out = pd.DataFrame({
        "date": fx.date.dt.strftime("%Y-%m-%d"),
        "league": fx.league.map(names),
        "home_team": fx.home_team, "away_team": fx.away_team,
        "p_home": fx.model_p_home.round(3), "p_draw": fx.model_p_draw.round(3),
        "p_away": fx.model_p_away.round(3),
        "fair_odds_home": (1 / fx.model_p_home).round(2),
        "fair_odds_draw": (1 / fx.model_p_draw).round(2),
        "fair_odds_away": (1 / fx.model_p_away).round(2),
        "exp_goals_home": fx.dc_xg_home.round(2), "exp_goals_away": fx.dc_xg_away.round(2),
        "market_p_home": fx.mkt_p_home.round(3), "market_p_draw": fx.mkt_p_draw.round(3),
        "market_p_away": fx.mkt_p_away.round(3),
    }).sort_values(["date", "league", "home_team"])
    csv = paths.reports / "upcoming_predictions.csv"
    out.to_csv(csv, index=False)
    md = ["# Upcoming match probabilities", "",
          "Ensemble of Dixon-Coles and LightGBM, trained on all available history. "
          "Fair odds = 1 / probability (no margin).", "",
          out.drop(columns=[c for c in out if c.startswith("market_")]).to_markdown(index=False)]
    (paths.reports / "upcoming_predictions.md").write_text("\n".join(md), encoding="utf-8")
    return csv
