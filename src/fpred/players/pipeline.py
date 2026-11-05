"""Orchestration for the player-data steps."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, Paths
from . import experiment, mapping, parse
from .features import team_player_features

log = logging.getLogger(__name__)


def api_store(paths: Paths) -> Path:
    return paths.root / "data" / "api"


def collect(cfg: dict, paths: Paths) -> None:
    from .api import ApiFootball, load_api_key
    from .collect import collect as run_collect
    key = load_api_key(paths.root if (paths.root / ".env").exists() else PROJECT_ROOT)
    api = ApiFootball(key, cfg["players"], api_store(paths))
    run_collect(cfg, api, api_store(paths))


def build_and_evaluate(cfg: dict, paths: Paths, feat: pd.DataFrame) -> Path:
    store = api_store(paths)
    api_fx, apps = parse.parse_details(store)
    if api_fx.empty:
        raise FileNotFoundError(f"No player data in {store}. Run `fpred players-collect` first.")
    injuries = parse.parse_injuries(store)
    log.info("Parsed %d fixtures, %d player appearances, %d injury records",
             len(api_fx), len(apps), len(injuries))

    played = feat[~feat.is_fixture & feat.result.notna()]
    team_map = mapping.learn_team_map(api_fx, played)
    linked = mapping.link_fixtures(api_fx, played, team_map)
    cov_md = mapping.mapping_report(team_map, linked)
    log.info("Linked %d of %d API fixtures to football-data matches",
             linked.match_id.notna().sum(), len(linked))

    pl = team_player_features(api_fx, apps, injuries, cfg["players"]["form_matches"])
    pl = pl.merge(linked[["fixture_id", "match_id"]], on="fixture_id").dropna(subset=["match_id"])

    team_map.to_csv(paths.processed / "team_map.csv", index=False)
    apps.to_parquet(paths.processed / "player_appearances.parquet", index=False)
    injuries.to_parquet(paths.processed / "injuries.parquet", index=False)
    pl.to_parquet(paths.processed / "player_features.parquet", index=False)
    (paths.reports / "player_data_quality.md").write_text(cov_md, encoding="utf-8")

    summary, per_match = experiment.run(feat, pl.drop(columns="fixture_id"), cfg)
    per_match.to_parquet(paths.processed / "player_experiment.parquet", index=False)
    out = paths.reports / "player_features_report.md"
    out.write_text(experiment.to_markdown(summary, "See `player_data_quality.md` for linking "
                                          "and coverage details."), encoding="utf-8")
    return out
