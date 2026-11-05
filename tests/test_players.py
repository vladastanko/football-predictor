"""Player-data tests: parsing, source linking, leakage, API client safety."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from fpred import clean
from fpred.config import Paths, load_config
from fpred.pipeline import fix_known_issues
from fpred.players import mapping, parse
from fpred.players.api import ApiFootball, BudgetExhausted, load_api_key
from fpred.players.features import player_form, team_player_features
from fpred.players.synthetic import generate_api_store
from fpred.synthetic import generate


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("pl")
    paths = Paths(root).ensure()
    generate(paths.raw, ["E0"], 2020, 2023, seed=5)
    played = clean.played(fix_known_issues(clean.build_matches(paths.raw, ["E0"]))[0])
    store = root / "data" / "api"
    generate_api_store(played, store, ["E0"], [2021, 2022])
    api_fx, apps = parse.parse_details(store)
    return {"played": played, "api_fx": api_fx, "apps": apps,
            "inj": parse.parse_injuries(store)}


def test_parse_shapes(world):
    assert world["api_fx"].fixture_id.is_unique
    assert (world["apps"].groupby(["fixture_id", "team_id"]).starter.sum() == 11).all()
    assert len(world["inj"]) > 0


def test_team_map_and_linking(world):
    tm = mapping.learn_team_map(world["api_fx"], world["played"])
    assert tm.accepted.all()
    assert (tm.api_name.str[-2:] == tm.fd_name.str[-2:]).all()   # 'FC 03' <-> 'E0 Club 03'
    linked = mapping.link_fixtures(world["api_fx"], world["played"], tm)
    assert linked.match_id.notna().mean() == 1.0


def test_player_form_uses_only_past_matches(world):
    """Inflating every stat of one match must not change form *at* that match."""
    api_fx, apps = world["api_fx"], world["apps"]
    target = api_fx.sort_values("date").fixture_id.iloc[200]
    f1 = player_form(api_fx, apps, 5)
    apps2 = apps.copy()
    hit = apps2.fixture_id == target
    apps2.loc[hit, ["goals", "rating", "minutes"]] = [9, 10.0, 90]
    f2 = player_form(api_fx, apps2, 5)
    cols = ["min_share", "ga90", "rating"]
    a = f1[f1.fixture_id == target].set_index(["team_id", "player_id"])[cols].sort_index()
    b = f2[f2.fixture_id == target].set_index(["team_id", "player_id"])[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_team_features_have_expected_columns(world):
    t = team_player_features(world["api_fx"], world["apps"], world["inj"])
    assert {"pl_h_inj_min_share", "pl_a_xi_rating", "pl_diff_xi_rotation"} <= set(t.columns)
    assert t.fixture_id.is_unique
    assert (t.pl_h_inj_count >= 0).all()


def test_api_key_never_hardcoded(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_api_key(tmp_path)
    (tmp_path / ".env").write_text("API_FOOTBALL_KEY=abc123\n")
    assert load_api_key(tmp_path) == "abc123"


def test_budget_and_cache(tmp_path, monkeypatch):
    cfg = load_config()["players"] | {"daily_request_budget": 1, "min_seconds_between_requests": 0}
    api = ApiFootball("k", cfg, tmp_path)
    calls = []

    class Resp:
        status_code = 200
        headers = {"x-ratelimit-requests-remaining": "99"}
        def raise_for_status(self): pass
        def json(self): return {"errors": [], "response": [{"ok": 1}]}

    monkeypatch.setattr(api.session, "get", lambda *a, **k: calls.append(1) or Resp())
    assert api.get("fixtures", {"league": 39})["response"] == [{"ok": 1}]
    assert api.get("fixtures", {"league": 39})["response"] == [{"ok": 1}]   # served from cache
    assert len(calls) == 1
    with pytest.raises(BudgetExhausted):
        api.get("fixtures", {"league": 40})
    assert json.loads((tmp_path / "_usage.json").read_text())["requests"] == 1
