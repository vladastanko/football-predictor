"""Tests: parsing, leakage, model correctness, scoring rules, data checks, end-to-end."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fpred import clean, validate
from fpred.config import Paths, load_config, season_code
from fpred.evaluate import brier, log_loss, rps
from fpred.features import build_features, compute_elo
from fpred.models import DixonColes
from fpred.synthetic import generate

LEAGUES = ["E0", "SP1"]


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["seasons"] = {"first": 2018, "last": 2023}
    c["evaluation"] = {"test_seasons": [2021, 2022]}
    c["leagues"] = {k: c["leagues"][k] for k in LEAGUES}
    return c


@pytest.fixture(scope="module")
def raw(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("proj")
    paths = Paths(root).ensure()
    generate(paths.raw, LEAGUES, 2018, 2023, inject_errors=True, seed=1)
    return paths


@pytest.fixture(scope="module")
def matches(raw):
    return clean.build_matches(raw.raw, LEAGUES)


# ------------------------------------------------------------------ parsing
def test_season_code():
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"


def test_dates_two_formats():
    s = pd.Series(["05/08/2023", "12/08/23"])
    d = clean._parse_dates(s)
    assert list(d.dt.strftime("%Y-%m-%d")) == ["2023-08-05", "2023-08-12"]


def test_market_probs_priority_and_normalisation():
    df = pd.DataFrame({"B365H": [2.0], "B365D": [3.5], "B365A": [4.0],
                       "PSCH": [2.1], "PSCD": [3.4], "PSCA": [3.9]})
    out = clean._market_probs(df)
    assert out.mkt_source.iloc[0] == "pinnacle_closing"
    assert out[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].sum(axis=1).iloc[0] == pytest.approx(1)
    assert 0 < out.mkt_overround.iloc[0] < 0.1


def test_latin1_file(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nSP1,01/09/2020,Alavés,Cádiz,1,0\n"
                  .encode("latin-1"))
    df = clean.standardize(clean._read_csv(p))
    assert df.home_team.iloc[0] == "Alavés"


def test_unplayed_matches_do_not_break_checks(matches, cfg):
    """Real current-season files can contain rows without a score (regression test)."""
    m = matches.copy()
    m.loc[m.index[:5], ["home_goals", "away_goals", "home_sot", "away_sot"]] = pd.NA
    res = validate.run_checks(m, cfg["seasons"]["last"])
    assert any(r.name == "result_vs_score" for r in res)
    assert (clean.result_from_score(m.home_goals, m.away_goals)[:5] == "D").all()


def test_files_missing_stat_columns(tmp_path, cfg):
    """Older seasons of some leagues have no shot columns (regression test)."""
    full = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST\nE0,10/08/2024,A,B,2,1,H,10,8,4,3\n"
    bare = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,12/08/2023,A,B,0,0,D\nE0,13/08/2023,C,D,,,\n"
    (tmp_path / "E0").mkdir()
    (tmp_path / "E0" / "2425.csv").write_text(full)
    (tmp_path / "E0" / "2324.csv").write_text(bare)
    m = clean.build_matches(tmp_path, ["E0"])
    res = {r.name: r for r in validate.run_checks(m, 2024)}
    assert res["goals_gt_sot_home"].passed
    assert m.home_sot.dtype == float


# ------------------------------------------------------------ data quality
def test_injected_errors_are_caught(matches, cfg):
    res = {r.name: r for r in validate.run_checks(matches, cfg["seasons"]["last"])}
    assert not res["duplicate_match"].passed
    assert not res["result_vs_score"].passed
    assert not res["sot_gt_shots_home"].passed
    assert res["missing_result_past_season"].passed


# ------------------------------------------------------------------ leakage
def test_features_do_not_see_the_future(matches, cfg):
    """Changing results after a cut-off date must not change any earlier feature."""
    from fpred.pipeline import fix_known_issues
    m, _ = fix_known_issues(matches)
    played = clean.played(m)
    cut = played.date.quantile(0.6)
    altered = played.copy()
    late = altered.date >= cut
    altered.loc[late, ["home_goals", "away_goals"]] = altered.loc[late, ["away_goals", "home_goals"]].values
    altered = clean.played(altered)

    cols = ["elo_home", "elo_away", "h_form_points", "a_form_goals_for", "h_form_sot_for"]
    f1 = build_features(played, cfg).set_index("match_id")
    f2 = build_features(altered, cfg).set_index("match_id")
    early = f1.index[f1.date < cut]
    pd.testing.assert_frame_equal(f1.loc[early, cols], f2.loc[early, cols])
    # sanity: later features do change
    later = f1.index[f1.date > cut + pd.Timedelta(days=30)]
    assert not f1.loc[later, "elo_home"].equals(f2.loc[later, "elo_home"])


def test_elo_is_zero_sum(matches, cfg):
    played = clean.played(matches.drop_duplicates("match_id"))
    elo, final = compute_elo(played[played.season == 2018], cfg)
    league = [v for (lg, _), v in final.items() if lg == "E0"]
    assert np.mean(league) == pytest.approx(cfg["elo"]["initial_rating"], abs=1e-6)


# ------------------------------------------------------------------ models
def test_dixon_coles_recovers_strengths():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(16)]
    att = dict(zip(teams, np.linspace(-0.5, 0.5, 16)))
    rows = []
    date = pd.Timestamp("2020-01-01")
    for _ in range(4):
        for h in teams:
            for a in teams:
                if h != a:
                    lh, la = np.exp(0.1 + 0.3 + att[h] - att[a]), np.exp(0.1 + att[a] - att[h])
                    rows.append((date, h, a, rng.poisson(lh), rng.poisson(la)))
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    m = DixonColes(half_life_days=1e6).fit(df, date)
    est = pd.Series(m.att, index=list(m.teams)).loc[teams]
    assert np.corrcoef(est, list(att.values()))[0, 1] > 0.95
    assert m.home == pytest.approx(0.3, abs=0.08)
    p = m.predict(df.head(50))[["dc_p_home", "dc_p_draw", "dc_p_away"]].sum(axis=1)
    assert np.allclose(p, 1)


# ----------------------------------------------------------- scoring rules
def test_scoring_rules():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert rps(perfect, y) == pytest.approx(0)
    assert brier(perfect, y) == pytest.approx(0)
    uniform = np.full((3, 3), 1 / 3)
    assert log_loss(uniform, y) == pytest.approx(np.log(3))
    # RPS rewards near misses: predicting draw is better than away when home wins
    near, far = np.array([[0, 1, 0]]), np.array([[0, 0, 1]])
    assert rps(near, np.array([0])) < rps(far, np.array([0]))


# --------------------------------------------------------------- end to end
def test_end_to_end(raw, cfg):
    from fpred import pipeline
    c = copy.deepcopy(cfg)
    state = pipeline.build(c, raw)
    report = pipeline.run_evaluation(c, raw, state["feat"], state["dq_summary"])
    assert report.exists()
    scores = pd.read_csv(raw.reports / "scores_overall.csv")
    assert set(scores.model) == {"market", "elo", "dixon_coles", "lightgbm", "ensemble"}
    assert (scores.rps < 0.25).all()                 # all better than a naive guess
    pred = pipeline.predict_fixtures(c, raw, state["feat"])
    out = pd.read_csv(pred)
    assert len(out) > 0
    assert np.allclose(out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1, atol=0.01)
