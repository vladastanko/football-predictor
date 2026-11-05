"""Does player-level information improve the forecasts?

Three nested feature sets are compared on exactly the same matches:
    base        team-level model (Elo, Dixon-Coles, form)
    +injuries   adds absence features (available ~1 day before kick-off)
    +lineup     adds confirmed starting XI features (available ~30 min before)

Two model types are fitted per feature set:
    logistic    multinomial logistic regression on a compact feature list
                (few parameters: robust while only a few seasons of player data exist)
    lightgbm    gradient boosting on all features, trained on the full history
                (player columns are missing for seasons without player data)

Uncertainty: a paired bootstrap over matches gives a 95% interval for the RPS change.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluate import rps
from ..features import model_feature_columns
from ..models import GBMModel, result_to_int
from .features import feature_groups

log = logging.getLogger(__name__)

BASE_LOGIT = ["elo_diff", "dc_p_home", "dc_p_away"]


def per_match_rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    cp, co = np.cumsum(p, 1)[:, :2], np.cumsum(np.eye(3)[y], 1)[:, :2]
    return np.sum((cp - co) ** 2, axis=1) / 2


def bootstrap_delta(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    """Mean of (b - a) with a 95% paired bootstrap interval."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), (n, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def _logit(train, test, cols):
    tr = train.dropna(subset=cols)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    model.fit(tr[cols], result_to_int(tr.result))
    return model.predict_proba(test[cols].fillna(tr[cols].median()))


def run(feat: pd.DataFrame, pl: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """feat: base feature table; pl: player features with match_id."""
    df = feat.merge(pl, on="match_id", how="left")
    hist = df[~df.is_fixture & df.result.notna() & df.dc_p_home.notna()]
    groups = feature_groups(df)
    covered = hist[hist[groups["injuries"][0]].notna()]
    seasons = sorted(covered.season.unique())
    if len(seasons) < 2:
        raise ValueError("Player data for at least two seasons is needed (train on one, test on the next).")

    base_cols = model_feature_columns(feat)
    variants = {
        "base": [],
        "+injuries": groups["injuries"],
        "+lineup": groups["injuries"] + groups["lineup"],
    }
    compact = {
        "base": [],
        "+injuries": ["pl_diff_inj_min_share", "pl_diff_inj_ga_share"],
        "+lineup": ["pl_diff_inj_min_share", "pl_diff_inj_ga_share",
                    "pl_diff_xi_rating", "pl_diff_xi_min_share"],
    }

    rows = []
    for season in seasons[1:]:
        test = covered[covered.season == season]
        train_pl = covered[covered.season < season]
        train_all = hist[hist.season < season]
        y = result_to_int(test.result)
        for name, extra in variants.items():
            p_l = _logit(train_pl, test, BASE_LOGIT + compact[name])
            gbm = GBMModel(cfg["lightgbm"], base_cols + extra).fit(train_all)
            p_g = gbm.predict_proba(test)
            for model, p in (("logistic", p_l), ("lightgbm", p_g)):
                rows.append(pd.DataFrame({"season": season, "match_id": test.match_id.values,
                                          "model": model, "variant": name,
                                          "rps": per_match_rps(p, y)}))
        log.info("Season %s: %d test matches", season, len(test))
    long = pd.concat(rows, ignore_index=True)

    out = []
    for model in ("logistic", "lightgbm"):
        w = long[long.model == model].pivot(index="match_id", columns="variant", values="rps")
        for v in ("base", "+injuries", "+lineup"):
            mean = w[v].mean()
            if v == "base":
                out.append({"model": model, "variant": v, "rps": mean, "delta_vs_base": 0.0,
                            "ci_low": np.nan, "ci_high": np.nan, "matches": len(w)})
            else:
                d, lo, hi = bootstrap_delta(w["base"].values, w[v].values)
                out.append({"model": model, "variant": v, "rps": mean, "delta_vs_base": d,
                            "ci_low": lo, "ci_high": hi, "matches": len(w)})
    return pd.DataFrame(out), long


def to_markdown(summary: pd.DataFrame, coverage_md: str) -> str:
    s = summary.copy()
    s["relative_change_%"] = s.delta_vs_base / (s.rps - s.delta_vs_base) * 100
    s["significant"] = np.where(s.variant == "base", "",
                                np.where(s.ci_high < 0, "yes (better)",
                                         np.where(s.ci_low > 0, "yes (worse)", "no")))
    for c in ("rps", "delta_vs_base", "ci_low", "ci_high"):
        s[c] = s[c].round(5)
    s["relative_change_%"] = s["relative_change_%"].round(2)
    return "\n".join([
        "# Player-level features: do they help?", "",
        "Walk-forward by season on matches with player data. Lower RPS is better; a negative",
        "delta means the variant improved on the team-only model. The 95% interval comes from",
        "a paired bootstrap over matches: if it contains 0, the change is not distinguishable from noise.",
        "", s.to_markdown(index=False), "", coverage_md])
