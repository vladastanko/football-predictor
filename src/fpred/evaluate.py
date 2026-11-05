"""Walk-forward evaluation and probabilistic scoring rules."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .features import model_feature_columns
from .models import EloLogit, GBMModel, result_to_int

log = logging.getLogger(__name__)

MODELS = ["market", "elo", "dixon_coles", "lightgbm", "ensemble"]
MODEL_LABELS = {
    "market": "Bookmaker (closing odds)",
    "elo": "Elo + logistic",
    "dixon_coles": "Dixon-Coles",
    "lightgbm": "LightGBM",
    "ensemble": "Ensemble (DC + LGBM)",
}


# --------------------------------------------------------------- scoring rules
def _onehot(y: np.ndarray) -> np.ndarray:
    return np.eye(3)[y]


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((p - _onehot(y)) ** 2, axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score: respects the ordering home > draw > away.
    The standard metric in football forecasting literature (lower is better)."""
    cp, co = np.cumsum(p, axis=1)[:, :2], np.cumsum(_onehot(y), axis=1)[:, :2]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(p.argmax(axis=1) == y))


def score(p: np.ndarray, y: np.ndarray) -> dict:
    return {"log_loss": log_loss(p, y), "brier": brier(p, y), "rps": rps(p, y),
            "accuracy": accuracy(p, y), "n": len(y)}


# -------------------------------------------------------------- walk-forward
def _probs(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].to_numpy(float)


def _put(out: pd.DataFrame, name: str, p: np.ndarray) -> None:
    out[f"{name}_p_home"], out[f"{name}_p_draw"], out[f"{name}_p_away"] = p.T


def walk_forward(feat: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """For each test season, fit on all earlier seasons and predict that season."""
    hist = feat[~feat.is_fixture & feat.result.notna()].copy()
    hist = hist[hist.dc_p_home.notna()]               # DC needs one year of history
    fcols = model_feature_columns(hist)
    folds, importance = [], None

    for season in cfg["evaluation"]["test_seasons"]:
        train, test = hist[hist.season < season], hist[hist.season == season]
        if len(test) == 0 or len(train) < 500:
            log.warning("Skipping season %s (train=%d, test=%d)", season, len(train), len(test))
            continue
        log.info("Fold %s: train %d matches, test %d matches", season, len(train), len(test))
        out = test[["match_id", "league", "season", "date", "home_team", "away_team",
                    "home_goals", "away_goals", "result"]].copy()
        _put(out, "market", _probs(test, "mkt_p"))
        _put(out, "elo", EloLogit().fit(train).predict_proba(test))
        _put(out, "dixon_coles", _probs(test, "dc_p"))
        gbm = GBMModel(cfg["lightgbm"], fcols).fit(train)
        _put(out, "lightgbm", gbm.predict_proba(test))
        _put(out, "ensemble", (_probs(out, "dixon_coles_p") + _probs(out, "lightgbm_p")) / 2)
        folds.append(out)
        importance = gbm.importance()
    if not folds:
        raise ValueError("No evaluation folds could be built; check test_seasons in config.")
    return pd.concat(folds, ignore_index=True), importance


def summary(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Scores for every model on the same matches (those with market odds)."""
    cols = [f"{m}_p_home" for m in MODELS]
    common = preds.dropna(subset=cols)
    groups = [((), common)] if not by else common.groupby(by)
    rows = []
    for key, g in groups:
        y = result_to_int(g["result"])
        for m in MODELS:
            r = score(_probs(g, f"{m}_p"), y)
            r["model"] = m
            if by:
                r.update(dict(zip(by, key if isinstance(key, tuple) else (key,))))
            rows.append(r)
    return pd.DataFrame(rows)


def calibration_table(preds: pd.DataFrame, model: str, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, pooling home/draw/away probabilities."""
    y = _onehot(result_to_int(preds["result"]))
    p = _probs(preds, f"{model}_p")
    df = pd.DataFrame({"p": p.ravel(), "y": y.ravel()}).dropna()
    df["bin"] = pd.cut(df.p, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"),
                                             n=("y", "size")).reset_index()
    return t[t.n >= 30]
