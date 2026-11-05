"""Probability models for match outcomes (home win / draw / away win).

1. MarketModel      - bookmaker implied probabilities (benchmark, not trained by us)
2. EloLogit         - multinomial logistic regression on the Elo difference
3. DixonColes       - time-weighted Poisson goal model with Dixon-Coles low-score correction
4. GBMModel         - LightGBM on Elo, form and Dixon-Coles features
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)

CLASSES = ["H", "D", "A"]
PROB_COLS = ["p_home", "p_draw", "p_away"]


def result_to_int(result: pd.Series) -> np.ndarray:
    return result.map({"H": 0, "D": 1, "A": 2}).to_numpy()


# ----------------------------------------------------------------- Elo + logit
class EloLogit:
    features = ["elo_diff"]

    def fit(self, df: pd.DataFrame) -> "EloLogit":
        d = df.dropna(subset=self.features)
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(d[self.features] / 100, result_to_int(d["result"]))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full((len(df), 3), np.nan)
        ok = df[self.features].notna().all(axis=1).to_numpy()
        out[ok] = self.model.predict_proba(df.loc[ok, self.features] / 100)
        return out


# ------------------------------------------------------------- Dixon-Coles
def _tau(x, y, lh, la, rho):
    """Dixon-Coles adjustment for 0-0, 1-0, 0-1 and 1-1 scores."""
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class DixonColes:
    """Team attack/defence strengths, fitted by weighted maximum likelihood.

    log(lambda_home) = mu + home_adv + attack[home] + defence[away]
    log(lambda_away) = mu +            attack[away] + defence[home]
    Matches are weighted by exp(-ln2 * age / half_life). A small ridge penalty
    keeps the parameters identifiable and stabilises teams with few matches.
    """

    def __init__(self, half_life_days: float = 365, ridge: float = 0.01, max_goals: int = 10):
        self.xi = np.log(2) / half_life_days
        self.ridge = ridge
        self.max_goals = max_goals

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp) -> "DixonColes":
        teams = sorted(set(df.home_team) | set(df.away_team))
        self.teams = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = df.home_team.map(self.teams).to_numpy()
        a = df.away_team.map(self.teams).to_numpy()
        x = df.home_goals.to_numpy(float)
        y = df.away_goals.to_numpy(float)
        age = (ref_date - df.date).dt.days.to_numpy(float)
        w = np.exp(-self.xi * age)

        def nll(p):
            mu, home, att, dfn = p[0], p[1], p[2:2 + n], p[2 + n:]
            llh = mu + home + att[h] + dfn[a]
            lla = mu + att[a] + dfn[h]
            lh, la = np.exp(llh), np.exp(lla)
            # weighted Poisson log-likelihood (constant terms dropped)
            f = -np.sum(w * (x * llh - lh + y * lla - la))
            f += self.ridge * (att @ att + dfn @ dfn)
            rh, ra = w * (x - lh), w * (y - la)        # d(loglik)/d(log lambda)
            g = np.empty_like(p)
            g[0] = -(rh.sum() + ra.sum())
            g[1] = -rh.sum()
            g[2:2 + n] = -(np.bincount(h, rh, n) + np.bincount(a, ra, n)) + 2 * self.ridge * att
            g[2 + n:] = -(np.bincount(a, rh, n) + np.bincount(h, ra, n)) + 2 * self.ridge * dfn
            return f, g

        p0 = np.zeros(2 + 2 * n)
        p0[0] = np.log(max(np.average((x + y) / 2, weights=w), 0.1))
        p0[1] = 0.25
        r = minimize(nll, p0, jac=True, method="L-BFGS-B")
        if not r.success:
            log.debug("Dixon-Coles optimiser: %s", r.message)
        self.mu, self.home = r.x[0], r.x[1]
        self.att, self.dfn = r.x[2:2 + n], r.x[2 + n:]

        lh, la = self._lambdas(h, a)
        low = (x <= 1) & (y <= 1)

        def rho_nll(rho):
            t = _tau(x[low], y[low], lh[low], la[low], rho)
            return -np.sum(w[low] * np.log(np.clip(t, 1e-9, None)))
        self.rho = minimize_scalar(rho_nll, bounds=(-0.2, 0.2), method="bounded").x

        # parameters for teams with no history in the window (e.g. promoted clubs)
        self.new_att = np.quantile(self.att, 0.25)
        self.new_dfn = np.quantile(self.dfn, 0.75)
        return self

    def _lambdas(self, h, a, att=None, dfn=None):
        att = self.att if att is None else att
        dfn = self.dfn if dfn is None else dfn
        lh = np.exp(self.mu + self.home + att[h] + dfn[a])
        la = np.exp(self.mu + att[a] + dfn[h])
        return lh, la

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(self.teams)
        att = np.append(self.att, self.new_att)      # index n = unknown team
        dfn = np.append(self.dfn, self.new_dfn)
        h = df.home_team.map(self.teams).fillna(n).astype(int).to_numpy()
        a = df.away_team.map(self.teams).fillna(n).astype(int).to_numpy()
        lh, la = self._lambdas(h, a, att, dfn)

        g = np.arange(self.max_goals + 1)
        ph = poisson.pmf(g[None, :], lh[:, None])           # (m, G)
        pa = poisson.pmf(g[None, :], la[:, None])
        grid = ph[:, :, None] * pa[:, None, :]              # (m, G, G)
        for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
            grid[:, i, j] *= _tau(np.array(i), np.array(j), lh, la, self.rho)
        grid /= grid.sum(axis=(1, 2), keepdims=True)
        tri_l = np.tril(np.ones((len(g), len(g)), bool), -1)  # home goals > away goals
        return pd.DataFrame({
            "dc_p_home": grid[:, tri_l].sum(axis=1),
            "dc_p_draw": np.trace(grid, axis1=1, axis2=2),
            "dc_p_away": grid[:, tri_l.T].sum(axis=1),
            "dc_xg_home": lh, "dc_xg_away": la,
        }, index=df.index)


def dixon_coles_walk_forward(played: pd.DataFrame, cfg: dict,
                             fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Out-of-sample Dixon-Coles predictions for every match.

    For each league the model is refitted every `refit_every_days`, using only
    matches played before the refit date, and predicts the matches in the
    following block. The first year of each league has no prediction.
    """
    c = cfg["dixon_coles"]
    step = pd.Timedelta(days=c["refit_every_days"])
    window = pd.Timedelta(days=c["train_window_days"])
    preds = []
    for league, df in played.groupby("league"):
        df = df.sort_values("date")
        t = df.date.min() + pd.Timedelta(days=365)
        while t <= df.date.max():
            nxt = t + step
            block = df[(df.date >= t) & (df.date < nxt)]
            if len(block):
                train = df[(df.date < t) & (df.date >= t - window)]
                model = DixonColes(c["half_life_days"]).fit(train, t)
                preds.append(model.predict(block).assign(match_id=block.match_id.values))
            t = nxt
        # upcoming fixtures: one fit on the most recent window of played matches
        if fixtures is not None:
            fx = fixtures[fixtures.league == league]
            if len(fx):
                ref = df.date.max() + pd.Timedelta(days=1)
                model = DixonColes(c["half_life_days"]).fit(df[df.date >= ref - window], ref)
                preds.append(model.predict(fx).assign(match_id=fx.match_id.values))
    return pd.concat(preds, ignore_index=True).drop_duplicates("match_id", keep="last")


# ------------------------------------------------------------------- LightGBM
class GBMModel:
    def __init__(self, params: dict, features: list[str]):
        self.params = params
        self.features = features

    def fit(self, df: pd.DataFrame) -> "GBMModel":
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                        verbose=-1, random_state=42, **self.params)
        self.model.fit(df[self.features], result_to_int(df["result"]))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])

    def importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)
