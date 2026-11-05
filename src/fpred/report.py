"""Charts (PNG) and a Markdown evaluation report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import MODEL_LABELS, MODELS, calibration_table  # noqa: E402

# Reference palette (validated categorical order) + text tokens
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# charts with 2-3 series use the first three slots (validated for all pairs)
MODEL_COLOR = dict(zip(["market", "ensemble", "dixon_coles", "elo", "lightgbm"], SERIES))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def chart_model_scores(overall: pd.DataFrame, path: Path) -> str:
    # dot plot: differences are small, and bars on a truncated axis would exaggerate them
    d = overall.set_index("model").loc[MODELS[::-1]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = range(len(d))
    ax.scatter(d.rps, y, s=70, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(d.rps):
        ax.annotate(f"{v:.4f}", (v, i), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TEXT_2, fontsize=9)
    ax.set_yticks(list(y), [MODEL_LABELS[m] for m in d.index])
    pad = (d.rps.max() - d.rps.min()) * 0.25
    ax.set_xlim(d.rps.min() - pad, d.rps.max() + pad * 1.6)
    ax.set_xlabel("Ranked Probability Score (lower is better)")
    ax.set_title("Out-of-sample accuracy by model", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def chart_calibration(preds: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1, ls="--", label="Perfect calibration")
    for m in ("market", "dixon_coles", "ensemble"):
        t = calibration_table(preds, m)
        ax.plot(t.predicted, t.observed, marker="o", ms=5, lw=2,
                color=MODEL_COLOR[m], label=MODEL_LABELS[m])
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (home, draw and away pooled)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def chart_league_rps(by_league: pd.DataFrame, league_names: dict, path: Path) -> str:
    d = by_league.pivot(index="league", columns="model", values="rps")
    d = d.sort_values("market")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = range(len(d))
    labels = [league_names.get(lg, lg) for lg in d.index]
    for m in ("market", "ensemble"):
        ax.scatter(d[m], y, s=60, color=MODEL_COLOR[m], label=MODEL_LABELS[m], zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
    for i, (a, b) in enumerate(zip(d["market"], d["ensemble"])):
        ax.plot([a, b], [i, i], color=GRID, lw=2, zorder=2)
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("RPS (lower is better)")
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=2, borderaxespad=0)
    ax.set_title("Model vs bookmaker, by league", loc="left", fontweight="bold", pad=28)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def chart_importance(imp: pd.Series, path: Path, top: int = 15) -> str:
    s = (imp / imp.sum()).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(s.index, s.values * 100, color=SERIES[0], height=0.6)
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("LightGBM: most important features", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def write_report(out_dir: Path, overall: pd.DataFrame, by_league: pd.DataFrame,
                 by_season: pd.DataFrame, preds: pd.DataFrame, imp: pd.Series,
                 league_names: dict, dq_summary: str) -> Path:
    imgs = {
        "scores": chart_model_scores(overall, out_dir / "model_scores.png"),
        "calib": chart_calibration(preds, out_dir / "calibration.png"),
        "league": chart_league_rps(by_league, league_names, out_dir / "league_rps.png"),
        "imp": chart_importance(imp, out_dir / "feature_importance.png"),
    }
    tbl = overall.copy()
    tbl["model"] = tbl.model.map(MODEL_LABELS)
    tbl = tbl[["model", "rps", "log_loss", "brier", "accuracy", "n"]].round(4)
    best = overall[overall.model != "market"].sort_values("rps").iloc[0]
    mkt = overall.set_index("model").loc["market"]
    seasons = by_season.pivot(index="season", columns="model", values="rps")[MODELS].round(4)
    seasons.columns = [MODEL_LABELS[c] for c in seasons.columns]

    md = f"""# Evaluation report

Walk-forward backtest: for each test season the models are trained only on earlier
seasons, then scored on every match of that season. All models are compared on the
same {int(mkt.n):,} matches (those with bookmaker closing odds).

**Best own model:** {MODEL_LABELS[best.model]} - RPS {best.rps:.4f}
vs bookmaker closing odds RPS {mkt.rps:.4f}
({(best.rps / mkt.rps - 1) * 100:+.2f}% relative difference).

![Model scores]({imgs['scores']})

## Overall scores

{tbl.to_markdown(index=False)}

- **RPS** (Ranked Probability Score) is the standard metric for football forecasts; it
  rewards probability mass close to the actual outcome (home > draw > away).
- **Log loss** punishes confident wrong predictions hardest.
- Closing odds are the efficient-market benchmark: matching them is a strong result,
  beating them consistently is rare.

## By season (RPS)

{seasons.to_markdown()}

## By league

![League RPS]({imgs['league']})

## Calibration

When the model says 40%, the event should happen about 40% of the time.

![Calibration]({imgs['calib']})

## What the model learned

![Feature importance]({imgs['imp']})

## Data quality summary

{dq_summary}
"""
    path = out_dir / "evaluation_report.md"
    path.write_text(md, encoding="utf-8")
    return path
