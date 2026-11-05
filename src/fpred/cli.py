"""Command line interface.

    fpred download   # fetch CSVs from football-data.co.uk
    fpred build      # clean, validate, build features + Dixon-Coles predictions
    fpred evaluate   # walk-forward backtest + report
    fpred predict    # probabilities for upcoming fixtures
    fpred all        # everything above, in order
    fpred demo       # full pipeline on synthetic data (no internet needed)

    fpred players-collect   # download line-ups, player stats, injuries (API-Football)
    fpred players           # link sources, build player features, run the experiment
    fpred players-demo      # the player steps on synthetic API data
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .config import PROJECT_ROOT, Paths, load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fpred", description="Football match outcome model")
    ap.add_argument("command", choices=["download", "build", "evaluate", "predict", "all", "demo",
                                       "players-collect", "players", "players-demo"])
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    ap.add_argument("--root", default=None, help="Project root for data/ and reports/")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("fpred")
    cfg = load_config(args.config)
    root = Path(args.root) if args.root else PROJECT_ROOT
    t0 = time.time()
    paths, state = None, None

    if args.command == "players-collect":
        from .players.pipeline import collect
        collect(cfg, Paths(root).ensure())
        log.info("Done in %.0fs", time.time() - t0)
        return

    if args.command == "players-demo":
        from . import pipeline
        from .players.pipeline import api_store
        from .players.synthetic import generate_api_store
        from .synthetic import generate
        root = Path(args.root) if args.root else PROJECT_ROOT / "demo"
        paths = Paths(root).ensure()
        if not any(paths.raw.glob("*/*.csv")):
            generate(paths.raw, list(cfg["leagues"]), cfg["seasons"]["first"],
                     cfg["seasons"]["last"], inject_errors=True)
        state = pipeline.build(cfg, paths)
        played = state["feat"][~state["feat"].is_fixture]
        seasons = [s for s in sorted(played.season.unique())][-4:-1]
        generate_api_store(played, api_store(paths), ["E0"], seasons)
        cfg["players"]["collect"] = {"E0": seasons}
        args.command = "players"

    if args.command == "players":
        from . import pipeline
        from .players.pipeline import build_and_evaluate
        paths = paths or Paths(root).ensure()
        state = state or pipeline.build(cfg, paths)
        try:
            out = build_and_evaluate(cfg, paths, state["feat"])
        except (FileNotFoundError, ValueError) as exc:
            log.error("%s", exc)
            return
        log.info("Player report written: %s", out)
        log.info("Done in %.0fs", time.time() - t0)
        return

    if args.command == "demo":
        from .synthetic import generate
        root = Path(args.root) if args.root else PROJECT_ROOT / "demo"
        paths = Paths(root).ensure()
        log.info("Generating synthetic data in %s", paths.raw)
        generate(paths.raw, list(cfg["leagues"]), cfg["seasons"]["first"],
                 cfg["seasons"]["last"], inject_errors=True)
        steps = ["build", "evaluate", "predict"]
    else:
        paths = Paths(root).ensure()
        steps = ["download", "build", "evaluate", "predict"] if args.command == "all" \
            else [args.command]

    from . import pipeline
    state = None
    for step in steps:
        log.info("=== %s ===", step)
        if step == "download":
            from .ingest import download_fixtures, download_history
            files = download_history(cfg, paths)
            download_fixtures(cfg, paths)
            log.info("%d season files available", len(files))
        elif step == "build":
            state = pipeline.build(cfg, paths)
        else:
            if state is None:
                state = pipeline.build(cfg, paths)
            if step == "evaluate":
                path = pipeline.run_evaluation(cfg, paths, state["feat"], state["dq_summary"])
                log.info("Report written: %s", path)
            elif step == "predict":
                path = pipeline.predict_fixtures(cfg, paths, state["feat"])
                if path:
                    log.info("Predictions written: %s", path)
    log.info("Done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
