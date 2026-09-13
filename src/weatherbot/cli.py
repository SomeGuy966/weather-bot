"""Command-line entry point: ``weatherbot <stage> [options]``.

Stages, in pipeline order::

    fetch      download / refresh the raw IEM archives (CLI labels, METARs, NBM + GFS MOS)
    build      turn raw archives into the snapshot feature table
    train      fit LightGBM + calibration on the time-based training split
    evaluate   score the held-out years against baselines; write results/ and figures/
    kalshi     archive Kalshi markets, candles and trades for the eight series
    backtest   replay archived Kalshi prices against model probabilities
    live       compute current contract probabilities (and optionally paper-trade)
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from weatherbot.config import HISTORY_START


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weatherbot", description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="stage", required=True)

    fetch = sub.add_parser("fetch", help="download / refresh the raw IEM archives")
    fetch.add_argument("--start", type=_parse_date, default=HISTORY_START)
    fetch.add_argument("--end", type=_parse_date, default=None)
    fetch.add_argument("--force", action="store_true", help="re-download completed years too")

    build = sub.add_parser("build", help="build the snapshot feature table")
    build.add_argument("--start", type=_parse_date, default=HISTORY_START)
    build.add_argument("--end", type=_parse_date, default=None)

    train = sub.add_parser("train", help="fit the model on the training split")
    train.add_argument("--train-end", type=_parse_date, default=date(2024, 12, 31))
    train.add_argument("--calib-end", type=_parse_date, default=date(2025, 6, 30))

    evaluate = sub.add_parser("evaluate", help="score the test split; write results/ + figures/")
    evaluate.add_argument("--results-dir", default="results")
    evaluate.add_argument("--figures-dir", default="figures")

    kalshi = sub.add_parser("kalshi", help="archive Kalshi markets, candles and trades")
    kalshi.add_argument("--no-trades", action="store_true", help="skip the trade tape")

    backtest = sub.add_parser("backtest", help="replay archived Kalshi prices vs model")
    backtest.add_argument("--results-dir", default="results")
    backtest.add_argument("--figures-dir", default="figures")

    live = sub.add_parser("live", help="current contract probabilities for open markets")
    live.add_argument("--station", default=None, help="ICAO id; default: all eight")
    live.add_argument("--paper", action="store_true", help="record paper trades to live/")
    live.add_argument("--loop", type=int, default=0, help="repeat every N seconds (0 = once)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.stage == "fetch":
        from weatherbot.data.iem import load_cli, load_metar, load_mos

        cli = load_cli(start=args.start, end=args.end, force=args.force)
        metar = load_metar(start=args.start, end=args.end, force=args.force)
        nbs = load_mos("NBS", start=args.start, end=args.end, force=args.force)
        gfs = load_mos("GFS", start=args.start, end=args.end, force=args.force)
        print(
            f"CLI days: {len(cli):,}  METARs: {len(metar):,}  "
            f"NBS rows: {len(nbs):,}  GFS rows: {len(gfs):,}"
        )
    elif args.stage == "build":
        from weatherbot.features import build_snapshots

        table = build_snapshots(start=args.start, end=args.end)
        print(f"snapshot rows: {len(table):,}  columns: {table.shape[1]}")
    elif args.stage == "train":
        from weatherbot.model import train_and_save

        train_and_save(train_end=args.train_end, calib_end=args.calib_end)
    elif args.stage == "evaluate":
        from weatherbot.evaluate import run_evaluation

        run_evaluation(results_dir=args.results_dir, figures_dir=args.figures_dir)
    elif args.stage == "kalshi":
        from weatherbot.data.kalshi import archive_all

        archive_all(with_trades=not args.no_trades)
    elif args.stage == "backtest":
        from weatherbot.backtest import run_backtest

        run_backtest(results_dir=args.results_dir, figures_dir=args.figures_dir)
    elif args.stage == "live":
        from weatherbot.live import run_live

        run_live(station=args.station, paper=args.paper, loop_seconds=args.loop)
