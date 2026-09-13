"""Replay archived Kalshi prices against the model's contract probabilities.

For every archived, settled market and every hourly snapshot while it was open, the model's
``P(YES)`` is compared with the order book as of that hour (last hourly candle's closing bid /
ask). Two things come out:

1. **Information**: Brier score / log-loss of the model vs the market mid on the same
   contract-hours, plus a 50/50 blend. If the model does not beat the mid, there is nothing
   to trade; if the blend beats both, the model carries information the market lacks.
2. **P&L**: a deliberately simple rule — buy YES at the ask when ``P - ask`` exceeds a margin
   after Kalshi's taker fee, buy NO at ``1 - bid`` symmetrically, one contract, at most one
   entry per contract per day, hold to settlement. Reported after fees with a bootstrap
   confidence interval over days, because two months of summer is a small sample.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from weatherbot.config import KALSHI_TAKER_FEE_RATE, STATIONS_BY_SERIES
from weatherbot.contracts import contract_probability, parse_contract
from weatherbot.data.kalshi import load_archive
from weatherbot.features import load_snapshots
from weatherbot.model import HighModel
from weatherbot.util import records

log = logging.getLogger(__name__)

EDGE_MARGINS = (0.02, 0.05, 0.10)
MAX_SPREAD = 0.20


def taker_fee(price: float | np.ndarray) -> np.ndarray:
    """Kalshi taker fee per contract in dollars: ``0.07 · p · (1 - p)`` rounded up to a cent."""
    p = np.asarray(price, dtype=float)
    fee: np.ndarray = np.ceil(KALSHI_TAKER_FEE_RATE * p * (1 - p) * 100 - 1e-9) / 100
    return fee


def contract_hours(model: HighModel) -> pd.DataFrame:
    """One row per (settled market, hourly snapshot while open) with model and market probs."""
    markets, candles, _ = load_archive()
    markets = markets[markets["result"].isin(["yes", "no"])].copy()
    snaps = load_snapshots()
    snaps = snaps[snaps["day"] >= markets["day"].min()].copy()
    support, pmf = model.predict_high_pmf(snaps)
    snaps["_row"] = np.arange(len(snaps))

    candles = candles.sort_values(["ticker", "end_ts"])
    candles["end_ts"] = candles["end_ts"].astype("datetime64[us, UTC]")
    snaps["t"] = snaps["t"].astype("datetime64[us, UTC]")
    frames = []
    for mkt in records(markets):
        station = STATIONS_BY_SERIES.get(mkt["series_ticker"])
        if station is None:
            continue
        try:
            contract = parse_contract(mkt)
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        s = snaps[(snaps["station"] == station.icao) & (snaps["day"] == mkt["day"])]
        s = s[(s["t"] >= mkt["open_time"]) & (s["t"] < mkt["close_time"])]
        if s.empty:
            continue
        c = candles[candles["ticker"] == mkt["ticker"]]
        if c.empty:
            continue
        idx = s["_row"].to_numpy()
        part = pd.merge_asof(
            s[["t", "hour_lst"]].reset_index(drop=True),
            c[["end_ts", "bid_close", "ask_close", "close"]].rename(
                columns={"end_ts": "t", "bid_close": "bid", "ask_close": "ask", "close": "last"}
            ),
            on="t",
            direction="backward",
        )
        part["p_model"] = contract_probability(support[idx], pmf[idx], contract)
        part["ticker"] = mkt["ticker"]
        part["series"] = mkt["series_ticker"]
        part["station"] = station.icao
        part["day"] = mkt["day"]
        part["contract"] = contract.label
        part["outcome"] = 1.0 if mkt["result"] == "yes" else 0.0
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["tradeable"] = (
        df["bid"].notna()
        & df["ask"].notna()
        & (df["spread"] <= MAX_SPREAD)
        & (df["ask"] < 1)
        & (df["bid"] > 0)
    )
    return df


def information_scores(ch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier / log-loss of model, market mid and their blend on tradeable contract-hours."""
    d = ch[ch["tradeable"]].copy()
    d["p_blend"] = 0.5 * (d["p_model"] + d["mid"])
    rows = []
    rel = []
    for name, col in (("model", "p_model"), ("market", "mid"), ("blend", "p_blend")):
        p = d[col].to_numpy(float)
        o = d["outcome"].to_numpy(float)
        pc = np.clip(p, 1e-4, 1 - 1e-4)
        rows.append(
            {
                "model": name,
                "brier": float(((p - o) ** 2).mean()),
                "logloss": float(-(o * np.log(pc) + (1 - o) * np.log(1 - pc)).mean()),
                "n": len(d),
            }
        )
        edges = np.linspace(0, 1, 11)
        which = np.clip(np.digitize(p, edges) - 1, 0, 9)
        for b in range(10):
            m = which == b
            if m.sum() >= 20:
                rel.append(
                    {
                        "model": name,
                        "bin": b,
                        "p_mean": p[m].mean(),
                        "freq": o[m].mean(),
                        "n": int(m.sum()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(rel)


def simulate(ch: pd.DataFrame, margin: float) -> pd.DataFrame:
    """One contract per signal, first signal per contract-day, hold to settlement."""
    d = ch[ch["tradeable"]].sort_values(["ticker", "t"]).copy()
    fee_yes = taker_fee(d["ask"].to_numpy(float))
    fee_no = taker_fee(1 - d["bid"].to_numpy(float))
    edge_yes = d["p_model"] - d["ask"] - fee_yes
    edge_no = d["bid"] - d["p_model"] - fee_no
    d["side"] = np.where(edge_yes > margin, "yes", np.where(edge_no > margin, "no", ""))
    d = d[d["side"] != ""]
    d = d.groupby("ticker", as_index=False).first()  # first signal per contract
    price = np.where(d["side"] == "yes", d["ask"], 1 - d["bid"])
    won = np.where(d["side"] == "yes", d["outcome"] == 1, d["outcome"] == 0)
    d["price"] = price
    d["fee"] = taker_fee(price)
    d["pnl"] = won.astype(float) - price - d["fee"]
    d["won"] = won
    d["margin"] = margin
    return d


def summarize(trades: pd.DataFrame, n_days: int) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "hit_rate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "sharpe_daily": float("nan"),
        }
    daily = trades.groupby("day")["pnl"].sum()
    rng = np.random.default_rng(0)
    boots = [
        daily.sample(len(daily), replace=True, random_state=int(rng.integers(1 << 31))).sum()
        for _ in range(2000)
    ]
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else float("nan")
    return {
        "trades": float(len(trades)),
        "pnl_total": float(trades["pnl"].sum()),
        "pnl_per_trade": float(trades["pnl"].mean()),
        "hit_rate": float(trades["won"].mean()),
        "avg_price": float(trades["price"].mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "sharpe_daily": sharpe,
        "days_traded": float(daily.shape[0]),
        "days_available": float(n_days),
    }


def run_backtest(results_dir: str = "results", figures_dir: str = "figures") -> None:
    from weatherbot import plotting

    model = HighModel.load()
    ch = contract_hours(model)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = Path(figures_dir)
    fig.mkdir(parents=True, exist_ok=True)
    ch.to_parquet(out / "contract_hours.parquet", index=False)
    log.info(
        "%d contract-hours over %d markets, %d tradeable",
        len(ch),
        ch["ticker"].nunique(),
        int(ch["tradeable"].sum()),
    )

    info, rel = information_scores(ch)
    info.to_csv(out / "backtest_information.csv", index=False)
    rel.to_csv(out / "backtest_reliability.csv", index=False)

    n_days = ch["day"].nunique()
    summaries = []
    curves = []
    for margin in EDGE_MARGINS:
        trades = simulate(ch, margin)
        trades.to_csv(out / f"backtest_trades_m{int(margin * 100):02d}.csv", index=False)
        summaries.append({"margin": margin, **summarize(trades, n_days)})
        daily = pd.DataFrame(trades.groupby("day", as_index=False)["pnl"].sum())
        daily["strategy"] = f"margin {margin:.2f}"
        curves.append(daily)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    by_hour = (
        simulate(ch, 0.05).groupby("hour_lst")["pnl"].agg(["count", "sum", "mean"]).reset_index()
    )
    by_hour.to_csv(out / "backtest_by_hour.csv", index=False)
    by_station = (
        simulate(ch, 0.05).groupby("station")["pnl"].agg(["count", "sum", "mean"]).reset_index()
    )
    by_station.to_csv(out / "backtest_by_station.csv", index=False)

    plotting.backtest_curves(pd.concat(curves, ignore_index=True), fig / "backtest_pnl.png")
    plotting.market_vs_model(rel, fig / "market_vs_model.png")

    text = _markdown(ch, info, summary, by_station)
    (out / "backtest_summary.md").write_text(text)
    print(text)


def _markdown(
    ch: pd.DataFrame, info: pd.DataFrame, summary: pd.DataFrame, by_station: pd.DataFrame
) -> str:
    lines = [
        "# Backtest on archived Kalshi prices",
        "",
        f"Markets: {ch['ticker'].nunique():,} contracts over {ch['day'].nunique()} days "
        f"({ch['day'].min()} → {ch['day'].max()}), {len(ch):,} contract-hours, "
        f"{int(ch['tradeable'].sum()):,} with a two-sided book (spread ≤ {MAX_SPREAD:.2f}).",
        "",
        "## Information: who knows more, the model or the market?",
        "",
        "| forecaster | Brier | log-loss | n |",
        "| :-- | --: | --: | --: |",
    ]
    for r in records(info):
        lines.append(f"| {r['model']} | {r['brier']:.4f} | {r['logloss']:.4f} | {r['n']:,} |")
    lines += [
        "",
        "## P&L of the simple rule",
        "",
        "One contract per signal, first signal per contract-day, held to settlement, taker fees.",
        "",
        "| margin | trades | hit rate | avg price | P&L total | P&L / trade | 95% CI (total) "
        "| daily Sharpe |",
        "| --: | --: | --: | --: | --: | --: | :--: | --: |",
    ]
    for r in records(summary):
        lines.append(
            f"| {r['margin']:.2f} | {int(r['trades'])} | {r['hit_rate']:.3f} "
            f"| {r['avg_price']:.2f} "
            f"| ${r['pnl_total']:+.2f} | ${r['pnl_per_trade']:+.3f} "
            f"| [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] | {r['sharpe_daily']:.2f} |"
        )
    lines += [
        "",
        "## P&L by station (margin 0.05)",
        "",
        "| station | trades | P&L total | P&L / trade |",
        "| :-- | --: | --: | --: |",
    ]
    for r in records(by_station):
        lines.append(
            f"| {r['station']} | {int(r['count'])} | ${r['sum']:+.2f} | ${r['mean']:+.3f} |"
        )
    return "\n".join(lines)
