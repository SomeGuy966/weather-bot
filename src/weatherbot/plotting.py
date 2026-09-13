"""Figures for the README. Matplotlib only, Agg backend, one function per figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weatherbot.model import HighModel

COLORS = {
    "model": "#1f5fbf",
    "nbm": "#d0782b",
    "persistence": "#3a9d5d",
    "residual_hist": "#9a9a9a",
    "market": "#2e8b57",
    "blend": "#7b4fbf",
}
LABELS = {
    "model": "LightGBM (calibrated)",
    "nbm": "NBM N(txn, xnd), truncated at running max",
    "persistence": "Running max + historical residual",
    "residual_hist": "NBM forecast + historical residual",
    "market": "Kalshi mid",
    "blend": "50/50 model + market",
}


def reliability(rel: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="perfect")
    for name, grp in rel.groupby("model"):
        grp = grp.sort_values("p_mean")
        ax.plot(
            grp["p_mean"],
            grp["freq"],
            marker="o",
            ms=4,
            color=COLORS.get(str(name)),
            label=LABELS.get(str(name), str(name)),
        )
    ax.set_xlabel("forecast P(YES)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Contract-level reliability (test years)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_hour(by_hour: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, grp in by_hour.groupby("model"):
        grp = grp.sort_values("hour_lst")
        axes[0].plot(
            grp["hour_lst"],
            grp["rps"],
            color=COLORS.get(str(name)),
            label=LABELS.get(str(name), str(name)),
        )
        axes[1].plot(
            grp["hour_lst"],
            grp["cov90"],
            color=COLORS.get(str(name)),
            label=LABELS.get(str(name), str(name)),
        )
    axes[0].set_ylabel("ranked probability score (lower = better)")
    axes[1].axhline(0.9, color="black", lw=1, ls="--")
    axes[1].set_ylabel("coverage of the 90% interval")
    for ax in axes:
        ax.set_xlabel("hour of climate day (local standard time; <0 = evening before)")
        ax.axvline(0, color="grey", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Skill through the day")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pit_histogram(pit: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(pit, bins=20, range=(0, 1), color=COLORS["model"], edgecolor="white")
    ax.axhline(len(pit) / 20, color="black", lw=1, ls="--")
    ax.set_xlabel("randomised PIT")
    ax.set_ylabel("count")
    ax.set_title("PIT histogram (flat = calibrated)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importance(imp: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    imp = imp.iloc[::-1]
    ax.barh(imp["feature"], imp["share"], color=COLORS["model"])
    ax.set_xlabel("share of total gain")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_day(test: pd.DataFrame, model: HighModel, path: Path, station: str = "KNYC") -> None:
    """Predictive distribution of one day's high, hour by hour, against the realised value."""
    days = test.loc[test["station"] == station, "day"]
    if days.empty:
        return
    # Pick a summer day with a large NBM error so the plot shows the model doing work.
    sub = test[(test["station"] == station) & (test["hour_lst"] == 8)].copy()
    sub["err"] = (sub["high"] - sub["nbs_txn"]).abs()
    day = sub.sort_values("err", ascending=False)["day"].iloc[min(3, len(sub) - 1)]
    rows = test[(test["station"] == station) & (test["day"] == day)].sort_values("hour_lst")
    support, pmf = model.predict_high_pmf(rows)
    cdf = np.cumsum(pmf, axis=1)
    q = {}
    for name, level in (("q05", 0.05), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95)):
        q[name] = support[np.arange(len(rows)), (cdf >= level).argmax(axis=1)]
    h = rows["hour_lst"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(h, q["q05"], q["q95"], color=COLORS["model"], alpha=0.15, label="model 5–95%")
    ax.fill_between(h, q["q25"], q["q75"], color=COLORS["model"], alpha=0.35, label="model 25–75%")
    ax.plot(h, q["q50"], color=COLORS["model"], label="model median")
    ax.plot(h, rows["nbs_txn"], color=COLORS["nbm"], ls="--", label="NBM txn")
    ax.plot(h, rows["run_max_f"], color="black", lw=1, label="running max (METAR)")
    ax.plot(h, rows["cur_temp_f"], color="grey", lw=0.8, ls=":", label="current temp")
    ax.axhline(
        float(rows["high"].iloc[0]),
        color="red",
        lw=1.2,
        label=f"CLI high = {rows['high'].iloc[0]:.0f}°F",
    )
    ax.set_xlabel("hour of climate day (LST)")
    ax.set_ylabel("°F")
    ax.set_title(f"{station} {day}: predictive distribution of the daily high through the day")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def backtest_curves(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, grp in daily.groupby("strategy"):
        ax.plot(pd.to_datetime(grp["day"]), grp["pnl"].cumsum(), label=str(name))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative P&L ($ per 1 contract/signal)")
    ax.set_title("Backtest on archived Kalshi prices (after taker fees)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def market_vs_model(rel: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--")
    for name, grp in rel.groupby("model"):
        grp = grp.sort_values("p_mean")
        ax.plot(
            grp["p_mean"],
            grp["freq"],
            marker="o",
            ms=4,
            color=COLORS.get(str(name)),
            label=LABELS.get(str(name), str(name)),
        )
    ax.set_xlabel("P(YES): model / Kalshi mid")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability on real Kalshi contracts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
