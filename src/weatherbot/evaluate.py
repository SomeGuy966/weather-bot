"""Score the held-out years: proper scoring rules, calibration, and contract-level reliability.

Every model is evaluated on the same integer-degree grid (``anchor + RESIDUALS``) so scores
are directly comparable:

* ``model``       – the calibrated LightGBM classifier;
* ``nbm``         – NBM's own guidance: ``N(txn, xnd)`` discretised, and once the day has
                    started, truncated below the running max (what a careful human does);
* ``persistence`` – the running max plus the training-set histogram of how much further the
                    day went from this hour (pre-day: the NBM forecast plus its error histogram);
* ``residual_hist`` – the NBM forecast plus its historical error histogram at every hour.

Contract-level scores use a synthetic Kalshi-style ladder centred on the NBM forecast, so
calibration can be measured on every historical day and not only on the two months Kalshi
keeps prices for.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from weatherbot.config import MIN_RESIDUAL_DEGREES
from weatherbot.contracts import standard_ladder
from weatherbot.features import load_snapshots
from weatherbot.model import N_CLASSES, RESIDUALS, HighModel, split_by_day
from weatherbot.util import records

log = logging.getLogger(__name__)


# --- baselines ------------------------------------------------------------------------------


def nbm_gaussian_pmf(frame: pd.DataFrame) -> np.ndarray:
    """Discretised ``N(txn, xnd)`` on the residual grid; truncated at the running max in-day."""
    mean = frame["nbs_txn"].fillna(frame["gfs_nx"]).fillna(frame["high_normal"]).to_numpy(float)
    sd = frame["nbs_xnd"].fillna(4.0).clip(lower=1.0).to_numpy(float)
    sd = np.where(frame["nbs_txn"].isna(), 5.0, sd)
    anchor = frame["anchor"].to_numpy(float)
    grid = anchor[:, None] + RESIDUALS[None, :]  # (n, K)
    upper = norm.cdf((grid + 0.5 - mean[:, None]) / sd[:, None])
    lower = norm.cdf((grid - 0.5 - mean[:, None]) / sd[:, None])
    pmf = upper - lower
    pmf[:, 0] = upper[:, 0]  # pool the tails into the edge classes
    pmf[:, -1] = 1.0 - lower[:, -1]
    # In-day: the high cannot end below the running max (allow one degree of rounding slack).
    run_max = frame["run_max_f"].to_numpy(float)
    floor = np.where(np.isnan(run_max), -np.inf, np.round(run_max) - 1)
    pmf = np.where(grid < floor[:, None], 0.0, pmf)
    pmf = np.clip(pmf, 1e-9, None)
    return pmf / pmf.sum(axis=1, keepdims=True)


@dataclass
class ResidualHistogram:
    """Empirical distribution of ``high - reference`` given the hour, from the training split.

    With ``reference="anchor"`` this is the NBM forecast plus its historical error spread.
    With ``reference="run_max"`` it is *persistence*: the running max plus the historical
    distribution of how much further the day went from this hour — the rule of thumb a
    disciplined human trader applies in-day (falls back to the forecast version pre-day).
    Either way the result is expressed on the model's ``anchor + RESIDUALS`` grid.
    """

    reference: str
    table: dict[int, np.ndarray]
    fallback: np.ndarray
    offsets: np.ndarray

    @classmethod
    def fit(cls, train: pd.DataFrame, reference: str = "anchor") -> ResidualHistogram:
        ref = train[reference].round()
        resid = (train["high"] - ref).to_numpy(float)
        ok = ~np.isnan(resid)
        offsets = np.arange(-30, 41)  # wide enough for persistence residuals
        idx = np.clip(np.round(resid[ok]).astype(int) - offsets[0], 0, len(offsets) - 1)
        hours = train["hour_lst"].astype(int).to_numpy()[ok]
        table: dict[int, np.ndarray] = {}
        for hour in np.unique(hours):
            counts = np.bincount(idx[hours == hour], minlength=len(offsets)).astype(float) + 0.1
            table[int(hour)] = counts / counts.sum()
        counts = np.bincount(idx, minlength=len(offsets)).astype(float) + 0.1
        return cls(reference, table, counts / counts.sum(), offsets)

    def pmf(self, frame: pd.DataFrame, fallback: ResidualHistogram | None = None) -> np.ndarray:
        """Histogram re-expressed on the ``anchor + RESIDUALS`` grid (edge classes pool tails)."""
        hours = frame["hour_lst"].astype(int).to_numpy()
        ref = frame[self.reference].round().to_numpy(float)
        anchor = frame["anchor"].to_numpy(float)
        out = np.zeros((len(frame), N_CLASSES))
        for i in range(len(frame)):
            if np.isnan(ref[i]):
                if fallback is None:
                    raise ValueError("reference missing and no fallback histogram given")
                out[i] = fallback.pmf(frame.iloc[i : i + 1])[0]
                continue
            hist = self.table.get(int(hours[i]), self.fallback)
            # shift: value = ref + offset  →  class = value - anchor - MIN_RESIDUAL
            classes = (ref[i] + self.offsets - anchor[i] - MIN_RESIDUAL_DEGREES).astype(int)
            classes = np.clip(classes, 0, N_CLASSES - 1)
            np.add.at(out[i], classes, hist)
        return out


# --- scoring --------------------------------------------------------------------------------


def _cdf(pmf: np.ndarray) -> np.ndarray:
    return np.cumsum(pmf, axis=1)


def scores(pmf: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """NLL, ranked probability score, Brier, MAE of the median, central-interval coverage."""
    n = len(y)
    idx = np.arange(n)
    p_true = np.clip(pmf[idx, y], 1e-12, 1.0)
    cdf = _cdf(pmf)
    step = (np.arange(N_CLASSES)[None, :] >= y[:, None]).astype(float)
    rps = ((cdf - step) ** 2).sum(axis=1)
    onehot = np.zeros_like(pmf)
    onehot[idx, y] = 1.0
    brier = ((pmf - onehot) ** 2).sum(axis=1)
    median = (cdf >= 0.5).argmax(axis=1)
    lo50, hi50 = (cdf >= 0.25).argmax(axis=1), (cdf >= 0.75).argmax(axis=1)
    lo90, hi90 = (cdf >= 0.05).argmax(axis=1), (cdf >= 0.95).argmax(axis=1)
    return {
        "nll": float(-np.log(p_true).mean()),
        "rps": float(rps.mean()),
        "brier": float(brier.mean()),
        "mae_median": float(np.abs(median - y).mean()),
        "exact_hit": float((median == y).mean()),
        "cov50": float(((y >= lo50) & (y <= hi50)).mean()),
        "cov90": float(((y >= lo90) & (y <= hi90)).mean()),
        "width90": float((hi90 - lo90).mean() + 1),
        "n": float(n),
    }


def pit_values(pmf: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Randomised PIT for a discrete forecast (uniform on [0,1] iff calibrated)."""
    cdf = _cdf(pmf)
    idx = np.arange(len(y))
    upper = cdf[idx, y]
    lower = upper - pmf[idx, y]
    rng = np.random.default_rng(0)
    pit: np.ndarray = lower + rng.uniform(size=len(y)) * (upper - lower)
    return pit


def contract_scores(
    frame: pd.DataFrame, pmfs: dict[str, np.ndarray], bins: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier / log-loss / ECE of YES-probabilities on a synthetic Kalshi ladder per snapshot.

    Returns (summary per model, reliability table per model × bin).
    """
    center = (
        frame["nbs_txn"]
        .fillna(frame["gfs_nx"])
        .fillna(frame["high_normal"])
        .round()
        .to_numpy(float)
    )
    high = frame["high"].to_numpy(float)
    support = frame["anchor"].to_numpy(float)[:, None] + RESIDUALS[None, :]
    rows = []
    rel_rows = []
    for name, pmf in pmfs.items():
        probs: list[np.ndarray] = []
        outcomes: list[np.ndarray] = []
        # Vectorise over the six ladder positions: each contract depends on the row's centre.
        for k in range(6):
            lows = np.empty(len(frame))
            highs = np.empty(len(frame))
            for i, c in enumerate(center):
                contract = standard_ladder(int(c))[k]
                lows[i], highs[i] = contract.low, contract.high
            inside = (support >= lows[:, None]) & (support <= highs[:, None])
            p = (pmf * inside).sum(axis=1)
            # open-ended tails beyond the support edge pool the edge class
            p = np.where(np.isinf(highs) & (lows > support[:, -1]), pmf[:, -1], p)
            p = np.where(np.isinf(lows) & (highs < support[:, 0]), pmf[:, 0], p)
            probs.append(p)
            outcomes.append(((high >= lows) & (high <= highs)).astype(float))
        p_all = np.concatenate(probs)
        o_all = np.concatenate(outcomes)
        p_clip = np.clip(p_all, 1e-6, 1 - 1e-6)
        brier = float(((p_all - o_all) ** 2).mean())
        logloss = float(-(o_all * np.log(p_clip) + (1 - o_all) * np.log(1 - p_clip)).mean())
        edges = np.linspace(0, 1, bins + 1)
        which = np.clip(np.digitize(p_all, edges) - 1, 0, bins - 1)
        ece = 0.0
        for b in range(bins):
            m = which == b
            if m.any():
                gap = abs(p_all[m].mean() - o_all[m].mean())
                ece += gap * m.mean()
                rel_rows.append(
                    {
                        "model": name,
                        "bin": b,
                        "p_mean": p_all[m].mean(),
                        "freq": o_all[m].mean(),
                        "n": int(m.sum()),
                    }
                )
        rows.append(
            {
                "model": name,
                "brier": brier,
                "logloss": logloss,
                "ece": ece,
                "n_contracts": len(p_all),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rel_rows)


# --- driver ---------------------------------------------------------------------------------


def _by_group(
    frame: pd.DataFrame, pmfs: dict[str, np.ndarray], y: np.ndarray, key: str
) -> pd.DataFrame:
    rows = []
    for value, idx in frame.groupby(key).indices.items():
        for name, pmf in pmfs.items():
            s = scores(pmf[idx], y[idx])
            rows.append({key: value, "model": name, **s})
    return pd.DataFrame(rows)


def run_evaluation(results_dir: str = "results", figures_dir: str = "figures") -> None:
    from weatherbot import plotting

    model = HighModel.load()
    table = load_snapshots()
    train_end = date.fromisoformat(model.train_days[1])
    calib_end = date.fromisoformat(model.calib_days[1])
    split = split_by_day(table, train_end, calib_end)
    test = split.test.reset_index(drop=True)
    log.info("test split: %d rows, %s → %s", len(test), test["day"].min(), test["day"].max())

    y = (test["target"].to_numpy() - MIN_RESIDUAL_DEGREES).astype(int)
    fc_hist = ResidualHistogram.fit(split.train, "anchor")
    persist = ResidualHistogram.fit(split.train, "run_max_f")
    pmfs = {
        "model": model.predict_residual_pmf(test),
        "nbm": nbm_gaussian_pmf(test),
        "persistence": persist.pmf(test, fallback=fc_hist),
        "residual_hist": fc_hist.pmf(test),
    }

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = Path(figures_dir)
    fig.mkdir(parents=True, exist_ok=True)

    overall = pd.DataFrame([{"model": k, **scores(v, y)} for k, v in pmfs.items()])
    overall.to_csv(out / "metrics_overall.csv", index=False)
    by_hour = _by_group(test, pmfs, y, "hour_lst")
    by_hour.to_csv(out / "metrics_by_hour.csv", index=False)
    by_station = _by_group(test, pmfs, y, "station")
    by_station.to_csv(out / "metrics_by_station.csv", index=False)
    csum, rel = contract_scores(test, pmfs)
    csum.to_csv(out / "contract_metrics.csv", index=False)
    rel.to_csv(out / "contract_reliability.csv", index=False)

    importance = pd.DataFrame(
        {"feature": model.features, "gain": model.booster.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    importance["share"] = importance["gain"] / importance["gain"].sum()
    importance.to_csv(out / "feature_importance.csv", index=False)

    pit = pit_values(pmfs["model"], y)
    plotting.reliability(rel, fig / "reliability.png")
    plotting.by_hour(by_hour, fig / "skill_by_hour.png")
    plotting.pit_histogram(pit, fig / "pit.png")
    plotting.feature_importance(importance.head(20), fig / "feature_importance.png")
    plotting.example_day(test, model, fig / "example_day.png")

    summary = _summary_markdown(model, test, overall, by_hour, csum, importance)
    (out / "summary.md").write_text(summary)
    (out / "model_meta.json").write_text(json.dumps(model.meta, indent=2, default=str))
    log.info("wrote %s and %s", out, fig)
    print(summary)


def _summary_markdown(
    model: HighModel,
    test: pd.DataFrame,
    overall: pd.DataFrame,
    by_hour: pd.DataFrame,
    csum: pd.DataFrame,
    importance: pd.DataFrame,
) -> str:
    def table(df: pd.DataFrame, cols: list[str], fmt: str = "{:.3f}") -> str:
        head = (
            "| "
            + " | ".join(cols)
            + " |\n|"
            + "|".join([" :-- "] + [" --: "] * (len(cols) - 1))
            + "|\n"
        )
        body = ""
        for r in records(df):
            cells = [str(r[cols[0]])] + [
                fmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols[1:]
            ]
            body += "| " + " | ".join(cells) + " |\n"
        return head + body

    lines = [
        "# Evaluation summary",
        "",
        f"Test window: {test['day'].min()} → {test['day'].max()}, "
        f"{test['day'].nunique():,} station-days, {len(test):,} snapshots.",
        f"Train: {model.train_days[0]} → {model.train_days[1]} "
        f"({model.meta['train_rows']:,} rows); "
        f"calibration: {model.calib_days[0]} → {model.calib_days[1]} "
        f"({model.meta['calib_rows']:,} rows). "
        f"LightGBM best iteration {model.meta['best_iteration']}, "
        f"softmax temperature {model.temperature:.3f}.",
        "",
        "## Whole-distribution scores (integer-degree PMF, all snapshots)",
        "",
        table(
            overall,
            [
                "model",
                "nll",
                "rps",
                "brier",
                "mae_median",
                "exact_hit",
                "cov50",
                "cov90",
                "width90",
            ],
        ),
        "## Contract-level scores (synthetic Kalshi ladder around the NBM forecast)",
        "",
        table(csum, ["model", "brier", "logloss", "ece"]),
        "## Skill by hour of the climate day (ranked probability score)",
        "",
    ]
    piv = by_hour.pivot(index="hour_lst", columns="model", values="rps").reset_index()
    piv["model_vs_nbm"] = piv["model"] / piv["nbm"]
    piv["model_vs_persistence"] = piv["model"] / piv["persistence"]
    lines.append(
        table(
            piv,
            [
                "hour_lst",
                "model",
                "nbm",
                "persistence",
                "residual_hist",
                "model_vs_nbm",
                "model_vs_persistence",
            ],
        )
    )
    lines += [
        "## Top features (LightGBM gain share)",
        "",
        table(importance.head(15), ["feature", "share"]),
    ]
    return "\n".join(lines)
