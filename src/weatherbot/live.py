"""Real-time inference: current contract probabilities for every open Kalshi daily-high market.

The live path re-uses the exact feature code used for training — the same IEM endpoints
serve the archive and the last few hours — so there is no train/serve skew. Each run:

1. pulls the last 36 h of METARs, the last two days of NBM / GFS MOS bulletins and this
   year's CLI reports for a station;
2. builds one snapshot row per open climate day (today, and tomorrow once its market opens)
   at the current wall-clock time;
3. maps the predicted PMF onto every open contract and compares with the order book.

Signals are appended to ``live/signals.jsonl``. With ``--paper``, entries that clear the
edge margin are recorded to ``live/paper_trades.jsonl`` (one per contract-day) and settled
later by ``settle_paper`` against Kalshi's results. Nothing here places real orders.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from weatherbot.backtest import taker_fee
from weatherbot.config import STATIONS, USER_AGENT, Station
from weatherbot.config import station as get_station
from weatherbot.contracts import contract_probability, parse_contract
from weatherbot.data import iem, kalshi
from weatherbot.features import features_for_snapshots, snapshot_frame_at
from weatherbot.metar import c_to_f
from weatherbot.model import HighModel
from weatherbot.util import records

log = logging.getLogger(__name__)

LIVE_DIR = Path("live")
PAPER_MARGIN = 0.05
STALE_OBS_MINUTES = 90


# --- fresh inputs ---------------------------------------------------------------------------


def _nws_recent_metars(station: Station, hours: int = 36) -> pd.DataFrame:
    """Fallback observation source: api.weather.gov, reshaped like IEM's ``asos.py`` output."""
    resp = requests.get(
        f"https://api.weather.gov/stations/{station.icao}/observations",
        params={"limit": 200},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    rows = []
    for f in resp.json().get("features", []):
        p = f["properties"]
        t = p.get("temperature", {}).get("value")
        rows.append(
            {
                "station": station.icao,
                "valid": pd.Timestamp(p["timestamp"]),
                "tmpf": c_to_f(t) if t is not None else float("nan"),
                "dwpf": c_to_f(p["dewpoint"]["value"])
                if p.get("dewpoint", {}).get("value") is not None
                else float("nan"),
                "relh": p.get("relativeHumidity", {}).get("value"),
                "drct": p.get("windDirection", {}).get("value"),
                "sknt": (p.get("windSpeed", {}).get("value") or 0) / 1.852,
                "gust": (p.get("windGust", {}).get("value") or float("nan")) / 1.852,
                "alti": (p.get("barometricPressure", {}).get("value") or float("nan")) * 0.0002953,
                "mslp": float("nan"),
                "p01i": (p.get("precipitationLastHour", {}).get("value") or 0) / 25.4,
                "vsby": (p.get("visibility", {}).get("value") or float("nan")) / 1609.34,
                "skyc1": (p.get("cloudLayers") or [{}])[0].get("amount"),
                "skyl1": float("nan"),
                "wxcodes": p.get("presentWeather")
                and " ".join(w.get("rawString", "") for w in p["presentWeather"]),
                "metar": p.get("rawMessage") or "",
            }
        )
    df = pd.DataFrame(rows)
    cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(hours=hours)
    return df[df["valid"] >= cutoff].sort_values("valid").reset_index(drop=True)


def fresh_inputs(
    station: Station, now: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(cli, metar, nbs, gfs) covering the last two days up to ``now``."""
    today = now.date()
    start = today - timedelta(days=2)
    end = today + timedelta(days=1)
    metar = iem.fetch_metar(station, start, end)
    latest = metar["valid"].max() if not metar.empty else None
    if latest is None or (now - latest) > pd.Timedelta(minutes=STALE_OBS_MINUTES):
        log.warning(
            "%s: IEM METARs stale (latest %s); falling back to api.weather.gov",
            station.icao,
            latest,
        )
        try:
            metar = _nws_recent_metars(station)
        except requests.RequestException as exc:
            log.error("NWS fallback failed: %s", exc)
    nbs = iem.fetch_mos(station, "NBS", start, end)
    gfs = iem.fetch_mos(station, "GFS", start, end)
    cli = iem.fetch_cli(station, today.year)
    if today.month == 1 and today.day <= 3:
        cli = pd.concat([iem.fetch_cli(station, today.year - 1), cli], ignore_index=True)
    return cli, metar, nbs, gfs


# --- inference ------------------------------------------------------------------------------


def open_markets_by_day(station: Station) -> dict[date, list[dict[str, Any]]]:
    out: dict[date, list[dict[str, Any]]] = {}
    for m in kalshi.list_markets(station.kalshi_series, status="open"):
        day = kalshi.ticker_day(m["ticker"].split("-")[1])
        out.setdefault(day, []).append(m)
    return out


def _book_top(market: dict[str, Any]) -> tuple[float, float]:
    """(best bid, best ask) for YES in dollars from a market object; NaN when absent."""

    def dollars(key: str) -> float:
        v = market.get(f"{key}_dollars")
        if v not in (None, ""):
            return float(v)
        c = market.get(key)
        return float(c) / 100 if c is not None else float("nan")

    return dollars("yes_bid"), dollars("yes_ask")


def evaluate_station(model: HighModel, station: Station, now: pd.Timestamp) -> pd.DataFrame:
    """Model vs market for every open contract of one station, as of ``now``."""
    markets = open_markets_by_day(station)
    if not markets:
        return pd.DataFrame()
    cli, metar, nbs, gfs = fresh_inputs(station, now)
    days = sorted(markets)
    snaps = snapshot_frame_at(station, now, days)
    feats = features_for_snapshots(station, snaps, cli, metar, nbs, gfs)
    feats = feats.dropna(subset=["anchor"])
    if feats.empty:
        log.warning("%s: no anchor (no NBM/GFS forecast available)", station.icao)
        return pd.DataFrame()
    support, pmf = model.predict_high_pmf(feats)
    rows = []
    for i, snap in enumerate(records(feats)):
        for m in markets.get(snap["day"], []):
            try:
                contract = parse_contract(m)
            except ValueError as exc:
                log.warning("%s", exc)
                continue
            p = float(contract_probability(support[i : i + 1], pmf[i : i + 1], contract)[0])
            bid, ask = _book_top(m)
            rows.append(
                {
                    "asof": now.isoformat(),
                    "station": station.icao,
                    "day": str(snap["day"]),
                    "hour_lst": round(float(snap["hour_lst"]), 2),
                    "ticker": m["ticker"],
                    "contract": contract.label,
                    "p_model": round(p, 4),
                    "bid": bid,
                    "ask": ask,
                    "edge_yes": _edge(p - ask, ask),
                    "edge_no": _edge(bid - p, 1 - bid),
                    "run_max_f": _num(snap["run_max_f"]),
                    "cur_temp_f": _num(snap["cur_temp_f"]),
                    "nbs_txn": _num(snap["nbs_txn"]),
                    "anchor": float(snap["anchor"]),
                    "obs_age_min": _num(snap["cur_age_min"]),
                }
            )
    return pd.DataFrame(rows)


def _num(value: Any) -> float | None:
    """Float or ``None`` for NaN (keeps the JSONL log clean)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _edge(gross: float, price: float) -> float:
    """Expected edge after Kalshi's taker fee; NaN when there is no price."""
    if price != price:
        return float("nan")
    return round(gross - float(taker_fee(price)), 4)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


def record_paper_trades(
    signals: pd.DataFrame, margin: float = PAPER_MARGIN
) -> list[dict[str, Any]]:
    """Log one paper entry per contract-day whose edge clears ``margin`` (never re-enter)."""
    path = LIVE_DIR / "paper_trades.jsonl"
    seen: set[str] = set()
    if path.exists():
        seen = {
            json.loads(line)["ticker"] for line in path.read_text().splitlines() if line.strip()
        }
    trades = []
    for r in records(signals):
        if r["ticker"] in seen:
            continue
        edge_yes, edge_no = _num(r["edge_yes"]), _num(r["edge_no"])
        if edge_yes is not None and edge_yes > margin:
            side, price = "yes", float(r["ask"])
        elif edge_no is not None and edge_no > margin:
            side, price = "no", 1 - float(r["bid"])
        else:
            continue
        trades.append(
            {
                "asof": r["asof"],
                "ticker": r["ticker"],
                "station": r["station"],
                "day": r["day"],
                "contract": r["contract"],
                "side": side,
                "price": price,
                "fee": float(taker_fee(price)),
                "p_model": r["p_model"],
                "result": None,
                "pnl": None,
            }
        )
        seen.add(str(r["ticker"]))
    if trades:
        _append_jsonl(path, trades)
    return trades


def settle_paper() -> pd.DataFrame:
    """Fill in results for paper trades whose markets have settled; return the full ledger."""
    path = LIVE_DIR / "paper_trades.jsonl"
    if not path.exists():
        return pd.DataFrame()
    ledger = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed = False
    for t in ledger:
        if t.get("result") in ("yes", "no"):
            continue
        try:
            m = kalshi.get_market(t["ticker"])
        except requests.RequestException:
            continue
        if m.get("result") in ("yes", "no"):
            t["result"] = m["result"]
            won = t["side"] == m["result"]
            t["pnl"] = round(float(won) - t["price"] - t["fee"], 4)
            changed = True
    if changed:
        path.write_text("".join(json.dumps(t) + "\n" for t in ledger))
    df = pd.DataFrame(ledger)
    if not df.empty and df["pnl"].notna().any():
        settled = df.dropna(subset=["pnl"])
        log.info(
            "paper: %d settled, P&L $%+.2f, hit rate %.2f",
            len(settled),
            settled["pnl"].sum(),
            (settled["result"] == settled["side"]).mean(),
        )
    return df


def run_live(station: str | None = None, paper: bool = False, loop_seconds: int = 0) -> None:
    model = HighModel.load()
    stations = [get_station(station)] if station else list(STATIONS)
    while True:
        now = pd.Timestamp.now(tz=UTC).floor("min")
        frames = []
        for st in stations:
            try:
                frames.append(evaluate_station(model, st, now))
            except (requests.RequestException, ValueError) as exc:
                log.error("%s: %s", st.icao, exc)
        signals = (
            pd.concat([f for f in frames if not f.empty], ignore_index=True)
            if any(not f.empty for f in frames)
            else pd.DataFrame()
        )
        if signals.empty:
            print("no open markets / no signals")
        else:
            cols = [
                "station",
                "day",
                "hour_lst",
                "contract",
                "p_model",
                "bid",
                "ask",
                "edge_yes",
                "edge_no",
                "run_max_f",
                "cur_temp_f",
                "nbs_txn",
            ]
            with pd.option_context("display.width", 200, "display.max_rows", 500):
                print(f"\n=== {now:%Y-%m-%d %H:%M}Z ===")
                print(signals[cols].to_string(index=False))
            _append_jsonl(LIVE_DIR / "signals.jsonl", records(signals))
            if paper:
                new = record_paper_trades(signals)
                if new:
                    print(f"paper: recorded {len(new)} new entries")
                settle_paper()
        if loop_seconds <= 0:
            return
        time.sleep(loop_seconds)
