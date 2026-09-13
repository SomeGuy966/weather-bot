"""Read-only Kalshi client and a local archive of the eight daily-high series.

Everything here uses the public, unauthenticated market-data endpoints. Kalshi only keeps
recent markets (the ``KXHIGH*`` series start in July 2026 and earlier generations are gone),
so the archive is append-only: run ``weatherbot kalshi`` regularly to accumulate history.

Archive layout (Parquet under ``data/kalshi/``)::

    markets.parquet   one row per market: strikes, open/close, result, rules text
    candles.parquet   hourly OHLC of price / yes_bid / yes_ask, volume, open interest
    trades.parquet    the full public trade tape
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from weatherbot.config import KALSHI_API, STATIONS, USER_AGENT, data_dir
from weatherbot.util import records

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT

MARKET_COLUMNS = [
    "ticker",
    "event_ticker",
    "series_ticker",
    "status",
    "result",
    "strike_type",
    "floor_strike",
    "cap_strike",
    "yes_sub_title",
    "open_time",
    "close_time",
    "expiration_time",
    "rules_primary",
]


def _get(path: str, params: dict[str, Any] | None = None, *, tries: int = 4) -> dict[str, Any]:
    delay = 1.0
    for attempt in range(1, tries + 1):
        try:
            resp = _session.get(f"{KALSHI_API}{path}", params=params, timeout=60)
            if resp.status_code == 429:
                raise requests.HTTPError("rate limited", response=resp)
            resp.raise_for_status()
            return dict(resp.json())
        except requests.RequestException as exc:
            if attempt == tries:
                raise
            log.warning("Kalshi %s failed (%s); retry in %.0fs", path, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


# --- live reads -----------------------------------------------------------------------------


def list_markets(series_ticker: str, status: str | None = None) -> list[dict[str, Any]]:
    """All markets of a series (paginated), optionally filtered by status."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"series_ticker": series_ticker, "limit": 1000}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        page = _get("/markets", params)
        out.extend(page.get("markets", []))
        cursor = page.get("cursor")
        if not cursor or not page.get("markets"):
            return out


def get_market(ticker: str) -> dict[str, Any]:
    return dict(_get(f"/markets/{ticker}")["market"])


def get_orderbook(ticker: str, depth: int = 5) -> dict[str, Any]:
    return dict(_get(f"/markets/{ticker}/orderbook", {"depth": depth}).get("orderbook", {}))


def get_candles(
    series_ticker: str, ticker: str, start: datetime, end: datetime, minutes: int = 60
) -> pd.DataFrame:
    page = _get(
        f"/series/{series_ticker}/markets/{ticker}/candlesticks",
        {
            "start_ts": int(start.timestamp()),
            "end_ts": int(end.timestamp()),
            "period_interval": minutes,
        },
    )
    rows = []
    for c in page.get("candlesticks", []):
        price, bid, ask = c.get("price", {}), c.get("yes_bid", {}), c.get("yes_ask", {})
        rows.append(
            {
                "ticker": ticker,
                "end_ts": pd.Timestamp(int(c["end_period_ts"]), unit="s", tz="UTC"),
                "open": _dollars(price.get("open_dollars")),
                "high": _dollars(price.get("high_dollars")),
                "low": _dollars(price.get("low_dollars")),
                "close": _dollars(price.get("close_dollars")),
                "mean": _dollars(price.get("mean_dollars")),
                "bid_open": _dollars(bid.get("open_dollars")),
                "bid_close": _dollars(bid.get("close_dollars")),
                "ask_open": _dollars(ask.get("open_dollars")),
                "ask_close": _dollars(ask.get("close_dollars")),
                "volume": float(c.get("volume_fp") or 0),
                "open_interest": float(c.get("open_interest_fp") or 0),
            }
        )
    return pd.DataFrame(rows)


def get_trades(ticker: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = _get("/markets/trades", params)
        for t in page.get("trades", []):
            rows.append(
                {
                    "ticker": ticker,
                    "trade_id": t["trade_id"],
                    "created_time": pd.Timestamp(t["created_time"]),
                    "yes_price": _dollars(t.get("yes_price_dollars")),
                    "count": float(t.get("count_fp") or 0),
                    "taker_side": t.get("taker_side"),
                }
            )
        cursor = page.get("cursor")
        if not cursor or not page.get("trades"):
            break
    return pd.DataFrame(rows)


def _dollars(text: str | None) -> float:
    return float(text) if text not in (None, "") else float("nan")


# --- archive --------------------------------------------------------------------------------


def archive_dir() -> Path:
    return data_dir() / "kalshi"


def _merge_parquet(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    if not new.empty:
        new = new.drop_duplicates(keys, keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def markets_frame(markets: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame([{k: m.get(k) for k in MARKET_COLUMNS} for m in markets])
    if df.empty:
        return df
    for col in ("open_time", "close_time", "expiration_time"):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    df["series_ticker"] = df["ticker"].str.split("-").str[0]
    df["day"] = df["ticker"].str.split("-").str[1].map(ticker_day)
    return df


def ticker_day(code: str) -> date:
    """``26SEP11`` → ``date(2026, 9, 11)``."""
    return datetime.strptime(code, "%y%b%d").date()


def archive_all(with_trades: bool = True, series: list[str] | None = None) -> None:
    """Fetch every market of the eight series plus candles (and trades) for new ones."""
    series = series or [s.kalshi_series for s in STATIONS]
    all_markets: list[dict[str, Any]] = []
    for ticker in series:
        ms = list_markets(ticker)
        log.info("%s: %d markets", ticker, len(ms))
        all_markets.extend(ms)
    markets = _merge_parquet(
        archive_dir() / "markets.parquet", markets_frame(all_markets), ["ticker"]
    )

    candles_path = archive_dir() / "candles.parquet"
    done: set[str] = set()
    if candles_path.exists():
        existing = pd.read_parquet(candles_path)
        # re-fetch anything not yet settled at last archive time
        settled = set(markets.loc[markets["status"].isin(["settled", "finalized"]), "ticker"])
        done = set(existing["ticker"]) & settled
    todo = records(markets[~markets["ticker"].isin(done)])
    log.info("fetching candles for %d markets", len(todo))
    frames = []
    now = pd.Timestamp.now(tz=UTC)
    for i, row in enumerate(todo, 1):
        open_time, close_time = pd.Timestamp(row["open_time"]), pd.Timestamp(row["close_time"])
        end = min(close_time, now) + pd.Timedelta(hours=1)
        try:
            frames.append(
                get_candles(
                    row["series_ticker"],
                    row["ticker"],
                    open_time.to_pydatetime(),
                    end.to_pydatetime(),
                )
            )
        except requests.RequestException as exc:
            log.warning("candles %s: %s", row["ticker"], exc)
        if i % 50 == 0:
            log.info("  %d/%d", i, len(todo))
        time.sleep(0.4)
    if frames:
        _merge_parquet(candles_path, pd.concat(frames, ignore_index=True), ["ticker", "end_ts"])

    if with_trades:
        trades_path = archive_dir() / "trades.parquet"
        done_t: set[str] = set()
        if trades_path.exists():
            settled = set(markets.loc[markets["status"].isin(["settled", "finalized"]), "ticker"])
            done_t = set(pd.read_parquet(trades_path)["ticker"]) & settled
        todo_t = records(markets[~markets["ticker"].isin(done_t)])
        log.info("fetching trades for %d markets", len(todo_t))
        frames = []
        for i, row in enumerate(todo_t, 1):
            try:
                frames.append(get_trades(row["ticker"]))
            except requests.RequestException as exc:
                log.warning("trades %s: %s", row["ticker"], exc)
            if i % 50 == 0:
                log.info("  %d/%d", i, len(todo_t))
            time.sleep(0.4)
        if frames:
            _merge_parquet(trades_path, pd.concat(frames, ignore_index=True), ["trade_id"])


def load_archive() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = archive_dir()
    markets = pd.read_parquet(d / "markets.parquet")
    candles = (
        pd.read_parquet(d / "candles.parquet")
        if (d / "candles.parquet").exists()
        else pd.DataFrame()
    )
    trades = (
        pd.read_parquet(d / "trades.parquet") if (d / "trades.parquet").exists() else pd.DataFrame()
    )
    return markets, candles, trades
