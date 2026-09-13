"""Downloaders for the Iowa Environmental Mesonet (IEM) archives.

Three IEM services cover every input the model needs, all free and token-less:

* ``json/cli.py``          – parsed NWS Daily Climate Reports (CLI): the settlement labels.
* ``cgi-bin/request/asos.py`` – METAR archive (routine + SPECI), including the raw report.
* ``cgi-bin/request/mos.py``  – MOS / NBM text-bulletin archive: the forecast features.

Everything is cached as one Parquet file per (dataset, station, year) under ``raw_dir()``.
Completed years are never re-downloaded; the current year is refreshed when the cache is
older than :data:`REFRESH_AFTER`.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

from weatherbot.config import HISTORY_START, IEM_BASE, STATIONS, USER_AGENT, Station, raw_dir

log = logging.getLogger(__name__)

REFRESH_AFTER = timedelta(hours=6)
MosModel = Literal["NBS", "GFS"]

#: METAR fields requested from ``asos.py``. ``metar`` is the raw report, from which the
#: 0.1 °C temperature group and the 6-hourly max/min groups are parsed later.
METAR_FIELDS = (
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "gust",
    "alti",
    "mslp",
    "p01i",
    "vsby",
    "skyc1",
    "skyl1",
    "wxcodes",
    "metar",
)

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


def _get(url: str, params: dict[str, Any], *, tries: int = 4, timeout: int = 120) -> str:
    """GET with simple exponential backoff; IEM occasionally returns 5xx under load."""
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            resp = _session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == tries:
                raise
            log.warning("IEM request failed (%s); retry %d/%d in %.0fs", exc, attempt, tries, delay)
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def _cache_path(dataset: str, station: Station, year: int) -> Path:
    return raw_dir() / dataset / f"{station.icao}_{year}.parquet"


def _is_fresh(path: Path, year: int) -> bool:
    if not path.exists():
        return False
    if year < datetime.now(UTC).year:
        return True
    age = datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return age < REFRESH_AFTER


# --- CLI: daily climate reports (labels) -----------------------------------------------------


def fetch_cli(station: Station, year: int) -> pd.DataFrame:
    """Parsed CLI reports for one station-year; one row per climate day."""
    text = _get(f"{IEM_BASE}/json/cli.py", {"station": station.icao, "year": year})
    rows = pd.read_json(io.StringIO(text))["results"]
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=["station", "day", "high", "high_time", "low", "product"])
    keep = {
        "valid": "day",
        "high": "high",
        "high_time": "high_time",
        "high_normal": "high_normal",
        "high_record": "high_record",
        "low": "low",
        "low_time": "low_time",
        "precip": "precip",
        "snow": "snow",
        "product": "product",
        "wfo": "wfo",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    df.insert(0, "station", station.icao)
    df["day"] = pd.to_datetime(df["day"]).dt.date
    for col in ("high", "high_normal", "high_record", "low"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    for col in ("precip", "snow"):
        if col in df:  # "T" = trace
            df[col] = pd.to_numeric(df[col].replace({"T": 0.001}), errors="coerce").astype(
                "Float64"
            )
    return df.sort_values("day").reset_index(drop=True)


# --- METAR archive (observation features) ---------------------------------------------------


def fetch_metar(station: Station, start: date, end: date) -> pd.DataFrame:
    """All routine and special METARs in ``[start, end)`` with ``valid`` as tz-aware UTC."""
    params: dict[str, Any] = {
        "station": station.icao,
        "data": ",".join(METAR_FIELDS),
        "year1": start.year,
        "month1": start.month,
        "day1": start.day,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": [3, 4],  # 3 = routine (hourly), 4 = SPECI
    }
    text = _get(f"{IEM_BASE}/cgi-bin/request/asos.py", params)
    df = pd.read_csv(
        io.StringIO(text), na_values=["M"], keep_default_na=False, dtype={"metar": str}
    )
    if df.empty:
        return df
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["station"] = station.icao
    df = df.replace({"T": 0.005})  # trace precipitation
    for col in (
        "tmpf",
        "dwpf",
        "relh",
        "drct",
        "sknt",
        "gust",
        "alti",
        "mslp",
        "p01i",
        "vsby",
        "skyl1",
    ):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("valid").drop_duplicates(["valid", "metar"]).reset_index(drop=True)


# --- MOS / NBM bulletins (forecast features) ------------------------------------------------

MOS_FIELDS = {
    "NBS": ["runtime", "ftime", "tmp", "dpt", "sky", "wsp", "p06", "p12", "txn", "xnd", "tsd"],
    "GFS": ["runtime", "ftime", "tmp", "dpt", "cld", "wsp", "p06", "p12", "n_x"],
}


def fetch_mos(station: Station, model: MosModel, start: date, end: date) -> pd.DataFrame:
    """MOS/NBM rows for runs in ``[start, end)``; ``runtime``/``ftime`` tz-aware UTC."""
    params: dict[str, Any] = {
        "station": station.icao,
        "model": model,
        "sts": f"{start:%Y-%m-%d}T00:00Z",
        "ets": f"{end:%Y-%m-%d}T00:00Z",
        "format": "csv",
    }
    text = _get(f"{IEM_BASE}/cgi-bin/request/mos.py", params)
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "runtime" not in df:
        return pd.DataFrame(columns=["station", *MOS_FIELDS[model]])
    df = df[[c for c in MOS_FIELDS[model] if c in df.columns]].copy()
    df["runtime"] = pd.to_datetime(df["runtime"], utc=True)
    df["ftime"] = pd.to_datetime(df["ftime"], utc=True)
    df.insert(0, "station", station.icao)
    for col in df.columns.drop(["station", "runtime", "ftime"]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["runtime", "ftime"]).reset_index(drop=True)


# --- Cached loaders -------------------------------------------------------------------------


def _years(start: date, end: date) -> list[int]:
    return list(range(start.year, end.year + 1))


def _load_cached(
    dataset: str,
    station: Station,
    year: int,
    fetch: Callable[[], pd.DataFrame],
    *,
    force: bool,
) -> pd.DataFrame:
    path = _cache_path(dataset, station, year)
    if not force and _is_fresh(path, year):
        return pd.read_parquet(path)
    log.info("fetching %s %s %d", dataset, station.icao, year)
    df: pd.DataFrame = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    time.sleep(0.5)  # be polite to IEM
    return df


def load_cli(
    stations: tuple[Station, ...] = STATIONS,
    start: date = HISTORY_START,
    end: date | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    end = end or date.today()
    frames = [
        _load_cached("cli", s, y, partial(fetch_cli, s, y), force=force)
        for s in stations
        for y in _years(start, end)
    ]
    df = pd.concat(frames, ignore_index=True)
    return df[(df["day"] >= start) & (df["day"] <= end)].reset_index(drop=True)


def load_metar(
    stations: tuple[Station, ...] = STATIONS,
    start: date = HISTORY_START,
    end: date | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """METARs from the day before ``start`` (pre-day snapshots need it) through ``end``."""
    end = end or date.today()
    frames = []
    for s in stations:
        for y in _years(start - timedelta(days=1), end):
            y0, y1 = date(y, 1, 1), date(y + 1, 1, 1)
            frames.append(_load_cached("metar", s, y, partial(fetch_metar, s, y0, y1), force=force))
    return pd.concat(frames, ignore_index=True)


def load_mos(
    model: MosModel,
    stations: tuple[Station, ...] = STATIONS,
    start: date = HISTORY_START,
    end: date | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    end = end or date.today()
    frames = []
    for s in stations:
        for y in _years(start - timedelta(days=1), end):
            y0, y1 = date(y, 1, 1), date(y + 1, 1, 1)
            frames.append(
                _load_cached(
                    f"mos_{model}",
                    s,
                    y,
                    partial(fetch_mos, s, model, y0, y1),
                    force=force,
                )
            )
    return pd.concat(frames, ignore_index=True)
