"""Static configuration: the eight stations, their Kalshi series, time conventions and paths.

Everything here was verified against live sources on 2026-09-12:

* Kalshi's ``KXHIGH*`` daily-high series settle on The Weather Company's republication of the
  NWS Daily Climate Report (CLI) for the station named in each market's rules (e.g. ``CLINYC``).
* The climate day is the local *standard* time calendar day (no DST). Kalshi's market close
  times (05Z for NYC, 06Z for Chicago, 07Z for Denver, ...) are exactly midnight LST.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class Station:
    """One ASOS station that has a Kalshi daily-high market."""

    icao: str
    """Four-letter ICAO identifier used by IEM and the NWS (e.g. ``KNYC``)."""
    city: str
    kalshi_series: str
    """Ticker of the Kalshi series whose markets settle on this station's CLI report."""
    cli_pil: str
    """AFOS product id of the Daily Climate Report (e.g. ``CLINYC``)."""
    utc_offset_std: int
    """UTC offset of local *standard* time, in hours (negative in the US)."""

    @property
    def lst(self) -> timezone:
        """Fixed-offset local standard time; the climate day never observes DST."""
        return timezone(timedelta(hours=self.utc_offset_std), name=f"LST{self.utc_offset_std:+d}")


STATIONS: tuple[Station, ...] = (
    Station("KNYC", "New York City", "KXHIGHNY", "CLINYC", -5),
    Station("KMDW", "Chicago (Midway)", "KXHIGHCHI", "CLIMDW", -6),
    Station("KMIA", "Miami", "KXHIGHMIA", "CLIMIA", -5),
    Station("KAUS", "Austin", "KXHIGHAUS", "CLIAUS", -6),
    Station("KHOU", "Houston (Hobby)", "KXHIGHTHOU", "CLIHOU", -6),
    Station("KDEN", "Denver", "KXHIGHDEN", "CLIDEN", -7),
    Station("KPHL", "Philadelphia", "KXHIGHPHIL", "CLIPHL", -5),
    Station("KLAX", "Los Angeles", "KXHIGHLAX", "CLILAX", -8),
)

STATIONS_BY_ICAO: dict[str, Station] = {s.icao: s for s in STATIONS}
STATIONS_BY_SERIES: dict[str, Station] = {s.kalshi_series: s for s in STATIONS}


def station(icao: str) -> Station:
    try:
        return STATIONS_BY_ICAO[icao.upper()]
    except KeyError:
        raise KeyError(f"unknown station {icao!r}; known: {sorted(STATIONS_BY_ICAO)}") from None


# --- Data range -----------------------------------------------------------------------------

#: First climate day in the training archive. IEM holds NBM ``NBS`` bulletins from 2019 on;
#: starting in 2020 keeps a clean year boundary and avoids the NBM v3.x → v4 transition period.
HISTORY_START = date(2020, 1, 1)

#: Snapshot hours (local standard time) at which features are computed for each climate day.
#: ``-6 .. -1`` are the evening before (markets open 10 am ET the day before); ``0 .. 23`` is the
#: day itself. Hourly snapshots match the ~hourly cadence of routine METARs.
SNAPSHOT_HOURS: tuple[int, ...] = tuple(range(-6, 24))

#: Modelled range of ``high - anchor`` in whole degrees, where the anchor is the NBM day-max
#: forecast. Residuals beyond the range are pooled into the edge classes (about 1% of days;
#: the open-ended Kalshi contracts are what pay on those tails anyway).
MIN_RESIDUAL_DEGREES = -15
MAX_RESIDUAL_DEGREES = 15


# --- Paths ----------------------------------------------------------------------------------


def data_dir() -> Path:
    """Root of the local raw/processed data cache (``$WEATHERBOT_DATA`` or ``./data``)."""
    return Path(os.environ.get("WEATHERBOT_DATA", "data")).expanduser()


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def models_dir() -> Path:
    return Path(os.environ.get("WEATHERBOT_MODELS", "models")).expanduser()


# --- Kalshi ---------------------------------------------------------------------------------

KALSHI_API = os.environ.get("KALSHI_API", "https://api.elections.kalshi.com/trade-api/v2")

#: Kalshi's published taker fee: ``ceil(0.07 * contracts * p * (1 - p))`` dollars, in cents.
KALSHI_TAKER_FEE_RATE = 0.07

#: IEM's public web services. Please be polite: they are run by one person at Iowa State.
IEM_BASE = "https://mesonet.agron.iastate.edu"
USER_AGENT = "weatherbot/1.0 (+https://github.com/SomeGuy966/weather-bot)"
