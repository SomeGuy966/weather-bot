"""Build the snapshot feature table: one row per (station, climate day, snapshot hour).

A *snapshot* is the information state at a wall-clock instant ``t``: every METAR received at
or before ``t``, and every MOS/NBM bulletin whose run was published by ``t``. Features are
computed with as-of joins on ``t`` so nothing from the future can leak into a row. The label
is the NWS CLI high for the climate day, which is what Kalshi settles on.

The model target is expressed relative to an *anchor* and the classifier learns
``target = clip(high - anchor, MIN_RESIDUAL, MAX_RESIDUAL)``:

* before the day starts, and while the running maximum is still far below the forecast, the
  anchor is the latest available NBM day-max forecast (``txn``; GFS MOS, then climatology as
  fallbacks) — residuals are centred near zero and the class grid covers forecast busts;
* once the running maximum is *in play* (within :data:`OBS_ANCHOR_WITHIN` °F of the forecast,
  or from :data:`OBS_ANCHOR_FROM_HOUR` LST), the anchor is the running maximum itself — the
  most common outcome ("the high is already in") becomes class 0, which is what lets the
  distribution collapse cleanly in the evening even on days the forecast busted badly.

Which regime a row is in is itself a feature (``anchor_is_obs``).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from weatherbot.config import (
    HISTORY_START,
    MAX_RESIDUAL_DEGREES,
    MIN_RESIDUAL_DEGREES,
    SNAPSHOT_HOURS,
    STATIONS,
    Station,
    processed_dir,
)
from weatherbot.data import iem
from weatherbot.metar import parse_temp_columns

log = logging.getLogger(__name__)

#: Assumed delay between a bulletin's nominal run time and its availability. NBM text
#: bulletins land within the hour; GFS MOS (MAV) bulletins arrive ~3 h after the model run.
NBS_LAG = pd.Timedelta(hours=1)
GFS_LAG = pd.Timedelta(hours=3, minutes=30)

#: Time after local midnight by which yesterday's CLI report is assumed published.
CLI_PUBLISHED_AFTER_HOURS = 4

#: Switch the anchor from the forecast to the running max once the latter is within this many
#: °F of the forecast, or from this hour (LST) regardless.
OBS_ANCHOR_WITHIN = 8.0
OBS_ANCHOR_FROM_HOUR = 10

_SKY_FRACTION = {
    "CLR": 0.0,
    "SKC": 0.0,
    "NSC": 0.0,
    "FEW": 0.2,
    "SCT": 0.45,
    "BKN": 0.75,
    "OVC": 1.0,
    "VV": 1.0,
}
_PRECIP_CODES = ("RA", "DZ", "SN", "TS", "SH", "GR", "GS", "PL", "IC", "UP", "FZ")

FEATURE_COLUMNS: tuple[str, ...] = (
    # calendar / clock
    "station_id",
    "hour_lst",
    "doy_sin",
    "doy_cos",
    # anchor bookkeeping
    "anchor",
    "anchor_is_nbm",
    "anchor_is_obs",
    # observations so far today
    "obs_count",
    "run_max_f",
    "run_max6_f",
    "run_max_minus_anchor",
    "cur_temp_f",
    "cur_temp_precise",
    "cur_dewpoint_f",
    "cur_depression_f",
    "cur_relh",
    "cur_sknt",
    "cur_gust",
    "cur_vsby",
    "cur_sky",
    "cur_p01i",
    "cur_precip",
    "cur_age_min",
    "d_temp_1h",
    "d_temp_3h",
    "d_alti_3h",
    "cur_minus_runmax",
    # yesterday
    "yday_metar_max_f",
    "yday_cli_high",
    "yday_cli_minus_metar",
    "high_normal",
    # NBM (NBS bulletin)
    "nbs_txn",
    "nbs_xnd",
    "nbs_txn_age_h",
    "nbs_txn_minus_anchor",
    "nbs_tmp_max_today",
    "nbs_tmp_max_remaining",
    "nbs_tmp_next",
    "nbs_sky_remaining",
    "nbs_p06_remaining",
    "nbs_run_age_h",
    # GFS MOS (MAV bulletin)
    "gfs_nx",
    "gfs_nx_minus_anchor",
    "gfs_nx_age_h",
)


# --- helpers --------------------------------------------------------------------------------


def climate_day_start(station: Station, day: date) -> datetime:
    """UTC instant at which ``day`` begins in the station's local standard time."""
    return datetime(day.year, day.month, day.day, tzinfo=station.lst).astimezone(UTC)


def _snapshot_frame(station: Station, days: pd.Series) -> pd.DataFrame:
    """Cartesian product of climate days × snapshot hours for one station."""
    day0 = pd.to_datetime(days).dt.tz_localize(station.lst).dt.tz_convert("UTC")
    frames = []
    for h in SNAPSHOT_HOURS:
        frames.append(
            pd.DataFrame(
                {
                    "station": station.icao,
                    "day": days.to_numpy(),
                    "day_start": day0.to_numpy(),
                    "hour_lst": h,
                    "t": (day0 + pd.Timedelta(hours=h)).to_numpy(),
                }
            )
        )
    snaps = pd.concat(frames, ignore_index=True)
    snaps["day_end"] = snaps["day_start"] + pd.Timedelta(hours=24)
    return snaps.sort_values("t").reset_index(drop=True)


def _prepare_metar(station: Station, metar: pd.DataFrame) -> pd.DataFrame:
    """Decode temperatures, drop delayed backfill rows, and tag each ob with its climate day."""
    m = metar[metar["station"] == station.icao].copy()
    m = m[~m["metar"].fillna("").str.contains("IEM_GHCNH")]  # not in the live feed
    m = m.sort_values("valid").reset_index(drop=True)
    temps = parse_temp_columns(m["metar"])
    m = pd.concat([m, temps], axis=1)
    m["temp_f"] = m["temp_f"].fillna(m["tmpf"])
    m["dewpoint_f"] = m["dewpoint_f"].fillna(m["dwpf"])
    m = m.dropna(subset=["temp_f"])

    local = m["valid"].dt.tz_convert(station.lst)
    m["day"] = local.dt.date
    m["day_start"] = local.dt.normalize().dt.tz_convert("UTC")

    # 6-hourly max group in the report near synoptic hour H covers (H-6h, H]. It is a clean
    # in-day checkpoint only when the whole window falls inside this climate day.
    synoptic = m["valid"].dt.round("h")
    window_start = synoptic - pd.Timedelta(hours=6)
    clean = (window_start >= m["day_start"]) & (synoptic <= m["day_start"] + pd.Timedelta(hours=24))
    m["max6_clean_f"] = m["max6_f"].where(clean)

    m["sky"] = m["skyc1"].map(_SKY_FRACTION).astype(float)
    wx = m["wxcodes"].fillna("").astype(str)
    m["precip"] = wx.apply(lambda s: float(any(code in s for code in _PRECIP_CODES)))

    grp = m.groupby("day", sort=False)
    m["run_max_f"] = grp["temp_f"].cummax()
    m["run_max6_f"] = grp["max6_clean_f"].cummax()
    m["run_max6_f"] = m.groupby("day", sort=False)["run_max6_f"].ffill()
    m["obs_count"] = grp.cumcount() + 1
    return m


def _asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on_left: str,
    on_right: str,
    cols: list[str],
    suffix: str = "",
) -> pd.DataFrame:
    """Left as-of join on time (``right[on_right] <= left[on_left]``).

    Returns ``cols`` (plus the matched ``on_right`` time) suffixed, aligned to ``left``'s index.
    """
    keep = [on_right, *[c for c in cols if c != on_right]]
    r = right[keep].sort_values(on_right).rename(columns={c: c + suffix for c in keep})
    out = pd.merge_asof(
        left[[on_left]].reset_index().sort_values(on_left),
        r,
        left_on=on_left,
        right_on=on_right + suffix,
        direction="backward",
    )
    out = out.set_index("index").sort_index()
    return out[[c + suffix for c in keep]]


# --- observation features -------------------------------------------------------------------


def _obs_features(station: Station, snaps: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=snaps.index)

    # Latest observation at or before t (may be from before the climate day started).
    cur_cols = [
        "valid",
        "temp_f",
        "temp_f_precise",
        "dewpoint_f",
        "relh",
        "sknt",
        "gust",
        "vsby",
        "sky",
        "p01i",
        "precip",
        "alti",
    ]
    cur = _asof(snaps, m, "t", "valid", cur_cols, suffix="_cur")
    out["cur_temp_f"] = cur["temp_f_cur"]
    out["cur_temp_precise"] = cur["temp_f_precise_cur"].astype(float)
    out["cur_dewpoint_f"] = cur["dewpoint_f_cur"]
    out["cur_depression_f"] = out["cur_temp_f"] - out["cur_dewpoint_f"]
    out["cur_relh"] = cur["relh_cur"]
    out["cur_sknt"] = cur["sknt_cur"]
    out["cur_gust"] = cur["gust_cur"].fillna(0.0)
    out["cur_vsby"] = cur["vsby_cur"]
    out["cur_sky"] = cur["sky_cur"]
    out["cur_p01i"] = cur["p01i_cur"].fillna(0.0)
    out["cur_precip"] = cur["precip_cur"]
    out["cur_age_min"] = (snaps["t"] - cur["valid_cur"]).dt.total_seconds() / 60.0

    # Tendencies: latest ob at or before t-1h / t-3h.
    for hours, name in ((1, "1h"), (3, "3h")):
        shifted = snaps[["t"]].copy()
        shifted["t"] = shifted["t"] - pd.Timedelta(hours=hours)
        past = _asof(shifted, m, "t", "valid", ["temp_f", "alti"], suffix=f"_{name}")
        out[f"d_temp_{name}"] = out["cur_temp_f"] - past[f"temp_f_{name}"]
        if hours == 3:
            out["d_alti_3h"] = cur["alti_cur"] - past["alti_3h"]

    # Running max within the climate day: last in-day ob at or before t carries the cummax.
    inday = _asof(
        snaps, m, "t", "valid", ["day_start", "run_max_f", "run_max6_f", "obs_count"], suffix="_rm"
    )
    same_day = inday["day_start_rm"] == snaps["day_start"]
    out["obs_count"] = inday["obs_count_rm"].where(same_day, 0).fillna(0).astype(int)
    out["run_max_f"] = inday["run_max_f_rm"].where(same_day)
    out["run_max6_f"] = inday["run_max6_f_rm"].where(same_day)
    out["run_max_f"] = np.fmax(out["run_max_f"], out["run_max6_f"])  # official checkpoint wins
    out["cur_minus_runmax"] = out["cur_temp_f"] - out["run_max_f"]

    # Yesterday's METAR max (complete by midnight, so always known).
    yday = m.groupby("day")["temp_f"].max().rename("yday_metar_max_f")
    yday.index = pd.to_datetime(yday.index) + pd.Timedelta(days=1)
    out["yday_metar_max_f"] = pd.to_datetime(snaps["day"]).map(yday).to_numpy()
    return out


# --- forecast features ----------------------------------------------------------------------


def _nbs_features(station: Station, snaps: pd.DataFrame, nbs: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=snaps.index)
    n = nbs[nbs["station"] == station.icao].copy()
    if n.empty:
        for c in (
            "nbs_txn",
            "nbs_xnd",
            "nbs_txn_age_h",
            "nbs_tmp_max_today",
            "nbs_tmp_max_remaining",
            "nbs_tmp_next",
            "nbs_sky_remaining",
            "nbs_p06_remaining",
            "nbs_run_age_h",
        ):
            out[c] = np.nan
        return out
    n["avail"] = n["runtime"] + NBS_LAG

    # (a) Day-max forecast: txn valid at 00Z the following UTC day. Only runs before the
    # afternoon carry it, so take the latest *available run that has it* for this day.
    txn = n.dropna(subset=["txn"]).copy()
    # The day-max forecast is valid at 00Z, which in US local standard time is the evening
    # of the climate day it closes, so the LST date of the valid time *is* the climate day.
    txn = txn[txn["ftime"].dt.hour == 0]
    txn["day"] = txn["ftime"].dt.tz_convert(station.lst).dt.date
    txn = txn.sort_values("avail")
    keyed = snaps[["day", "t"]].reset_index()
    merged = (
        pd.merge_asof(
            keyed.sort_values("t"),
            txn[["day", "avail", "txn", "xnd"]].rename(columns={"avail": "t"}).sort_values("t"),
            on="t",
            by="day",
            direction="backward",
            allow_exact_matches=True,
        )
        .set_index("index")
        .sort_index()
    )
    out["nbs_txn"] = merged["txn"]
    out["nbs_xnd"] = merged["xnd"]
    # age of that run
    age = (
        pd.merge_asof(
            keyed.sort_values("t"),
            txn[["day", "avail"]]
            .assign(t=txn["avail"], run_avail=txn["avail"])
            .sort_values("t")[["day", "t", "run_avail"]],
            on="t",
            by="day",
            direction="backward",
        )
        .set_index("index")
        .sort_index()
    )
    out["nbs_txn_age_h"] = (snaps["t"] - age["run_avail"]).dt.total_seconds() / 3600.0

    # (b) Hourly curve from the latest available run of any cycle: max over the climate day
    # and over the remaining part of it, next 3-hourly value, mean sky / max p06 remaining.
    runs = n[["runtime", "avail"]].drop_duplicates().sort_values("avail")
    latest = (
        pd.merge_asof(
            keyed.sort_values("t"),
            runs.rename(columns={"avail": "t"}),
            on="t",
            direction="backward",
        )
        .set_index("index")
        .sort_index()
    )
    snaps_run = snaps[["day_start", "day_end", "t"]].copy()
    snaps_run["runtime"] = latest["runtime"]
    snaps_run["snap"] = snaps_run.index
    near = n[n["ftime"] <= n["runtime"] + pd.Timedelta(hours=42)][
        ["runtime", "ftime", "tmp", "sky", "p06"]
    ]
    joined = snaps_run.merge(near, on="runtime", how="left")
    inday = joined[(joined["ftime"] > joined["day_start"]) & (joined["ftime"] <= joined["day_end"])]
    remaining = inday[inday["ftime"] > inday["t"]]
    g_all = inday.groupby("snap")["tmp"].max()
    g_rem = remaining.groupby("snap").agg(
        tmp_max=("tmp", "max"), sky=("sky", "mean"), p06=("p06", "max")
    )
    nxt = remaining.sort_values("ftime").groupby("snap")["tmp"].first()
    out["nbs_tmp_max_today"] = g_all.reindex(out.index).to_numpy()
    out["nbs_tmp_max_remaining"] = g_rem["tmp_max"].reindex(out.index).to_numpy()
    out["nbs_tmp_next"] = nxt.reindex(out.index).to_numpy()
    out["nbs_sky_remaining"] = g_rem["sky"].reindex(out.index).to_numpy()
    out["nbs_p06_remaining"] = g_rem["p06"].reindex(out.index).to_numpy()
    out["nbs_run_age_h"] = (snaps["t"] - (latest["runtime"] + NBS_LAG)).dt.total_seconds() / 3600.0
    return out


def _gfs_features(station: Station, snaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=snaps.index)
    g = gfs[gfs["station"] == station.icao].dropna(subset=["n_x"]).copy()
    if g.empty:
        out["gfs_nx"] = np.nan
        out["gfs_nx_age_h"] = np.nan
        return out
    g = g[g["ftime"].dt.hour == 0]
    g["day"] = g["ftime"].dt.tz_convert(station.lst).dt.date
    g["t"] = g["runtime"] + GFS_LAG
    g["run_avail"] = g["t"]
    keyed = snaps[["day", "t"]].reset_index()
    merged = (
        pd.merge_asof(
            keyed.sort_values("t"),
            g[["day", "t", "n_x", "run_avail"]].sort_values("t"),
            on="t",
            by="day",
            direction="backward",
        )
        .set_index("index")
        .sort_index()
    )
    out["gfs_nx"] = merged["n_x"]
    out["gfs_nx_age_h"] = (snaps["t"] - merged["run_avail"]).dt.total_seconds() / 3600.0
    return out


# --- labels & anchor ------------------------------------------------------------------------


def _label_features(station: Station, snaps: pd.DataFrame, cli: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=snaps.index)
    c = cli[cli["station"] == station.icao].set_index("day")
    days = snaps["day"]
    high = c["high"].to_dict()
    out["high"] = days.map(high).astype(float)
    out["high_time"] = days.map(c["high_time"].to_dict())
    ydays = (pd.to_datetime(days) - pd.Timedelta(days=1)).dt.date
    yday_high = ydays.map(high).astype(float)
    published = snaps["hour_lst"] >= CLI_PUBLISHED_AFTER_HOURS
    out["yday_cli_high"] = yday_high.where(published)
    out["high_normal"] = ydays.map(c["high_normal"].to_dict()).astype(float)
    return out


def _finalize(snaps: pd.DataFrame) -> pd.DataFrame:
    s = snaps
    s["yday_cli_minus_metar"] = s["yday_cli_high"] - s["yday_metar_max_f"]

    fc_anchor = s["nbs_txn"].fillna(s["gfs_nx"]).fillna(s["high_normal"]).round()
    obs_anchor = s["run_max_f"].round()
    use_obs = obs_anchor.notna() & (
        (obs_anchor >= fc_anchor - OBS_ANCHOR_WITHIN) | (s["hour_lst"] >= OBS_ANCHOR_FROM_HOUR)
    )
    s["anchor"] = obs_anchor.where(use_obs, fc_anchor)
    s["anchor_is_obs"] = use_obs.astype(float)
    s["anchor_is_nbm"] = s["nbs_txn"].notna().astype(float)
    s["run_max_minus_anchor"] = s["run_max_f"] - s["anchor"]
    s["nbs_txn_minus_anchor"] = s["nbs_txn"] - s["anchor"]
    s["gfs_nx_minus_anchor"] = s["gfs_nx"] - s["anchor"]

    doy = pd.to_datetime(s["day"]).dt.dayofyear.astype(float)
    s["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    s["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    s["station_id"] = s["station"].map({st.icao: i for i, st in enumerate(STATIONS)}).astype(int)

    resid = s["high"] - s["anchor"]
    s["residual"] = resid
    s["target"] = resid.clip(MIN_RESIDUAL_DEGREES, MAX_RESIDUAL_DEGREES)
    return s


# --- public API -----------------------------------------------------------------------------


def build_station(
    station: Station,
    cli: pd.DataFrame,
    metar: pd.DataFrame,
    nbs: pd.DataFrame,
    gfs: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    days = cli.loc[
        (cli["station"] == station.icao) & (cli["day"] >= start) & (cli["day"] <= end), "day"
    ]
    days = days.drop_duplicates().sort_values().reset_index(drop=True)
    snaps = _snapshot_frame(station, days)
    return features_for_snapshots(station, snaps, cli, metar, nbs, gfs)


def features_for_snapshots(
    station: Station,
    snaps: pd.DataFrame,
    cli: pd.DataFrame,
    metar: pd.DataFrame,
    nbs: pd.DataFrame,
    gfs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute every feature (and the label, where known) for an arbitrary snapshot frame.

    ``snaps`` needs ``station, day, day_start, day_end, hour_lst, t`` — see
    :func:`_snapshot_frame` for the historical grid and :func:`snapshot_frame_at` for live use.
    """
    m = _prepare_metar(station, metar)
    parts = [
        snaps,
        _obs_features(station, snaps, m),
        _nbs_features(station, snaps, nbs),
        _gfs_features(station, snaps, gfs),
        _label_features(station, snaps, cli),
    ]
    return _finalize(pd.concat(parts, axis=1))


def snapshot_frame_at(station: Station, now: pd.Timestamp, days: list[date]) -> pd.DataFrame:
    """Snapshot rows for the given climate ``days`` as seen at wall-clock ``now`` (live use)."""
    rows = []
    for day in days:
        day_start = pd.Timestamp(climate_day_start(station, day))
        rows.append(
            {
                "station": station.icao,
                "day": day,
                "day_start": day_start,
                "day_end": day_start + pd.Timedelta(hours=24),
                "hour_lst": (now - day_start).total_seconds() / 3600.0,
                "t": now,
            }
        )
    return pd.DataFrame(rows)


def build_snapshots(
    start: date = HISTORY_START,
    end: date | None = None,
    stations: tuple[Station, ...] = STATIONS,
    *,
    save: bool = True,
) -> pd.DataFrame:
    """Build (and cache) the full snapshot table for all stations."""
    end = end or (date.today() - timedelta(days=1))
    cli = iem.load_cli(stations, start, end)
    metar = iem.load_metar(stations, start, end)
    nbs = iem.load_mos("NBS", stations, start, end)
    gfs = iem.load_mos("GFS", stations, start, end)
    tables = []
    for st in stations:
        log.info("building snapshots for %s", st.icao)
        tables.append(build_station(st, cli, metar, nbs, gfs, start, end))
    table = pd.concat(tables, ignore_index=True)
    table = table.dropna(subset=["high", "anchor"]).reset_index(drop=True)
    if save:
        path = processed_dir() / "snapshots.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        log.info("wrote %s (%d rows)", path, len(table))
    return table


def load_snapshots() -> pd.DataFrame:
    return pd.read_parquet(processed_dir() / "snapshots.parquet")
