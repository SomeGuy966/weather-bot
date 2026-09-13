"""Feature builder on a hand-made day: as-of correctness, running max, forecast lags, labels."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from weatherbot.config import station
from weatherbot.features import build_station, climate_day_start
from weatherbot.metar import c_to_f

KNYC = station("KNYC")
DAY = date(2024, 7, 1)  # climate day: 2024-07-01 05:00Z → 2024-07-02 05:00Z (LST = UTC-5)


def _metar(valid: str, temp_c: float, remarks: str = "") -> dict:
    t = round(temp_c)
    tenths = f"T{'1' if temp_c < 0 else '0'}{abs(round(temp_c * 10)):03d}0150"
    stamp = f"{valid[8:10]}{valid[11:13]}{valid[14:16]}Z"
    raw = f"KNYC {stamp} AUTO 00000KT 10SM CLR {t:02d}/15 A3000 RMK AO2 {tenths} {remarks}".strip()
    return {
        "station": "KNYC",
        "valid": pd.Timestamp(valid, tz="UTC"),
        "tmpf": c_to_f(t),
        "dwpf": c_to_f(15),
        "relh": 60.0,
        "drct": 0.0,
        "sknt": 0.0,
        "gust": np.nan,
        "alti": 30.00,
        "mslp": np.nan,
        "p01i": 0.0,
        "vsby": 10.0,
        "skyc1": "CLR",
        "skyl1": np.nan,
        "wxcodes": None,
        "metar": raw,
    }


@pytest.fixture
def inputs():
    metar = pd.DataFrame(
        [
            _metar("2024-06-30 23:51", 23.9),  # evening before (18:51 LST): 75.0 °F
            # 05:51Z report carries a 6-hourly max group for 00-06Z, which straddles the
            # climate-day boundary (19:00-01:00 LST) and must NOT count as today's max.
            _metar("2024-07-01 05:51", 21.1, "10250"),  # 00:51 LST: 70.0 °F; group says 77 °F
            _metar("2024-07-01 07:51", 20.0),  # 68 °F
            # 11:51Z report: 6-hourly max for 06-12Z = 01:00-07:00 LST, fully in-day: 71.96 °F
            _metar("2024-07-01 11:51", 18.9, "10222"),  # 66 °F
            _metar("2024-07-01 17:51", 27.8),  # 12:51 LST: 82.04 °F
            _metar("2024-07-01 18:30", 28.9),  # SPECI 13:30 LST: 84.02 °F
            _metar("2024-07-01 20:51", 28.3),  # 83 °F
        ]
    )
    cli = pd.DataFrame(
        {
            "station": ["KNYC", "KNYC"],
            "day": [date(2024, 6, 30), DAY],
            "high": [76.0, 85.0],
            "high_time": ["300 PM", "145 PM"],
            "high_normal": [84.0, 84.0],
            "high_record": [98.0, 100.0],
            "low": [65.0, 66.0],
            "low_time": ["500 AM", "530 AM"],
            "precip": [0.0, 0.0],
            "snow": [0.0, 0.0],
            "product": ["x", "y"],
            "wfo": ["OKX", "OKX"],
        }
    )
    t = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    nbs = pd.DataFrame(
        [
            # 01Z run: available from 02Z, day-max forecast 84 ± 2 valid 00Z next day
            {
                "runtime": t("2024-07-01 01:00"),
                "ftime": t("2024-07-01 12:00"),
                "tmp": 68,
                "txn": np.nan,
                "xnd": np.nan,
            },
            {
                "runtime": t("2024-07-01 01:00"),
                "ftime": t("2024-07-01 18:00"),
                "tmp": 80,
                "txn": np.nan,
                "xnd": np.nan,
            },
            {
                "runtime": t("2024-07-01 01:00"),
                "ftime": t("2024-07-02 00:00"),
                "tmp": 78,
                "txn": 84.0,
                "xnd": 2.0,
            },
            # 07Z run: available from 08Z, revises the day max to 86
            {
                "runtime": t("2024-07-01 07:00"),
                "ftime": t("2024-07-01 18:00"),
                "tmp": 81,
                "txn": np.nan,
                "xnd": np.nan,
            },
            {
                "runtime": t("2024-07-01 07:00"),
                "ftime": t("2024-07-01 21:00"),
                "tmp": 85,
                "txn": np.nan,
                "xnd": np.nan,
            },
            {
                "runtime": t("2024-07-01 07:00"),
                "ftime": t("2024-07-02 00:00"),
                "tmp": 79,
                "txn": 86.0,
                "xnd": 1.0,
            },
            # 13Z run: no txn for today any more, only the hourly curve
            {
                "runtime": t("2024-07-01 13:00"),
                "ftime": t("2024-07-01 21:00"),
                "tmp": 84,
                "txn": np.nan,
                "xnd": np.nan,
            },
            {
                "runtime": t("2024-07-01 13:00"),
                "ftime": t("2024-07-02 00:00"),
                "tmp": 80,
                "txn": np.nan,
                "xnd": np.nan,
            },
            {
                "runtime": t("2024-07-01 13:00"),
                "ftime": t("2024-07-02 12:00"),
                "tmp": 70,
                "txn": 68.0,
                "xnd": 1.0,
            },
        ]
    )
    nbs["station"] = "KNYC"
    for col in ("dpt", "sky", "wsp", "p06", "p12", "tsd"):
        nbs[col] = np.nan
    gfs = pd.DataFrame(
        [
            {
                "station": "KNYC",
                "runtime": t("2024-07-01 00:00"),
                "ftime": t("2024-07-02 00:00"),
                "tmp": 78,
                "n_x": 83.0,
            }
        ]
    )
    for col in ("dpt", "cld", "wsp", "p06", "p12"):
        gfs[col] = np.nan
    return cli, metar, nbs, gfs


def test_climate_day_is_local_standard_time():
    assert climate_day_start(KNYC, DAY) == pd.Timestamp("2024-07-01 05:00", tz="UTC")
    assert climate_day_start(station("KLAX"), DAY) == pd.Timestamp("2024-07-01 08:00", tz="UTC")


def test_feature_table(inputs):
    cli, metar, nbs, gfs = inputs
    table = build_station(KNYC, cli, metar, nbs, gfs, DAY, DAY).set_index("hour_lst")
    assert len(table) == 30  # -6 .. 23

    # Before the day starts: no in-day obs, latest ob is yesterday evening's.
    h0 = table.loc[0]
    assert h0["obs_count"] == 0 and np.isnan(h0["run_max_f"])
    assert h0["cur_temp_f"] == pytest.approx(c_to_f(23.9))
    assert h0["yday_metar_max_f"] == pytest.approx(c_to_f(23.9))
    assert np.isnan(h0["yday_cli_high"])  # CLI for yesterday not published yet at 00:00 LST
    assert table.loc[4]["yday_cli_high"] == 76.0

    # Forecast availability lags: 01Z NBM run usable from 02Z; GFS 00Z run from 03:30Z.
    assert h0["nbs_txn"] == 84.0 and h0["nbs_xnd"] == 2.0
    assert h0["gfs_nx"] == 83.0
    assert np.isnan(table.loc[-4]["nbs_txn"])  # 01:00Z: the 01Z run is not out until 02Z
    assert table.loc[-3]["nbs_txn"] == 84.0  # 02:00Z: exactly the availability instant
    assert table.loc[2]["nbs_txn"] == 84.0  # 07:00Z: still the 01Z run
    assert table.loc[3]["nbs_txn"] == 86.0  # 08:00Z: the 07Z run has landed

    # Running max: straddling 6-hourly group ignored, in-day group used, SPECI as-of exact.
    assert table.loc[1]["run_max_f"] == pytest.approx(c_to_f(21.1))  # not the 77 °F group
    assert table.loc[7]["run_max6_f"] == pytest.approx(c_to_f(22.2))
    assert table.loc[7]["run_max_f"] == pytest.approx(c_to_f(22.2))
    assert table.loc[13]["run_max_f"] == pytest.approx(
        c_to_f(27.8)
    )  # 18:00Z: SPECI at 18:30Z excluded
    assert table.loc[14]["run_max_f"] == pytest.approx(c_to_f(28.9))
    assert table.loc[23]["obs_count"] == 6

    # Anchor: NBM day-max until the running max is in play, then the running max itself.
    assert table.loc[0]["anchor"] == 84.0 and table.loc[0]["residual"] == 1.0
    assert table.loc[1]["anchor"] == 84.0 and table.loc[1]["anchor_is_obs"] == 0  # 70 << 84
    assert table.loc[10]["anchor"] == 72.0 and table.loc[10]["anchor_is_obs"] == 1  # hour rule
    assert table.loc[14]["anchor"] == 84.0 and table.loc[14]["residual"] == 1.0  # run max 84.02
    assert (table["high"] == 85.0).all()

    # NBM hourly curve: from the latest run of any cycle; remaining-max shrinks through the day.
    assert table.loc[13]["nbs_tmp_max_remaining"] == 84.0  # 13Z run, 21Z value
    assert table.loc[17]["nbs_tmp_max_remaining"] == 80.0  # only 00Z left
    assert np.isnan(table.loc[20]["nbs_tmp_max_remaining"])
