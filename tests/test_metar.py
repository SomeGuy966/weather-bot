import math

import pandas as pd

from weatherbot.metar import c_to_f, parse_temp_columns, parse_temps


def test_t_group_beats_integer_body():
    p = parse_temps(
        "KNYC 010051Z AUTO 00000KT 4SM BR BKN030 23/21 A2988 RMK AO2 SLP110 T02280211 $"
    )
    assert p.temp_f_precise
    assert math.isclose(p.temp_f, c_to_f(22.8))
    assert math.isclose(p.dewpoint_f, c_to_f(21.1))


def test_integer_body_fallback_and_negative_temps():
    p = parse_temps("KDEN 010053Z 36012KT 10SM CLR M05/M12 A3021 RMK AO2 SLP257")
    assert not p.temp_f_precise
    assert math.isclose(p.temp_f, c_to_f(-5))
    assert math.isclose(p.dewpoint_f, c_to_f(-12))


def test_negative_t_group_sign_digit():
    p = parse_temps("KMDW 010051Z 00000KT 10SM CLR M03/M07 A3030 RMK AO2 T10281072")
    assert math.isclose(p.temp_f, c_to_f(-2.8))
    assert math.isclose(p.dewpoint_f, c_to_f(-7.2))


def test_six_hour_and_24_hour_groups():
    metar = (
        "KNYC 020451Z AUTO 00000KT 10SM CLR 20/15 A3001 RMK AO2 SLP160 "
        "T02000150 10272 20194 401231056 53012"
    )
    p = parse_temps(metar)
    assert math.isclose(p.max6_f, c_to_f(27.2))
    assert math.isclose(p.min6_f, c_to_f(19.4))
    assert math.isclose(p.max24_f, c_to_f(12.3))
    assert math.isclose(p.min24_f, c_to_f(-5.6))


def test_groups_only_parsed_from_remarks():
    # "10017" style tokens before RMK (e.g. a runway range) must not be read as a max group,
    # and a T-group with a missing dew point still parses.
    p = parse_temps("KLAX 010053Z 25008KT 10017 10SM FEW015 18/ A2995 RMK AO2 T0178")
    assert p.max6_f is None
    assert math.isclose(p.temp_f, c_to_f(17.8))
    assert p.dewpoint_f is None


def test_missing_temperature_gives_none():
    p = parse_temps("KNYC 010051Z AUTO 00000KT 4SM BR BKN030 A2988 RMK AO2 $")
    assert p.temp_f is None
    assert p.dewpoint_f is None


def test_vectorised_parser_matches_scalar():
    metars = pd.Series(
        [
            "KNYC 010051Z 00000KT 10SM CLR 23/21 A2988 RMK AO2 T02280211",
            None,
            "KNYC 010151Z 00000KT 10SM CLR 24/21 A2988 RMK AO2 10250 20200",
        ],
        index=[10, 11, 12],
    )
    df = parse_temp_columns(metars)
    assert list(df.index) == [10, 11, 12]
    assert math.isclose(df.loc[10, "temp_f"], c_to_f(22.8))
    assert pd.isna(df.loc[11, "temp_f"])
    assert math.isclose(df.loc[12, "max6_f"], c_to_f(25.0))
    assert not df.loc[12, "temp_f_precise"]
