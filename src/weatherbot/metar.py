"""Parse the temperature groups out of raw METAR reports.

Only the pieces the model needs are parsed (everything else comes pre-decoded from IEM):

* body ``23/21``           – integer-Celsius temperature / dew point;
* remark ``T02280211``     – temperature / dew point to 0.1 °C (the *hourly temperature*);
* remark ``10017`` / ``20006`` – 6-hourly max / min temperature to 0.1 °C (00/06/12/18Z reports);
* remark ``401231056``     – 24-hour max / min to 0.1 °C (reported once a day at local midnight).

Sign digit convention: ``0`` positive, ``1`` negative. All results are returned in °F.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_BODY_TEMP = re.compile(r"(?<!\S)(M?\d{2})/(M?\d{2})?(?!\S)")
_T_GROUP = re.compile(r"(?<!\S)T([01])(\d{3})(?:([01])(\d{3}))?(?!\S)")
_MAX6 = re.compile(r"(?<!\S)1([01])(\d{3})(?!\S)")
_MIN6 = re.compile(r"(?<!\S)2([01])(\d{3})(?!\S)")
_MAXMIN24 = re.compile(r"(?<!\S)4([01])(\d{3})([01])(\d{3})(?!\S)")


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _tenths(sign: str, digits: str) -> float:
    value = int(digits) / 10.0
    return -value if sign == "1" else value


def _int_c(token: str) -> float:
    return -int(token[1:]) if token.startswith("M") else int(token)


@dataclass(frozen=True)
class TempGroups:
    """Temperatures parsed from one METAR, all in °F (``None`` when the group is absent)."""

    temp_f: float | None
    """Best available temperature: the 0.1 °C T-group if present, else the integer-°C body."""
    temp_f_precise: bool
    """Whether ``temp_f`` came from the T-group (0.1 °C) rather than the integer body."""
    dewpoint_f: float | None
    max6_f: float | None
    min6_f: float | None
    max24_f: float | None
    min24_f: float | None


def parse_temps(metar: str) -> TempGroups:
    """Parse the temperature groups of one raw METAR string."""
    body, _, remarks = metar.partition(" RMK ")

    body_t: float | None = None
    body_td: float | None = None
    if m := _BODY_TEMP.search(body):
        body_t = c_to_f(_int_c(m.group(1)))
        if m.group(2):
            body_td = c_to_f(_int_c(m.group(2)))

    precise_t: float | None = None
    precise_td: float | None = None
    if remarks and (m := _T_GROUP.search(remarks)):
        precise_t = c_to_f(_tenths(m.group(1), m.group(2)))
        if m.group(3) is not None:
            precise_td = c_to_f(_tenths(m.group(3), m.group(4)))

    max6 = min6 = max24 = min24 = None
    if remarks:
        if m := _MAX6.search(remarks):
            max6 = c_to_f(_tenths(m.group(1), m.group(2)))
        if m := _MIN6.search(remarks):
            min6 = c_to_f(_tenths(m.group(1), m.group(2)))
        if m := _MAXMIN24.search(remarks):
            max24 = c_to_f(_tenths(m.group(1), m.group(2)))
            min24 = c_to_f(_tenths(m.group(3), m.group(4)))

    return TempGroups(
        temp_f=precise_t if precise_t is not None else body_t,
        temp_f_precise=precise_t is not None,
        dewpoint_f=precise_td if precise_td is not None else body_td,
        max6_f=max6,
        min6_f=min6,
        max24_f=max24,
        min24_f=min24,
    )


def parse_temp_columns(metars: pd.Series) -> pd.DataFrame:
    """Vectorised :func:`parse_temps` over a Series of raw METARs → one column per field."""
    parsed = [parse_temps(m) if isinstance(m, str) else parse_temps("") for m in metars]
    return pd.DataFrame(
        {
            "temp_f": [p.temp_f for p in parsed],
            "temp_f_precise": [p.temp_f_precise for p in parsed],
            "dewpoint_f": [p.dewpoint_f for p in parsed],
            "max6_f": [p.max6_f for p in parsed],
            "min6_f": [p.min6_f for p in parsed],
            "max24_f": [p.max24_f for p in parsed],
            "min24_f": [p.min24_f for p in parsed],
        },
        index=metars.index,
    )
