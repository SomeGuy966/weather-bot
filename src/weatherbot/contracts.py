"""Map Kalshi daily-high contracts onto integer-degree sets, and PMFs onto contract probabilities.

Kalshi's ``KXHIGH*`` events carry three contract shapes (all in whole °F, inclusive):

* ``between``  ``B79.5``  floor 79, cap 80 → "79° to 80°"  → {79, 80}
* ``greater``  ``T86``    floor 86         → "87° or above" → {87, 88, ...}
* ``less``     ``T79``    cap 79           → "78° or below" → {..., 77, 78}

The subtitle text is parsed as the source of truth and cross-checked against the strikes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

_BETWEEN = re.compile(r"(-?\d+)°?\s*to\s*(-?\d+)°?")
_ABOVE = re.compile(r"(-?\d+)°?\s*or\s*above")
_BELOW = re.compile(r"(-?\d+)°?\s*or\s*below")


@dataclass(frozen=True)
class Contract:
    ticker: str
    low: float
    """Lowest integer high (°F) that resolves YES; ``-inf`` for open-ended 'or below'."""
    high: float
    """Highest integer high (°F) that resolves YES; ``+inf`` for open-ended 'or above'."""

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    @property
    def label(self) -> str:
        if math.isinf(self.low):
            return f"{self.high:.0f}° or below"
        if math.isinf(self.high):
            return f"{self.low:.0f}° or above"
        return f"{self.low:.0f}° to {self.high:.0f}°"


def parse_contract(market: dict[str, Any]) -> Contract:
    """Build a :class:`Contract` from a Kalshi market object (``GET /markets/{ticker}``)."""
    ticker = str(market["ticker"])
    subtitle = str(market.get("yes_sub_title") or market.get("subtitle") or "")
    strike_type = market.get("strike_type")
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")

    if m := _BETWEEN.search(subtitle):
        low, high = float(m.group(1)), float(m.group(2))
    elif m := _ABOVE.search(subtitle):
        low, high = float(m.group(1)), math.inf
    elif m := _BELOW.search(subtitle):
        low, high = -math.inf, float(m.group(1))
    elif strike_type == "between" and floor is not None and cap is not None:
        low, high = float(floor), float(cap)
    elif strike_type == "greater" and floor is not None:
        low, high = float(floor) + 1, math.inf
    elif strike_type == "less" and cap is not None:
        low, high = -math.inf, float(cap) - 1
    else:
        raise ValueError(f"cannot parse contract range for {ticker}: {subtitle!r}")

    # Cross-check against the strikes when both are present.
    if (
        strike_type == "between"
        and floor is not None
        and cap is not None
        and (low, high) != (float(floor), float(cap))
    ):
        raise ValueError(f"{ticker}: subtitle {subtitle!r} disagrees with strikes {floor}-{cap}")
    if strike_type == "greater" and floor is not None and low != float(floor) + 1:
        raise ValueError(f"{ticker}: subtitle {subtitle!r} disagrees with floor {floor}")
    if strike_type == "less" and cap is not None and high != float(cap) - 1:
        raise ValueError(f"{ticker}: subtitle {subtitle!r} disagrees with cap {cap}")
    return Contract(ticker, low, high)


def contract_probability(support: np.ndarray, pmf: np.ndarray, contract: Contract) -> np.ndarray:
    """``P(low <= high <= high)`` for each row of a ``(n, k)`` support/PMF pair.

    The PMF's top and bottom classes are open-ended (they pool everything beyond the modelled
    residual range), so an open-ended contract that starts beyond the support's edge still
    receives the pooled tail mass rather than zero.
    """
    support = np.asarray(support, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    inside = (support >= contract.low) & (support <= contract.high)
    prob = (pmf * inside).sum(axis=1)
    # Open-ended contracts beyond the support edge: pool the edge class.
    if math.isinf(contract.high):
        beyond = contract.low > support[:, -1]
        prob = np.where(beyond, pmf[:, -1], prob)
    if math.isinf(contract.low):
        beyond = contract.high < support[:, 0]
        prob = np.where(beyond, pmf[:, 0], prob)
    return prob


def standard_ladder(center: int, width: int = 2, n_buckets: int = 4) -> list[Contract]:
    """Synthetic contract ladder shaped like Kalshi's, centred on ``center`` (°F).

    Used to score *contract-level* calibration on historical days that had no market.
    Kalshi lists ``n_buckets`` inclusive 2-degree buckets around the forecast high plus an
    open-ended contract on each side.
    """
    half = n_buckets // 2
    first_low = center - half * width + 1  # e.g. centre 80 → buckets 77-78, 79-80, 81-82, 83-84
    contracts = [Contract("BELOW", -math.inf, first_low - 1)]
    for i in range(n_buckets):
        lo = first_low + i * width
        contracts.append(Contract(f"B{lo + 0.5:.1f}", lo, lo + width - 1))
    contracts.append(Contract("ABOVE", first_low + n_buckets * width, math.inf))
    return contracts
