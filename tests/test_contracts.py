import math

import numpy as np
import pytest

from weatherbot.contracts import Contract, contract_probability, parse_contract, standard_ladder


def _market(ticker, subtitle, strike_type, floor=None, cap=None):
    return {
        "ticker": ticker,
        "yes_sub_title": subtitle,
        "strike_type": strike_type,
        "floor_strike": floor,
        "cap_strike": cap,
    }


def test_parse_between_greater_less_from_live_shapes():
    b = parse_contract(_market("KXHIGHNY-26SEP11-B79.5", "79° to 80°", "between", 79, 80))
    assert (b.low, b.high) == (79, 80)
    g = parse_contract(_market("KXHIGHNY-26SEP11-T86", "87° or above", "greater", floor=86))
    assert g.low == 87 and math.isinf(g.high)
    lo = parse_contract(_market("KXHIGHNY-26SEP11-T79", "78° or below", "less", cap=79))
    assert math.isinf(lo.low) and lo.high == 78


def test_strike_fallback_when_subtitle_missing():
    g = parse_contract(_market("X-T86", "", "greater", floor=86))
    assert g.low == 87
    lo = parse_contract(_market("X-T79", "", "less", cap=79))
    assert lo.high == 78


def test_subtitle_strike_disagreement_is_an_error():
    with pytest.raises(ValueError):
        parse_contract(_market("X-B79.5", "80° to 81°", "between", 79, 80))


def test_contract_probability_partial_sums_and_tails():
    support = np.array([[75, 76, 77, 78, 79, 80]])
    pmf = np.array([[0.05, 0.10, 0.20, 0.30, 0.25, 0.10]])
    assert contract_probability(support, pmf, Contract("b", 77, 78)) == pytest.approx(0.5)
    assert contract_probability(support, pmf, Contract("above", 79, math.inf)) == pytest.approx(
        0.35
    )
    assert contract_probability(support, pmf, Contract("below", -math.inf, 76)) == pytest.approx(
        0.15
    )
    # Open-ended contract beyond the support edge receives the pooled edge mass, not zero.
    assert contract_probability(support, pmf, Contract("far", 85, math.inf)) == pytest.approx(0.10)
    assert contract_probability(support, pmf, Contract("far", -math.inf, 70)) == pytest.approx(0.05)


def test_standard_ladder_partitions_the_line():
    ladder = standard_ladder(80)
    assert [c.label for c in ladder] == [
        "76° or below",
        "77° to 78°",
        "79° to 80°",
        "81° to 82°",
        "83° to 84°",
        "85° or above",
    ]
    for value in range(60, 100):
        assert sum(c.contains(value) for c in ladder) == 1
