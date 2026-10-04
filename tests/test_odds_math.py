import pytest

from betmap.odds.math import (
    american_to_decimal,
    decimal_to_american,
    devig,
    expected_value,
    kelly_fraction,
    overround,
    parse_odds,
)


@pytest.mark.parametrize(
    "american,decimal",
    [(-110, 1.9090909), (+150, 2.5), (-200, 1.5), (+100, 2.0)],
)
def test_american_decimal_roundtrip(american, decimal):
    assert american_to_decimal(american) == pytest.approx(decimal)
    assert decimal_to_american(decimal) == pytest.approx(american)


@pytest.mark.parametrize("bad", [0, 50, -99])
def test_invalid_american(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


def test_parse_odds():
    assert parse_odds("-110") == pytest.approx(1.9090909)
    assert parse_odds("+150") == pytest.approx(2.5)
    assert parse_odds("150") == pytest.approx(2.5)
    assert parse_odds("1.91") == pytest.approx(1.91)
    with pytest.raises(ValueError):
        parse_odds("0.9")


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_symmetric_market(method):
    price = american_to_decimal(-110)
    assert devig([price, price], method) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_sums_to_one(method):
    prices = [american_to_decimal(-250), american_to_decimal(+210)]
    fair = devig(prices, method)
    assert sum(fair) == pytest.approx(1.0)
    assert fair[0] > fair[1]


def test_power_and_shin_shift_vig_to_longshot():
    prices = [american_to_decimal(-400), american_to_decimal(+320)]
    mult = devig(prices, "multiplicative")
    for method in ("power", "shin"):
        assert devig(prices, method)[1] < mult[1]


def test_overround():
    price = american_to_decimal(-110)
    assert overround([price, price]) == pytest.approx(0.047619, abs=1e-6)


def test_expected_value():
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert expected_value(0.5, american_to_decimal(-110)) == pytest.approx(-0.04545, abs=1e-5)


def test_kelly_closed_form():
    # f* = (bp - q) / b; at even money with p=0.55, f* = 0.10.
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.40, 3.0) == pytest.approx((2 * 0.4 - 0.6) / 2)
    assert kelly_fraction(0.45, 2.0) == 0.0
