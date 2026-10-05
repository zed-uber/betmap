"""Odds conversion, de-vigging, expected value, and Kelly sizing.

All prices are decimal odds unless a function name says otherwise.
"""

import math
from collections.abc import Sequence

from scipy.optimize import brentq


def american_to_decimal(american: float) -> float:
    if american == 0 or -100 < american < 100:
        raise ValueError(f"invalid American odds: {american}")
    return 1 + (american / 100 if american > 0 else 100 / -american)


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1:
        raise ValueError(f"invalid decimal odds: {decimal}")
    return (decimal - 1) * 100 if decimal >= 2 else -100 / (decimal - 1)


def contract_to_decimal(price: float) -> float:
    """Prediction-market contract price (cost to win $1) to decimal odds, before fees."""
    if not 0 < price < 1:
        raise ValueError(f"invalid contract price: {price}")
    return 1 / price


def parse_odds(text: str) -> float:
    """Parse odds and return decimal odds.

    '+150', '-110' are American; '2.5' is decimal; '0.57', '57c' or '57¢' is a
    prediction-market contract price (decimal odds are always > 1, so 0-1 is unambiguous).
    """
    text = text.strip()
    if text.lower().endswith(("c", "¢")):
        return contract_to_decimal(float(text[:-1]) / 100)
    value = float(text)
    if text.startswith(("+", "-")) or abs(value) >= 100:
        return american_to_decimal(value)
    if 0 < value < 1:
        return contract_to_decimal(value)
    if value <= 1:
        raise ValueError(f"invalid odds: {text}")
    return value


def implied_prob(decimal: float) -> float:
    return 1 / decimal


def overround(decimals: Sequence[float]) -> float:
    """Book margin: sum of implied probabilities minus 1."""
    return sum(implied_prob(d) for d in decimals) - 1


def devig_multiplicative(decimals: Sequence[float]) -> list[float]:
    """Scale implied probabilities proportionally so they sum to 1."""
    probs = [implied_prob(d) for d in decimals]
    total = sum(probs)
    return [p / total for p in probs]


def devig_power(decimals: Sequence[float]) -> list[float]:
    """Find k with sum(q_i ** k) == 1. Shifts more vig onto longshots than multiplicative."""
    probs = [implied_prob(d) for d in decimals]
    if math.isclose(sum(probs), 1.0):
        return probs
    k = brentq(lambda k: sum(p**k for p in probs) - 1, 0.01, 100)
    return [p**k for p in probs]


def devig_shin(decimals: Sequence[float]) -> list[float]:
    """Shin (1993) method, modelling the margin as protection against insider money."""
    probs = [implied_prob(d) for d in decimals]
    total = sum(probs)
    if math.isclose(total, 1.0):
        return probs

    def fair(z: float) -> list[float]:
        return [(math.sqrt(z**2 + 4 * (1 - z) * p**2 / total) - z) / (2 * (1 - z)) for p in probs]

    z = brentq(lambda z: sum(fair(z)) - 1, 0.0, 0.99)
    return fair(z)


DEVIG_METHODS = {
    "multiplicative": devig_multiplicative,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(decimals: Sequence[float], method: str = "power") -> list[float]:
    return DEVIG_METHODS[method](decimals)


def expected_value(prob: float, decimal: float) -> float:
    """Expected profit per unit staked."""
    return prob * (decimal - 1) - (1 - prob)


def kelly_fraction(prob: float, decimal: float) -> float:
    """Full-Kelly bankroll fraction for a single binary bet; 0 when there is no edge."""
    b = decimal - 1
    return max(0.0, (b * prob - (1 - prob)) / b)


def fair_decimal(prob: float) -> float:
    return 1 / prob


def after_exchange_fee(decimal: float, rate: float) -> float:
    """Decimal odds net of an exchange taker fee of rate x P x (1 - P) per $1 contract.

    A contract priced P pays $1; with the fee it costs P + fee, so the odds you actually
    get are 1 / (P + fee). Kalshi's rate is 0.07 (rounding up to the cent is ignored).
    """
    if rate <= 0:
        return decimal
    p = 1 / decimal
    return 1 / (p + rate * p * (1 - p))
