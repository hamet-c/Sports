"""
American-odds math, kept in one place so it's easy to test and reuse.
"""
from __future__ import annotations


def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds of 0 are invalid")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def implied_probability(odds: int) -> float:
    """The book's implied probability (includes vig)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def expected_value(probability: float, american_odds: int) -> float:
    """EV per $1 stake. Positive => +EV bet."""
    decimal = american_to_decimal(american_odds)
    return probability * (decimal - 1.0) - (1.0 - probability)


def kelly_fraction(probability: float, american_odds: int) -> float:
    """Full-Kelly stake fraction. Returns 0 if EV<=0. (Pre-fractioning; the user
    can apply a quarter/half-Kelly multiplier downstream.)"""
    decimal = american_to_decimal(american_odds)
    b = decimal - 1.0
    if b <= 0:
        return 0.0
    p = probability
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)
