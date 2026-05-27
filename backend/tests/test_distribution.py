"""Distribution interpolation + edge math + sharp-book-disagreement detector."""
from app.models.base import Distribution
from app.services.edge import (
    american_to_decimal,
    expected_value,
    implied_probability,
    kelly_fraction,
)
from app.services.prediction_service import detect_sharp_disagreement


def test_distribution_probability_over_at_extremes():
    d = Distribution(mean=20.0, quantiles={0.1: 10.0, 0.5: 20.0, 0.9: 30.0})
    assert abs(d.probability_over(5.0) - 0.9) < 1e-9   # below the lowest quantile
    assert abs(d.probability_over(35.0) - 0.1) < 1e-9  # above the highest quantile


def test_distribution_probability_over_interpolation():
    d = Distribution(mean=20.0, quantiles={0.1: 10.0, 0.5: 20.0, 0.9: 30.0})
    # At the median value, P(over) should be 0.5
    assert abs(d.probability_over(20.0) - 0.5) < 1e-9
    # Halfway between p10 and p50: cdf ~ 0.3 -> P(over) ~ 0.7
    assert abs(d.probability_over(15.0) - 0.7) < 1e-9


def test_probability_over_plus_under_equals_one():
    d = Distribution(mean=20.0, quantiles={0.1: 10.0, 0.5: 20.0, 0.9: 30.0})
    assert abs(d.probability_over(17.5) + d.probability_under(17.5) - 1.0) < 1e-9


def test_american_to_decimal():
    assert abs(american_to_decimal(-110) - (1 + 100 / 110)) < 1e-9
    assert abs(american_to_decimal(150) - 2.5) < 1e-9


def test_implied_probability():
    # +100 -> 0.5 implied
    assert abs(implied_probability(100) - 0.5) < 1e-9
    # -110 -> 110/210 ≈ 0.5238
    assert abs(implied_probability(-110) - 110.0 / 210.0) < 1e-9


def test_expected_value_breakeven():
    # At -110, breakeven probability is 110/210
    p_break = 110.0 / 210.0
    assert abs(expected_value(p_break, -110)) < 1e-9


def test_kelly_zero_when_no_edge():
    # +100 line, true prob 0.5 -> EV is 0 -> kelly is 0
    assert kelly_fraction(0.5, 100) == 0.0


def test_kelly_positive_when_edge():
    f = kelly_fraction(0.6, 100)  # +100, true 60% -> EV +20%, kelly = 0.2
    assert abs(f - 0.2) < 1e-9


# ---------------------- sharp-book-disagreement detector --------------------- #
# Threshold: book implied ≥70% (≈ -233 or shorter) AND model ≤50% on that side.


def test_sharp_disagreement_fires_when_book_heavy_over_model_low():
    # -290 OVER → implied ≈ 74.4% (the row in the user's screenshot).
    # Model says OVER 32% → that's ≤50% → flag fires.
    flagged, side = detect_sharp_disagreement(-290, 210, p_over=0.323)
    assert flagged is True
    assert side == "OVER"


def test_sharp_disagreement_no_fire_when_book_heavy_and_model_agrees():
    # Book heavy on OVER; model also says OVER is very likely. No disagreement.
    flagged, side = detect_sharp_disagreement(-290, 210, p_over=0.85)
    assert flagged is False
    assert side == "OVER"  # we still identify which side the book favors


def test_sharp_disagreement_fires_on_under_side():
    # +210 OVER / -290 UNDER → book heavily favors UNDER.
    # Model says OVER 70% → P(under)=30% ≤50% → flag fires on UNDER side.
    flagged, side = detect_sharp_disagreement(210, -290, p_over=0.70)
    assert flagged is True
    assert side == "UNDER"


def test_sharp_disagreement_no_fire_on_routine_juice():
    # -110 / -110 → each side implied 52.4%, well below the 70% threshold.
    flagged, side = detect_sharp_disagreement(-110, -110, p_over=0.30)
    assert flagged is False
    assert side == "EVEN"


def test_sharp_disagreement_boundary_around_threshold():
    # -233 → implied ≈ 69.97%, just below the 70% threshold. EVEN.
    flagged_below, side_below = detect_sharp_disagreement(-233, 200, p_over=0.49)
    assert side_below == "EVEN"
    assert flagged_below is False
    # -234 → implied ≈ 70.06%, just above the threshold. Flag should fire when
    # the model also disagrees.
    flagged_above, side_above = detect_sharp_disagreement(-234, 200, p_over=0.49)
    assert side_above == "OVER"
    assert flagged_above is True
    # Same -234 line but model gives OVER 0.51 → above DISAGREEMENT_MAX_MODEL_PROB,
    # so the book is favored but no disagreement flag fires.
    flagged_agree, _ = detect_sharp_disagreement(-234, 200, p_over=0.51)
    assert flagged_agree is False
