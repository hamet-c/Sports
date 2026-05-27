"""
Prediction service — high-level orchestrator.
Routes call this. This calls features + model + edge calculator.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from app.features.builder import FeatureBuilder, FeatureVector, feature_dict_to_dataframe
from app.models.base import Distribution
from app.models.registry import ModelRegistry, RegisteredModel
from app.services.edge import expected_value, implied_probability, kelly_fraction


# Threshold for flagging "the book is shading this line hard and our model
# strongly disagrees on the shaded side". Implied probability ≥ 70% corresponds
# to roughly -233 American odds — books don't shade lines that hard without
# information (lineup news, in-game injury reports, sharp action). We flag
# when the book's favored side has implied prob ≥ this AND our model's
# probability on that same side is ≤ DISAGREEMENT_MAX_MODEL_PROB, i.e. we're
# saying the book's heavy favorite is actually a coin-flip or worse. That gap
# is the danger zone — most likely cause is information we don't have.
SHARP_BOOK_IMPLIED_THRESHOLD = 0.70
DISAGREEMENT_MAX_MODEL_PROB = 0.50


def detect_sharp_disagreement(
    over_odds: int, under_odds: int, p_over: float,
) -> tuple[bool, str]:
    """
    Pure helper. Returns (is_sharp_disagreement, book_favored_side).
    book_favored_side is "OVER" / "UNDER" / "EVEN".
    """
    over_imp = implied_probability(over_odds)
    under_imp = implied_probability(under_odds)
    if over_imp >= SHARP_BOOK_IMPLIED_THRESHOLD:
        return p_over <= DISAGREEMENT_MAX_MODEL_PROB, "OVER"
    if under_imp >= SHARP_BOOK_IMPLIED_THRESHOLD:
        return (1.0 - p_over) <= DISAGREEMENT_MAX_MODEL_PROB, "UNDER"
    return False, "EVEN"


@dataclass
class EdgeAnalysis:
    player_id: int
    game_id: int
    stat_type: str
    line: float
    over_odds: int
    under_odds: int
    book: str
    predicted_mean: float
    quantiles: dict[float, float]
    over_probability: float          # calibrated, post-isotonic
    under_probability: float
    raw_over_probability: float      # pre-calibration, for logging
    expected_value_over: float
    expected_value_under: float
    kelly_over: float
    kelly_under: float
    recommendation: str              # "OVER" | "UNDER" | "PASS"
    sharp_book_disagreement: bool    # True when the book is shading hard AND we disagree
    book_favored_side: str           # "OVER" | "UNDER" | "EVEN" — which side the book implies wins


class PredictionService:
    def __init__(
        self,
        db: Session,
        registry: ModelRegistry,
        edge_threshold: float = 0.05,
        over_edge_threshold: float | None = None,
    ) -> None:
        """
        edge_threshold: minimum EV to recommend either side.
        over_edge_threshold: optional stricter threshold for OVER recs only.
            Defaults to edge_threshold (symmetric). Set higher (e.g. 0.10)
            when the model is known to over-predict means (Phase 5.5
            diagnostic showed cal_off OVER recs at 46.2% vs UNDER at 53.3%
            on real lines May 11-17 — asymmetric thresholds demand more
            evidence on the biased side without a full retrain).
        """
        self.db = db
        self.registry = registry
        self.feature_builder = FeatureBuilder(db)
        self.edge_threshold = edge_threshold
        self.over_edge_threshold = (
            over_edge_threshold if over_edge_threshold is not None else edge_threshold
        )

    # ---------------------------- core operations --------------------------- #

    def predict_player_game(
        self,
        player_id: int,
        game_id: int,
        as_of: date,
        stat_types: list[str] | None = None,
        market: dict[str, float | None] | None = None,
    ) -> tuple[FeatureVector, dict[str, Distribution]]:
        """Return (feature vector, {stat: Distribution})."""
        stat_types = stat_types or self.registry.all_stats()
        feature_vec = self.feature_builder.build(player_id, game_id, as_of, market=market)
        X = feature_dict_to_dataframe(feature_vec.features)
        out: dict[str, Distribution] = {}
        for stat in stat_types:
            rm = self.registry.get(stat)
            if rm is None:
                continue
            out[stat] = rm.predictor.predict(X)[0]
        return feature_vec, out

    def analyze_prop(
        self,
        player_id: int,
        game_id: int,
        as_of: date,
        stat_type: str,
        line: float,
        over_odds: int,
        under_odds: int,
        book: str = "consensus",
        market: dict[str, float | None] | None = None,
    ) -> EdgeAnalysis | None:
        rm = self.registry.get(stat_type)
        if rm is None:
            return None
        feature_vec = self.feature_builder.build(player_id, game_id, as_of, market=market)
        X = feature_dict_to_dataframe(feature_vec.features)
        dist, cal_over = rm.predict_with_probability(X, line)
        raw_over = dist.probability_over(line)
        return self._build_edge(
            player_id=player_id,
            game_id=game_id,
            stat_type=stat_type,
            line=line,
            over_odds=over_odds,
            under_odds=under_odds,
            book=book,
            distribution=dist,
            cal_over_prob=cal_over,
            raw_over_prob=raw_over,
        )

    def analyze_distribution(
        self,
        player_id: int,
        game_id: int,
        stat_type: str,
        distribution: Distribution,
        line: float,
        over_odds: int,
        under_odds: int,
        book: str = "consensus",
    ) -> EdgeAnalysis:
        """For batch slate flow where the distribution is already in hand."""
        rm = self.registry.get(stat_type)
        raw_over = distribution.probability_over(line)
        if rm is not None and rm.calibrator is not None and rm.calibrator.is_fitted():
            import numpy as np
            cal_over = float(rm.calibrator.transform(np.array([raw_over]))[0])
        else:
            cal_over = raw_over
        return self._build_edge(
            player_id=player_id,
            game_id=game_id,
            stat_type=stat_type,
            line=line,
            over_odds=over_odds,
            under_odds=under_odds,
            book=book,
            distribution=distribution,
            cal_over_prob=cal_over,
            raw_over_prob=raw_over,
        )

    # ------------------------------ persistence ----------------------------- #

    def persist_prediction(
        self,
        feature_vec: FeatureVector,
        stat_type: str,
        distribution: Distribution,
        model_version: str,
        line: float | None = None,
        over_probability: float | None = None,
        ev_over: float | None = None,
        ev_under: float | None = None,
    ) -> None:
        from app.db.models import Prediction
        q = distribution.quantiles
        row = Prediction(
            player_id=feature_vec.player_id,
            game_id=feature_vec.game_id,
            stat_type=stat_type,
            model_version=model_version,
            predicted_mean=distribution.mean,
            predicted_p10=q.get(0.1),
            predicted_p25=q.get(0.25),
            predicted_p50=q.get(0.5),
            predicted_p75=q.get(0.75),
            predicted_p90=q.get(0.9),
            line=line,
            over_probability=over_probability,
            expected_value_over=ev_over,
            expected_value_under=ev_under,
            features_json=feature_vec.features,
        )
        self.db.add(row)
        self.db.commit()

    # ------------------------------ internals ------------------------------- #

    def _build_edge(
        self,
        *,
        player_id: int,
        game_id: int,
        stat_type: str,
        line: float,
        over_odds: int,
        under_odds: int,
        book: str,
        distribution: Distribution,
        cal_over_prob: float,
        raw_over_prob: float,
    ) -> EdgeAnalysis:
        p_over = float(cal_over_prob)
        p_under = 1.0 - p_over
        ev_over = expected_value(p_over, over_odds)
        ev_under = expected_value(p_under, under_odds)
        k_over = kelly_fraction(p_over, over_odds)
        k_under = kelly_fraction(p_under, under_odds)

        if ev_over >= self.over_edge_threshold and ev_over > ev_under:
            rec = "OVER"
        elif ev_under >= self.edge_threshold and ev_under > ev_over:
            rec = "UNDER"
        else:
            rec = "PASS"

        # Sharp-book-disagreement detection. See detect_sharp_disagreement
        # docstring and the SHARP_BOOK_IMPLIED_THRESHOLD comment up top.
        sharp_disagreement, book_side = detect_sharp_disagreement(
            over_odds, under_odds, p_over,
        )

        return EdgeAnalysis(
            player_id=player_id,
            game_id=game_id,
            stat_type=stat_type,
            line=line,
            over_odds=over_odds,
            under_odds=under_odds,
            book=book,
            predicted_mean=distribution.mean,
            quantiles=dict(distribution.quantiles),
            over_probability=p_over,
            under_probability=p_under,
            raw_over_probability=float(raw_over_prob),
            expected_value_over=ev_over,
            expected_value_under=ev_under,
            kelly_over=k_over,
            kelly_under=k_under,
            recommendation=rec,
            sharp_book_disagreement=sharp_disagreement,
            book_favored_side=book_side,
        )
