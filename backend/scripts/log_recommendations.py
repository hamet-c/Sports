"""
Persist the engine's recommendations for a slate date to RecommendationLog.

Runs as step 3 of the daily flow (ingest_props -> refresh_injuries -> this),
so the log reflects what the deployed model + thresholds said pre-tip. Not
hooked into GET /slate: writes-on-read would log only when the UI happens to
be open, and multiple renders per day would fight over the row.

Upserts on (player_id, game_id, stat_type, book) — re-running for the same
date overwrites with the latest pre-tip snapshot. PASS rows are logged too.

    python scripts/log_recommendations.py                 # today's slate
    python scripts/log_recommendations.py --date 2026-05-27
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.timeutil import utcnow
from app.db.models import Game, PropLine, RecommendationLog
from app.db.session import SessionLocal, init_db
from app.models.registry import ModelRegistry
from app.services.prediction_service import PredictionService
from app.utils.upsert import upsert


def main(target: date) -> None:
    init_db()
    registry = ModelRegistry(use_calibrators=settings.use_calibrators)
    registry.load()
    if not registry.all_stats():
        raise RuntimeError("No models loaded — train first.")

    db = SessionLocal()
    try:
        games = db.query(Game).filter(Game.game_date == target).all()
        if not games:
            logger.info(f"No games on {target} — nothing to log.")
            return
        game_ids = [g.id for g in games]

        prop_rows = db.query(PropLine).filter(PropLine.game_id.in_(game_ids)).all()
        # Latest line per (player, game, stat, book) — same dedupe as the slate.
        latest: dict[tuple[int, int, str, str], PropLine] = {}
        for p in prop_rows:
            key = (p.player_id, p.game_id, p.stat_type, p.book)
            if key not in latest or p.captured_at > latest[key].captured_at:
                latest[key] = p
        if not latest:
            logger.info(f"No prop lines captured for {target} — nothing to log.")
            return

        svc = PredictionService(
            db, registry,
            edge_threshold=settings.edge_threshold,
            over_edge_threshold=settings.edge_threshold_over,
        )
        dist_cache: dict[tuple[int, int], dict] = {}
        now = utcnow()
        rows: list[dict] = []
        n_skipped = 0

        for prop in latest.values():
            if prop.stat_type not in registry:
                continue
            key = (prop.player_id, prop.game_id)
            if key not in dist_cache:
                try:
                    _, dists = svc.predict_player_game(
                        prop.player_id, prop.game_id, target,
                        stat_types=registry.all_stats(),
                    )
                    dist_cache[key] = dists
                except Exception as e:
                    logger.warning(
                        f"Prediction failed for player={prop.player_id} "
                        f"game={prop.game_id}: {e}"
                    )
                    dist_cache[key] = {}
            dist = dist_cache[key].get(prop.stat_type)
            if dist is None:
                n_skipped += 1
                continue

            edge = svc.analyze_distribution(
                player_id=prop.player_id,
                game_id=prop.game_id,
                stat_type=prop.stat_type,
                distribution=dist,
                line=prop.line,
                over_odds=prop.over_odds,
                under_odds=prop.under_odds,
                book=prop.book,
            )
            model = registry.get(prop.stat_type)
            rows.append({
                "created_at": now,
                "game_date": target,
                "player_id": prop.player_id,
                "game_id": prop.game_id,
                "stat_type": prop.stat_type,
                "book": prop.book,
                "line": prop.line,
                "over_odds": prop.over_odds,
                "under_odds": prop.under_odds,
                "p_over": edge.over_probability,
                "raw_p_over": edge.raw_over_probability,
                "ev_over": edge.expected_value_over,
                "ev_under": edge.expected_value_under,
                "recommendation": edge.recommendation,
                "edge_threshold_used": svc.edge_threshold,
                "edge_threshold_over_used": svc.over_edge_threshold,
                "model_version": model.predictor.model_version,
                "calibrated": model.calibrator is not None,
                "sharp_flag": edge.sharp_book_disagreement,
            })

        if not rows:
            logger.info(f"No loggable props for {target} (skipped {n_skipped}).")
            return
        n = upsert(
            db, RecommendationLog, rows,
            index_elements=["player_id", "game_id", "stat_type", "book"],
            # Grading fields are managed by grade_recommendations.py; a re-log
            # of the same prop resets them so the fresher snapshot gets graded.
        )
        n_recs = sum(1 for r in rows if r["recommendation"] != "PASS")
        logger.info(
            f"Logged {n} recommendation rows for {target} "
            f"({n_recs} OVER/UNDER, {n - n_recs} PASS, {n_skipped} skipped)."
        )
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    main(args.date)
