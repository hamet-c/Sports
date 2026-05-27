"""
Refit per-stat IsotonicCalibrators against the val window without retraining
the underlying XGBoost models. Much faster than re-running train_models.py
when the only thing we want to change is the calibrator.

The fit logic mirrors run_backtest.py's synthetic line: round L10 mean to
nearest .5 (bump integers up by .5 so there are no pushes). Each val row
contributes one (raw_p_over, actual_over) pair.

Usage:
    cd backend
    .venv\\Scripts\\python.exe scripts\\refit_calibrators.py \\
        --start 2026-02-15 --end 2026-04-13
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger
from tqdm import tqdm

from app.core.config import settings
from app.core.logging import configure_logging
from app.db.models import PlayerGameStats
from app.db.session import SessionLocal, init_db
from app.features.builder import FEATURE_COLUMNS, FeatureBuilder, coerce_feature_frame
from app.models.calibration import IsotonicCalibrator
from app.models.registry import ModelRegistry


STATS = ["points", "rebounds", "assists", "threes_made"]
BASELINE_COL = {
    "points": "pts_avg_10",
    "rebounds": "reb_avg_10",
    "assists": "ast_avg_10",
    "threes_made": "threes_avg_10",
}


def _to_dataframe_row(features: dict):
    import pandas as pd
    return pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])


def refit(start: date, end: date, min_minutes: float) -> None:
    init_db()
    # Load models WITHOUT existing calibrators — we want raw P(over) for the fit.
    registry = ModelRegistry(use_calibrators=False)
    registry.load()
    if len(registry) == 0:
        logger.error("No models loaded — train first.")
        return

    db = SessionLocal()
    try:
        builder = FeatureBuilder(db)
        rows = (
            db.query(PlayerGameStats)
            .filter(PlayerGameStats.game_date >= start)
            .filter(PlayerGameStats.game_date <= end)
            .filter(PlayerGameStats.minutes != None)  # noqa: E711
            .order_by(PlayerGameStats.game_date)
            .all()
        )
        rows = [r for r in rows if r.minutes is not None and r.minutes >= min_minutes]
        logger.info(f"Calibrator refit window: {len(rows)} player-games")

        raw_probs: dict[str, list[float]] = {s: [] for s in STATS}
        outcomes: dict[str, list[int]] = {s: [] for s in STATS}

        for r in tqdm(rows, desc="scoring"):
            feat_vec = builder.build(r.player_id, r.game_id, as_of=r.game_date)
            X = coerce_feature_frame(_to_dataframe_row(feat_vec.features))
            for stat in STATS:
                rm = registry.get(stat)
                if rm is None:
                    continue
                actual = getattr(r, stat, None)
                if actual is None:
                    continue
                actual = float(actual)
                dist = rm.predictor.predict(X)[0]

                baseline = feat_vec.features.get(BASELINE_COL[stat])
                if baseline is not None:
                    line = round(float(baseline) * 2) / 2
                    if line == int(line):
                        line += 0.5
                else:
                    line = dist.quantiles.get(0.5, dist.mean)

                raw_probs[stat].append(float(dist.probability_over(line)))
                outcomes[stat].append(1 if actual > line else 0)

        for stat in STATS:
            n = len(raw_probs[stat])
            if n == 0:
                logger.warning(f"{stat}: no samples — skipping")
                continue
            arr_p = np.array(raw_probs[stat])
            arr_y = np.array(outcomes[stat])
            cal = IsotonicCalibrator()
            cal.fit(arr_p, arr_y)
            out = settings.models_dir / f"{stat}_xgbq_calibration.joblib"
            cal.save(str(out))
            # quick before/after sanity at a few representative probs
            probes = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
            mapped = cal.transform(probes)
            logger.info(
                f"{stat}: n={n} over_rate={arr_y.mean():.3f} "
                f"raw_mean_p={arr_p.mean():.3f} -> wrote {out.name}"
            )
            logger.info(
                f"  cal map: {dict(zip([f'{p:.1f}' for p in probes], [round(float(m), 3) for m in mapped]))}"
            )
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--min-minutes", type=float, default=8.0)
    args = p.parse_args()
    configure_logging()
    refit(args.start, args.end, args.min_minutes)


if __name__ == "__main__":
    main()
