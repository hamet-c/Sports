"""
Quantify the squared-error mean-head bias on the current models.

For each stat, walk PlayerGameStats in a held-out window, predict, and report:
    mean(pred_mean - actual)        # signed bias of the mean head
    median(pred_mean - actual)      # signed median error (less skew-sensitive)
    mean(pred_median - actual)      # same for the q=0.5 quantile head
    median(pred_median - actual)

If the mean head is biased high but the q=0.5 head isn't, the squared-error
loss + right-skewed target is the culprit and we can fix this by reading
`mean` from the q=0.5 head instead of training a separate squared-error head.

Phase 5 measured +1.77/+1.22/+0.22/-0.09 on points/rebounds/threes/assists
against a different model. This re-measures on the Phase 5.6 (is_playoff)
models.

Usage (from backend/):
    .venv\\Scripts\\python.exe scripts\\diagnose_mean_head_bias.py \\
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

from app.core.logging import configure_logging
from app.db.models import PlayerGameStats
from app.db.session import SessionLocal, init_db
from app.features.builder import FEATURE_COLUMNS, FeatureBuilder, coerce_feature_frame
from app.models.registry import ModelRegistry


STATS = ("points", "rebounds", "assists", "threes_made")


def _to_dataframe_row(features: dict):
    import pandas as pd
    return pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])


def main(start: date, end: date, min_minutes: float) -> None:
    init_db()
    registry = ModelRegistry(use_calibrators=False)
    registry.load()
    if len(registry) == 0:
        raise RuntimeError("No models loaded — train first.")

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
        logger.info(f"Window has {len(rows)} player-games meeting min_minutes={min_minutes}")

        # Accumulators per stat. We collect raw residual arrays so we can
        # report both mean and median residuals — and break them down by
        # target magnitude to expose the right-skew interaction.
        residuals_mean: dict[str, list[float]] = {s: [] for s in STATS}
        residuals_q50: dict[str, list[float]] = {s: [] for s in STATS}
        actuals_by_stat: dict[str, list[float]] = {s: [] for s in STATS}

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
                pred_mean = dist.mean
                pred_q50 = dist.quantiles.get(0.5, dist.mean)
                residuals_mean[stat].append(pred_mean - actual)
                residuals_q50[stat].append(pred_q50 - actual)
                actuals_by_stat[stat].append(actual)

        logger.info("=" * 70)
        logger.info("Mean-head vs q=0.5 head residuals (positive = over-predicting)")
        logger.info("=" * 70)
        for stat in STATS:
            rm = np.array(residuals_mean[stat])
            rq = np.array(residuals_q50[stat])
            actuals = np.array(actuals_by_stat[stat])
            if len(rm) == 0:
                continue
            logger.info(f"\n  {stat}  n={len(rm)}  actual_mean={actuals.mean():.2f}")
            logger.info(
                f"    mean head:  mean_resid={rm.mean():+.3f}  "
                f"median_resid={np.median(rm):+.3f}  "
                f"|mean_resid|={np.abs(rm).mean():.3f}"
            )
            logger.info(
                f"    q=0.5 head: mean_resid={rq.mean():+.3f}  "
                f"median_resid={np.median(rq):+.3f}  "
                f"|mean_resid|={np.abs(rq).mean():.3f}"
            )
            # Bias broken out by actual magnitude — confirms the right-skew
            # hypothesis: if mean head over-predicts on low-actual games but
            # is fine or under-predicts on high-actual games, that's the
            # squared-error trying to hedge against the long right tail.
            if len(actuals) >= 50:
                qs = np.quantile(actuals, [0.25, 0.5, 0.75])
                bins = [
                    ("low   (q<25%)", actuals < qs[0]),
                    ("mid-lo (25-50)", (actuals >= qs[0]) & (actuals < qs[1])),
                    ("mid-hi (50-75)", (actuals >= qs[1]) & (actuals < qs[2])),
                    ("high  (q>75%)", actuals >= qs[2]),
                ]
                logger.info("    bias by actual-magnitude bucket (mean head | q=0.5 head):")
                for label, mask in bins:
                    if mask.sum() > 0:
                        logger.info(
                            f"      {label}  n={int(mask.sum()):4d}  "
                            f"mean_head={rm[mask].mean():+.3f}  "
                            f"q50_head={rq[mask].mean():+.3f}  "
                            f"actual_mean={actuals[mask].mean():.2f}"
                        )
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--min-minutes", type=float, default=8.0)
    args = parser.parse_args()
    main(args.start, args.end, args.min_minutes)
