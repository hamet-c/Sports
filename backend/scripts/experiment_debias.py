"""
Controlled experiment: retrain a single stat with custom XGBoost hyperparams
and measure bucket bias against the val window.

Goal: figure out whether the magnitude shrinkage in the Phase 5.6 models is
caused by over-regularization (min_child_weight, reg_alpha) or insufficient
capacity (max_depth, n_estimators). The Phase 5.6 baseline uses
xgb_quantile.py defaults: max_depth=6, lr=0.04, n=600, min_child_weight=5,
reg_alpha=0.1, reg_lambda=1.0.

The retrained model is NEVER saved to data/models/ — it lives only in memory
for the duration of the script. So this is safe to run repeatedly. The
production joblibs stay untouched.

Usage:
    .venv\\Scripts\\python.exe scripts\\experiment_debias.py \\
        --stat points --min-child-weight 1 --reg-alpha 0
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from app.core.logging import configure_logging
from app.db.models import PlayerGameStats
from app.db.session import SessionLocal, init_db
from app.features.builder import FEATURE_COLUMNS, FeatureBuilder, coerce_feature_frame
from app.models.xgb_quantile import XGBQuantileRegressor


# Imported here so we can monkey-poke the train script's helpers without
# duplicating them.
from scripts.train_models import (
    _drop_label_nans,
    _recency_weights,
    build_training_dataframe,
    split_train_val,
)


def _to_dataframe_row(features: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])


def bucket_bias(predictions: np.ndarray, actuals: np.ndarray) -> dict[str, dict]:
    """Report (n, mean_resid) per actual-magnitude quartile."""
    if len(actuals) < 50:
        return {}
    qs = np.quantile(actuals, [0.25, 0.5, 0.75])
    masks = {
        "low   (q<25%)": actuals < qs[0],
        "mid-lo (25-50)": (actuals >= qs[0]) & (actuals < qs[1]),
        "mid-hi (50-75)": (actuals >= qs[1]) & (actuals < qs[2]),
        "high  (q>75%)": actuals >= qs[2],
    }
    out: dict[str, dict] = {}
    for label, mask in masks.items():
        if mask.sum() > 0:
            resid = predictions[mask] - actuals[mask]
            out[label] = {
                "n": int(mask.sum()),
                "mean_resid": float(resid.mean()),
                "actual_mean": float(actuals[mask].mean()),
            }
    return out


def main(
    stat: str,
    train_end: date,
    val_end: date,
    min_child_weight: int,
    reg_alpha: float,
    reg_lambda: float,
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    recency_half_life: float | None,
    min_minutes: float,
) -> None:
    init_db()
    db = SessionLocal()
    try:
        df = build_training_dataframe(db, train_end=train_end, val_end=val_end, min_minutes=min_minutes)
        train_df, val_df = split_train_val(df, train_end, val_end)
        logger.info(f"train rows={len(train_df)}  val rows={len(val_df)}")

        feature_cols = list(FEATURE_COLUMNS)
        X_train = train_df[feature_cols]
        y_train = train_df[stat]
        if recency_half_life is None:
            w_train = np.ones(len(train_df))
            logger.info("recency weights: DISABLED (uniform weights)")
        else:
            anchor = train_df["as_of"].max()
            w_train = _recency_weights(train_df["as_of"], anchor, half_life_days=recency_half_life)
            logger.info(f"recency weights: half_life_days={recency_half_life}")
        X_train, y_train, w_train = _drop_label_nans(X_train, y_train, w_train)

        # Validation rows for bucket-bias eval.
        val_mask = val_df[stat].notna()
        X_val = val_df.loc[val_mask, feature_cols]
        y_val = val_df.loc[val_mask, stat].to_numpy(dtype=float)
        logger.info(f"val rows with label: {len(y_val)}")

        custom_params = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )
        logger.info(f"custom hyperparams: {custom_params}")

        # Train a fresh XGBQuantileRegressor with the custom params.
        # Note: _drop_label_nans already returns y_train as a numpy array.
        model = XGBQuantileRegressor(stat_type=stat, model_version="experiment")
        logger.info("Fitting custom model — this overrides defaults (mean head + 5 quantiles).")
        model.fit(X_train, y_train, sample_weight=w_train, **custom_params)

        # Predict val.
        logger.info("Predicting val set")
        dists = model.predict(X_val)
        mean_preds = np.array([d.mean for d in dists])
        q50_preds = np.array([d.quantiles[0.5] for d in dists])

        # Aggregate metrics.
        mean_resid = mean_preds - y_val
        q50_resid = q50_preds - y_val
        logger.info("=" * 70)
        logger.info(f"EXPERIMENT RESULTS  stat={stat}  custom={custom_params}")
        logger.info("=" * 70)
        logger.info(
            f"  mean head:  mean_resid={mean_resid.mean():+.3f}  "
            f"|mean_resid|={np.abs(mean_resid).mean():.3f}"
        )
        logger.info(
            f"  q=0.5 head: mean_resid={q50_resid.mean():+.3f}  "
            f"|mean_resid|={np.abs(q50_resid).mean():.3f}"
        )
        logger.info("  bucket bias (mean head | q=0.5 head):")
        b_mean = bucket_bias(mean_preds, y_val)
        b_q50 = bucket_bias(q50_preds, y_val)
        for label in b_mean:
            mh = b_mean[label]
            qh = b_q50.get(label, {})
            logger.info(
                f"    {label}  n={mh['n']:4d}  "
                f"mean_head={mh['mean_resid']:+.3f}  "
                f"q50_head={qh.get('mean_resid', float('nan')):+.3f}  "
                f"actual_mean={mh['actual_mean']:.2f}"
            )

        logger.info("\nCompare against Phase 5.6 baseline (md=6, lr=0.04, n=600, mcw=5, ra=0.1):")
        logger.info(f"  baseline {stat} mean head bucket bias was:")
        if stat == "points":
            logger.info("    low=+5.32  mid-lo=+1.95  mid-hi=-0.97  high=-5.99")
        elif stat == "rebounds":
            logger.info("    low=+2.44  mid-lo=+1.16  mid-hi=+0.03  high=-2.37")
        elif stat == "assists":
            logger.info("    low=+1.63  mid-lo=+0.97  mid-hi=+0.19  high=-1.72")
        elif stat == "threes_made":
            logger.info("    low (n/a)  mid-lo=+0.90  mid-hi=+0.47  high=-1.14")
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--stat", choices=("points", "rebounds", "assists", "threes_made"), required=True)
    p.add_argument("--train-end", type=date.fromisoformat, default=date(2026, 2, 15))
    p.add_argument("--val-end", type=date.fromisoformat, default=date(2026, 4, 13))
    p.add_argument("--min-child-weight", type=int, default=5)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument(
        "--recency-half-life",
        type=float,
        default=365.0,
        help="Set to a very large number (e.g. 1e9) to effectively disable; "
             "negative value to truly disable (uniform weights).",
    )
    p.add_argument("--min-minutes", type=float, default=8.0)
    args = p.parse_args()
    half_life = None if args.recency_half_life < 0 else args.recency_half_life
    main(
        stat=args.stat,
        train_end=args.train_end,
        val_end=args.val_end,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        recency_half_life=half_life,
        min_minutes=args.min_minutes,
    )
