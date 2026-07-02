"""
Train one quantile regressor per stat type, fit isotonic calibration on a
held-out validation slice, and save artifacts.

    python scripts/train_models.py --train-end 2024-04-01 --val-end 2024-06-15

The training dataframe is built by walking PlayerGameStats rows; for each
game we call FeatureBuilder.build(player_id, game_id, as_of=game_date) so
no future info ever leaks. Time-aware split: train < train_end, val in
[train_end, val_end). Sample weights decay exponentially with age.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from tqdm import tqdm

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.timeutil import utcnow_iso_z
from app.db.models import PlayerGameStats, PropLine
from app.db.session import SessionLocal, init_db
from app.features.builder import FEATURE_COLUMNS, FeatureBuilder, coerce_feature_frame
from app.models.calibration import IsotonicCalibrator
from app.models.xgb_quantile import XGBQuantileRegressor


STATS_TO_MODEL = ["points", "rebounds", "assists", "threes_made"]
STAT_TO_COLUMN = {s: s for s in STATS_TO_MODEL}


def build_training_dataframe(
    db,
    train_end: date,
    val_end: date,
    min_minutes: float = 8.0,
) -> pd.DataFrame:
    """
    Produce a single dataframe with feature columns + label columns +
    `as_of` (game_date) for time-aware splitting.
    """
    builder = FeatureBuilder(db)
    rows = (
        db.query(PlayerGameStats)
        .filter(PlayerGameStats.game_date < val_end)
        .filter(PlayerGameStats.minutes != None)  # noqa: E711
        .order_by(PlayerGameStats.game_date)
        .all()
    )
    logger.info(f"Loaded {len(rows)} player-game rows for training/val build")

    records: list[dict] = []
    for r in tqdm(rows, desc="building features"):
        if r.minutes is None or r.minutes < min_minutes:
            continue
        feat_vec = builder.build(r.player_id, r.game_id, as_of=r.game_date)
        rec = {c: feat_vec.features.get(c) for c in FEATURE_COLUMNS}
        rec["as_of"] = r.game_date
        rec["player_id"] = r.player_id
        rec["game_id"] = r.game_id
        for stat in STATS_TO_MODEL:
            rec[stat] = getattr(r, STAT_TO_COLUMN[stat])
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    df["as_of"] = pd.to_datetime(df["as_of"])
    # Coerce feature columns to numeric so XGBoost accepts them. Label columns
    # and metadata stay as-is.
    feature_block = coerce_feature_frame(df[list(FEATURE_COLUMNS)])
    for c in FEATURE_COLUMNS:
        df[c] = feature_block[c]
    return df


def split_train_val(
    df: pd.DataFrame, train_end: date, val_end: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train = df[df["as_of"] < train_end_ts].copy()
    val = df[(df["as_of"] >= train_end_ts) & (df["as_of"] < val_end_ts)].copy()
    return train, val


def _recency_weights(as_of: pd.Series, anchor: pd.Timestamp, half_life_days: float = 365.0) -> np.ndarray:
    """Exponential decay; weight = 0.5 ** (days_old / half_life)."""
    days_old = (anchor - as_of).dt.days.clip(lower=0)
    return np.power(0.5, days_old.to_numpy() / half_life_days)


def _drop_label_nans(X: pd.DataFrame, y: pd.Series, w: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mask = y.notna().to_numpy()
    return X.loc[mask].reset_index(drop=True), y.loc[mask].to_numpy(dtype=float), w[mask]


def _tune_hyperparams(
    train_df: pd.DataFrame, val_df: pd.DataFrame, stat: str, grid_size: str,
) -> dict:
    """
    Search a small grid by fitting only the median (q=0.5) sub-model and
    scoring on val MAE. Returns the winning hyperparam dict (empty if val
    is unusable). Final model.fit will refit all 6 sub-models with these.
    """
    if grid_size == "small":
        grid = [
            {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 800},
            {"max_depth": 4, "learning_rate": 0.07, "n_estimators": 800},
            {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 800},
            {"max_depth": 8, "learning_rate": 0.07, "n_estimators": 800},
        ]
    else:  # full
        grid = [
            {"max_depth": d, "learning_rate": lr, "n_estimators": n}
            for d in (4, 6, 8)
            for lr in (0.03, 0.05, 0.07)
            for n in (400, 800)
        ]

    feature_cols = list(FEATURE_COLUMNS)
    val_mask = val_df[stat].notna()
    if val_df.empty or not val_mask.any():
        logger.info(f"  tune {stat}: no val rows, skipping search")
        return {}

    anchor = train_df["as_of"].max()
    X_train = train_df[feature_cols]
    y_train = train_df[stat]
    w_train = _recency_weights(train_df["as_of"], anchor)
    X_train, y_train, w_train = _drop_label_nans(X_train, y_train, w_train)

    X_val = val_df.loc[val_mask, feature_cols]
    y_val = val_df.loc[val_mask, stat].to_numpy(dtype=float)

    base = dict(
        objective="reg:quantileerror", quantile_alpha=0.5,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", n_jobs=0,
    )

    best_cfg: dict = {}
    best_mae = float("inf")
    for cfg in grid:
        m = xgb.XGBRegressor(**{**base, **cfg})
        m.fit(X_train, y_train, sample_weight=w_train)
        preds = m.predict(X_val)
        mae = float(np.mean(np.abs(preds - y_val)))
        logger.info(
            f"  tune {stat} cfg=md={cfg['max_depth']} "
            f"lr={cfg['learning_rate']} n={cfg['n_estimators']}  val_mae={mae:.4f}"
        )
        if mae < best_mae:
            best_mae = mae
            best_cfg = cfg
    logger.info(f"  best {stat}: {best_cfg} val_mae={best_mae:.4f}")
    return best_cfg


def train_for_stat(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    stat: str,
    tuned_params: dict | None = None,
    fit_calibrator: bool = True,
    model_version: str = "v0.2",
) -> tuple[XGBQuantileRegressor, IsotonicCalibrator | None]:
    feature_cols = list(FEATURE_COLUMNS)
    anchor = train_df["as_of"].max()

    X_train = train_df[feature_cols]
    y_train = train_df[stat]
    w_train = _recency_weights(train_df["as_of"], anchor)
    X_train, y_train, w_train = _drop_label_nans(X_train, y_train, w_train)

    model = XGBQuantileRegressor(stat_type=stat, model_version=model_version)
    eval_pair = None
    if not val_df.empty:
        X_val = val_df[feature_cols]
        y_val = val_df[stat]
        if y_val.notna().any():
            mask = y_val.notna()
            eval_pair = (X_val.loc[mask].reset_index(drop=True), y_val.loc[mask].to_numpy(dtype=float))

    fit_kwargs = dict(tuned_params or {})
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=eval_pair, **fit_kwargs)

    cal: IsotonicCalibrator | None = None
    if fit_calibrator and not val_df.empty and val_df[stat].notna().any():
        cal = _fit_calibrator_against_market(model, val_df, stat)
    return model, cal


def compute_coverage(df: pd.DataFrame, stat: str) -> dict[str, float]:
    """% non-null per feature column on the rows that have a label for this stat."""
    valid = df[df[stat].notna()]
    if valid.empty:
        return {c: 0.0 for c in FEATURE_COLUMNS}
    total = len(valid)
    return {c: float(valid[c].notna().sum() / total) for c in FEATURE_COLUMNS}


_BASELINE_COL = {
    "points": "pts_avg_10",
    "rebounds": "reb_avg_10",
    "assists": "ast_avg_10",
    "threes_made": "threes_avg_10",
}


def _fit_calibrator_against_market(
    model: XGBQuantileRegressor,
    val_df: pd.DataFrame,
    stat: str,
) -> IsotonicCalibrator | None:
    """
    Fit isotonic calibration on per-row synthetic-line outcomes, matching the
    line construction used by run_backtest.py synthetic mode (L10 mean rounded
    to nearest .5, .0 bumped up to .5 so there are no pushes). Each val row
    contributes one (raw_p_over, actual_over) pair so the calibrator sees the
    full distribution of P(over) — not just one point near the val median.

    Rows missing the L10 baseline fall back to the predicted median as the
    line; if both are missing the row is dropped. Pushes (actual == line)
    cannot occur given the .5 rounding, but the strict > comparison below
    also handles it correctly if it did.
    """
    feature_cols = list(FEATURE_COLUMNS)
    valid = val_df[val_df[stat].notna()].copy()
    if valid.empty:
        return None
    dists = model.predict(valid[feature_cols])

    baseline_col = _BASELINE_COL[stat]
    baselines = (
        valid[baseline_col].to_numpy(dtype=float)
        if baseline_col in valid.columns
        else np.full(len(valid), np.nan)
    )
    actuals = valid[stat].to_numpy(dtype=float)

    raw_probs: list[float] = []
    outcomes: list[int] = []
    for i, dist in enumerate(dists):
        b = baselines[i]
        if np.isnan(b):
            line = float(dist.quantiles.get(0.5, dist.mean))
        else:
            line = round(float(b) * 2) / 2
            if line == int(line):
                line += 0.5
        raw_probs.append(float(dist.probability_over(line)))
        outcomes.append(1 if actuals[i] > line else 0)

    cal = IsotonicCalibrator()
    try:
        cal.fit(np.array(raw_probs), np.array(outcomes))
    except Exception as e:
        logger.warning(f"Calibrator fit failed for {stat}: {e}")
        return None
    logger.info(
        f"  calibrator {stat}: fit on {len(raw_probs)} synthetic-line rows "
        f"(over rate={np.mean(outcomes):.3f}, mean raw p_over={np.mean(raw_probs):.3f})"
    )
    return cal


def evaluate(model: XGBQuantileRegressor, val_df: pd.DataFrame, stat: str) -> dict:
    if val_df.empty or not val_df[stat].notna().any():
        return {"n": 0}
    feature_cols = list(FEATURE_COLUMNS)
    sub = val_df[val_df[stat].notna()].copy()
    dists = model.predict(sub[feature_cols])
    means = np.array([d.mean for d in dists])
    y = sub[stat].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(means - y)))
    rmse = float(np.sqrt(np.mean((means - y) ** 2)))
    # Naive baseline: each player's L10 mean.
    baseline_map = {
        "points": "pts_avg_10",
        "rebounds": "reb_avg_10",
        "assists": "ast_avg_10",
        "threes_made": "threes_avg_10",
    }
    baseline_col = baseline_map.get(stat)
    if baseline_col and baseline_col in sub.columns and sub[baseline_col].notna().any():
        baseline_mae = float(np.mean(np.abs(sub[baseline_col].fillna(means.mean()) - y)))
    else:
        baseline_mae = float("nan")
    return {
        "n": int(len(sub)),
        "mae": mae,
        "rmse": rmse,
        "baseline_l10_mae": baseline_mae,
    }


def _backup_existing(path) -> None:
    """Copy an artifact aside before overwrite. The .bak-<ts> suffix must
    never match registry.load()'s *_xgbq.joblib glob."""
    if path.exists():
        backup = path.parent / f"{path.name}.bak-{utcnow_iso_z().replace(':', '')[:-1]}"
        shutil.copy2(path, backup)
        logger.info(f"Backed up existing artifact to {backup}")


def main(
    train_end: date, val_end: date, tune: str, skip_calibrator: bool,
    model_version: str,
) -> None:
    init_db()
    db = SessionLocal()
    try:
        df = build_training_dataframe(db, train_end=train_end, val_end=val_end)
        if df.empty:
            logger.error("No training rows. Run bootstrap_data.py first.")
            return
        train_df, val_df = split_train_val(df, train_end, val_end)
        logger.info(f"train rows={len(train_df)}  val rows={len(val_df)}")

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        coverage_per_stat: dict[str, dict[str, float]] = {}

        for stat in STATS_TO_MODEL:
            coverage_per_stat[stat] = compute_coverage(df, stat)

            logger.info(f"Training model for {stat}")
            tuned: dict = {}
            if tune != "none":
                logger.info(f"  hyperparam tune ({tune}) for {stat}")
                tuned = _tune_hyperparams(train_df, val_df, stat, grid_size=tune)

            model, calibrator = train_for_stat(
                train_df, val_df, stat,
                tuned_params=tuned,
                fit_calibrator=not skip_calibrator,
                model_version=model_version,
            )
            metrics = evaluate(model, val_df, stat)
            logger.info(f"{stat} metrics: {metrics}")

            out = settings.models_dir / f"{stat}_xgbq.joblib"
            _backup_existing(out)
            model.save(str(out))
            logger.info(f"Saved {out}")
            if calibrator is not None:
                cal_out = settings.models_dir / f"{stat}_xgbq_calibration.joblib"
                _backup_existing(cal_out)
                calibrator.save(str(cal_out))
                logger.info(f"Saved {cal_out}")
            elif skip_calibrator:
                logger.info(
                    f"  --skip-calibrator set: leaving existing "
                    f"{stat}_xgbq_calibration.joblib untouched"
                )

        # Persist coverage report for the API to surface.
        reports_dir = settings.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        coverage_path = reports_dir / "feature_coverage.json"
        coverage_path.write_text(json.dumps(
            {
                "generated_at": utcnow_iso_z(),
                "train_end": train_end.isoformat(),
                "val_end": val_end.isoformat(),
                "feature_columns": list(FEATURE_COLUMNS),
                "coverage": coverage_per_stat,
            },
            indent=2,
            default=float,
        ))
        logger.info(f"Wrote coverage report to {coverage_path}")
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", type=date.fromisoformat, required=True)
    parser.add_argument("--val-end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--tune",
        choices=("none", "small", "full"),
        default="small",
        help=(
            "Hyperparam grid size. 'small' = 4 configs (max_depth × lr) ~2 min/stat. "
            "'full' = 18 configs ~10 min/stat. 'none' = use defaults."
        ),
    )
    parser.add_argument(
        "--skip-calibrator",
        action="store_true",
        help=(
            "Don't fit or save isotonic calibrators. Set this whenever you "
            "retrain WITHOUT also implementing a real-line calibrator fit: "
            "the existing synthetic-line fit was rolled back in Phase 5.5 "
            "because it regressed real-line OVER win-rate from 46.2%% to "
            "32.7%%. With this flag the per-stat *_xgbq_calibration.joblib "
            "files are left untouched. Pair with use_calibrators=False."
        ),
    )
    parser.add_argument(
        "--model-version",
        default=date.today().isoformat(),
        help=(
            "Version string stamped into the joblib bundle (default: today's "
            "date). Surfaces in registry logs and RecommendationLog rows."
        ),
    )
    args = parser.parse_args()
    main(args.train_end, args.val_end, args.tune, args.skip_calibrator,
         args.model_version)
