"""
XGBoost quantile regressor — distributional predictions.

Trains separate XGBoost models for quantiles 0.1, 0.25, 0.5, 0.75, 0.9 plus a
mean (squared error) model. After per-row prediction we monotonize the
quantile values so they are non-decreasing — quantile crossing is a known
issue with independent quantile fits and a hard requirement for valid
P(over) interpolation.

Reference: XGBoost >= 1.7 native quantile via objective='reg:quantileerror'.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from app.models.base import Distribution, Predictor


class XGBQuantileRegressor(Predictor):
    QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __init__(
        self,
        stat_type: str,
        model_version: str = "v0.1",
        non_negative: bool = True,
    ) -> None:
        self.stat_type = stat_type
        self.model_version = model_version
        self.non_negative = non_negative
        self.models: dict[float, xgb.XGBRegressor] = {}
        self.mean_model: xgb.XGBRegressor | None = None
        self.feature_columns: list[str] = []
        # Hyperparams used for the fit (so we can reload identically and
        # report what was tuned).
        self.fit_params: dict = {}

    # ------------------------------- training ------------------------------- #

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
        **xgb_kwargs,
    ) -> None:
        self.feature_columns = list(X.columns)
        default_params = dict(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=0,
        )
        params = {**default_params, **xgb_kwargs}
        self.fit_params = dict(params)

        # Mean (squared-error) model — used only for the `mean` field of Distribution
        # because the median-quantile point estimate is not always the best mean.
        self.mean_model = xgb.XGBRegressor(objective="reg:squarederror", **params)
        fit_kwargs: dict = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        self.mean_model.fit(X, y, sample_weight=sample_weight, **fit_kwargs)

        for q in self.QUANTILES:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q, **params,
            )
            model.fit(X, y, sample_weight=sample_weight)
            self.models[q] = model

    # ------------------------------- inference ------------------------------ #

    def predict(self, features: pd.DataFrame) -> list[Distribution]:
        if not self.models or self.mean_model is None:
            raise RuntimeError("Model not fitted")
        X = features[self.feature_columns]
        per_q_preds = {q: self.models[q].predict(X) for q in self.QUANTILES}
        mean_preds = self.mean_model.predict(X)

        distributions: list[Distribution] = []
        for i in range(len(X)):
            quantiles = {q: float(per_q_preds[q][i]) for q in self.QUANTILES}
            quantiles = self._monotonize(quantiles)
            if self.non_negative:
                quantiles = {q: max(0.0, v) for q, v in quantiles.items()}
            mean = float(mean_preds[i])
            if self.non_negative:
                mean = max(0.0, mean)
            distributions.append(Distribution(mean=mean, quantiles=quantiles))
        return distributions

    # ----------------------------- introspection ---------------------------- #

    def feature_importance(self, kind: str = "gain") -> dict[str, float]:
        """
        Mean importance across all 6 sub-models (mean + 5 quantiles),
        normalized to sum=1. Returns a dict keyed by feature name. Features
        XGBoost never split on are returned as 0.0.
        """
        if not self.models or self.mean_model is None:
            raise RuntimeError("Model not fitted")
        all_models = [self.mean_model, *self.models.values()]
        # XGBoost reports importance keyed as f0, f1, ... when feature names
        # weren't passed; we know our column order from feature_columns.
        per_feature: dict[str, float] = {f: 0.0 for f in self.feature_columns}
        for m in all_models:
            booster = m.get_booster()
            try:
                booster.feature_names = self.feature_columns
            except Exception:
                pass
            scores = booster.get_score(importance_type=kind)
            for fname, s in scores.items():
                # Defensive: some XGBoost versions still return f0/f1/...
                if fname.startswith("f") and fname[1:].isdigit():
                    idx = int(fname[1:])
                    if 0 <= idx < len(self.feature_columns):
                        fname = self.feature_columns[idx]
                if fname in per_feature:
                    per_feature[fname] += s
        total = sum(per_feature.values())
        if total > 0:
            per_feature = {k: v / total for k, v in per_feature.items()}
        return per_feature

    @staticmethod
    def _monotonize(quantiles: dict[float, float]) -> dict[float, float]:
        """Force non-decreasing values across ascending quantile probabilities."""
        items = sorted(quantiles.items(), key=lambda kv: kv[0])
        running = -float("inf")
        out: dict[float, float] = {}
        for q, v in items:
            if v < running:
                v = running
            running = v
            out[q] = v
        return out

    # ----------------------------- persistence ------------------------------ #

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "stat_type": self.stat_type,
                "model_version": self.model_version,
                "non_negative": self.non_negative,
                "feature_columns": self.feature_columns,
                "models": self.models,
                "mean_model": self.mean_model,
                "fit_params": self.fit_params,
            },
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path: str) -> "XGBQuantileRegressor":
        bundle = joblib.load(path)
        instance = cls(
            stat_type=bundle["stat_type"],
            model_version=bundle["model_version"],
            non_negative=bundle.get("non_negative", True),
        )
        instance.feature_columns = bundle["feature_columns"]
        instance.models = bundle["models"]
        instance.mean_model = bundle.get("mean_model")
        instance.fit_params = bundle.get("fit_params", {})
        return instance
