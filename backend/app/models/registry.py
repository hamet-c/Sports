"""
Model registry — discovers and loads all per-stat .joblib artifacts at startup.

Convention:
    {models_dir}/{stat}_xgbq.joblib                   -> XGBQuantileRegressor
    {models_dir}/{stat}_xgbq_calibration.joblib       -> IsotonicCalibrator (optional)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.models.base import Distribution, Predictor
from app.models.calibration import IsotonicCalibrator
from app.models.xgb_quantile import XGBQuantileRegressor


@dataclass
class RegisteredModel:
    predictor: Predictor
    calibrator: IsotonicCalibrator | None

    def predict_with_probability(
        self, features, line: float,
    ) -> tuple[Distribution, float]:
        """Return distribution and calibrated P(over)."""
        dist = self.predictor.predict(features)[0]
        raw_p = dist.probability_over(line)
        if self.calibrator is not None and self.calibrator.is_fitted():
            import numpy as np
            cal_p = float(self.calibrator.transform(np.array([raw_p]))[0])
            return dist, cal_p
        return dist, raw_p


class ModelRegistry:
    def __init__(self, models_dir: Path | None = None, *, use_calibrators: bool = True) -> None:
        self.models_dir = Path(models_dir or settings.models_dir)
        self.use_calibrators = use_calibrators
        self._models: dict[str, RegisteredModel] = {}

    def load(self) -> None:
        self._models.clear()
        if not self.models_dir.exists():
            logger.warning(f"Models dir does not exist: {self.models_dir}")
            return
        for path in self.models_dir.glob("*_xgbq.joblib"):
            stat = path.stem.replace("_xgbq", "")
            try:
                pred = XGBQuantileRegressor.load(str(path))
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue
            cal: IsotonicCalibrator | None = None
            if self.use_calibrators:
                cal_path = self.models_dir / f"{stat}_xgbq_calibration.joblib"
                if cal_path.exists():
                    try:
                        cal = IsotonicCalibrator.load(str(cal_path))
                    except Exception as e:
                        logger.warning(f"Failed to load calibrator {cal_path}: {e}")
            self._models[stat] = RegisteredModel(predictor=pred, calibrator=cal)
            logger.info(
                f"Registered model for stat={stat} version={pred.model_version} "
                f"calibrated={cal is not None}"
            )

    def get(self, stat_type: str) -> RegisteredModel | None:
        return self._models.get(stat_type)

    def all_stats(self) -> list[str]:
        return list(self._models.keys())

    def __contains__(self, stat_type: str) -> bool:  # pragma: no cover - trivial
        return stat_type in self._models

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._models)


registry = ModelRegistry(use_calibrators=settings.use_calibrators)
