"""
Probability calibration via isotonic regression.

Fits an isotonic mapping from raw P(over) to empirical hit rate over a
held-out set. We fit one calibrator per (stat_type, model_version).

Usage:
    cal = IsotonicCalibrator()
    cal.fit(raw_probs, outcomes)   # outcomes ∈ {0, 1}
    p_calibrated = cal.transform(raw_probs)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None

    def fit(self, probabilities: np.ndarray, outcomes: np.ndarray) -> None:
        if len(probabilities) != len(outcomes):
            raise ValueError("probabilities and outcomes length mismatch")
        if len(probabilities) == 0:
            raise ValueError("cannot fit calibrator on empty data")
        self._iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True,
        )
        self._iso.fit(np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=float))

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return np.asarray(probabilities, dtype=float)
        return self._iso.transform(np.asarray(probabilities, dtype=float))

    def is_fitted(self) -> bool:
        return self._iso is not None

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"iso": self._iso}, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        bundle = joblib.load(path)
        c = cls()
        c._iso = bundle["iso"]
        return c
