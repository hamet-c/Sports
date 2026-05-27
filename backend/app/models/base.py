"""
Common predictor interface. All models implement Predictor — the rest of
the codebase doesn't care what's underneath.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Distribution:
    """Discrete quantile representation of a continuous predictive distribution."""
    mean: float
    quantiles: dict[float, float]

    def probability_over(self, line: float) -> float:
        """Linearly interpolate the empirical CDF defined by `quantiles`."""
        # quantiles: {p: value at that quantile}; sort by value to monotonize.
        pairs = sorted(self.quantiles.items(), key=lambda kv: kv[1])
        if line <= pairs[0][1]:
            return float(1.0 - pairs[0][0])
        if line >= pairs[-1][1]:
            return float(1.0 - pairs[-1][0])
        for (q_lo, v_lo), (q_hi, v_hi) in zip(pairs, pairs[1:]):
            if v_lo <= line <= v_hi:
                if v_hi == v_lo:
                    cdf_at_line = q_lo
                else:
                    frac = (line - v_lo) / (v_hi - v_lo)
                    cdf_at_line = q_lo + frac * (q_hi - q_lo)
                return float(1.0 - cdf_at_line)
        return float(1.0 - pairs[-1][0])

    def probability_under(self, line: float) -> float:
        return 1.0 - self.probability_over(line)


class Predictor(ABC):
    stat_type: str = ""
    model_version: str = ""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> list[Distribution]: ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Predictor": ...
