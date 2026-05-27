"""Smoke train the XGB quantile model on synthetic data and check shape/monotonicity."""
import numpy as np
import pandas as pd

from app.models.xgb_quantile import XGBQuantileRegressor


def test_xgb_quantile_fit_predict_monotonic():
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1.0, n)
    y = 20.0 + 3.0 * x1 - 2.0 * x2 + noise
    X = pd.DataFrame({"x1": x1, "x2": x2})

    model = XGBQuantileRegressor(stat_type="synthetic", non_negative=False)
    model.fit(X, y, n_estimators=100, max_depth=3)

    preds = model.predict(X.head(10))
    for d in preds:
        qs = sorted(d.quantiles.items(), key=lambda kv: kv[0])
        for (_, lo), (_, hi) in zip(qs, qs[1:]):
            assert lo <= hi + 1e-6  # monotonic after _monotonize()
