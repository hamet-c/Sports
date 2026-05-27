"""
Injury-report features for the player being predicted.

Latest report status before `as_of_dt` is what matters. Status is encoded
both ordinally (0..4) and as a few one-hot signals for tree models.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


_STATUS_ORDINAL = {
    "AVAILABLE": 4,
    "PROBABLE": 3,
    "QUESTIONABLE": 2,
    "DAY_TO_DAY": 2,
    "DOUBTFUL": 1,
    "OUT": 0,
}


@dataclass
class InjuryFeatures:
    status_ordinal: int | None  # 0=OUT, 4=AVAILABLE
    is_questionable: int | None
    is_doubtful: int | None
    is_out: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


def compute_injury_features(
    reports: pd.DataFrame,
    as_of_dt: datetime,
) -> InjuryFeatures:
    """
    reports columns expected: report_datetime, status.
    Picks the latest report strictly before `as_of_dt`.
    """
    if reports is None or reports.empty:
        return InjuryFeatures(
            status_ordinal=None, is_questionable=None,
            is_doubtful=None, is_out=None,
        )

    df = reports.copy()
    df["report_datetime"] = pd.to_datetime(df["report_datetime"])
    df = df[df["report_datetime"] < as_of_dt].sort_values("report_datetime")
    if df.empty:
        return InjuryFeatures(
            status_ordinal=None, is_questionable=None,
            is_doubtful=None, is_out=None,
        )

    latest = df.iloc[-1]["status"]
    ord_val = _STATUS_ORDINAL.get(latest)
    return InjuryFeatures(
        status_ordinal=ord_val,
        is_questionable=int(latest in ("QUESTIONABLE", "DAY_TO_DAY")),
        is_doubtful=int(latest == "DOUBTFUL"),
        is_out=int(latest == "OUT"),
    )
