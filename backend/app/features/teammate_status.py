"""
Teammate availability features.

For the player being predicted, aggregate the latest injury status of
every other player on their team as of `as_of`. Captures the "starter
went down → backup gets a usage bump" effect that pure player-form
features cannot see.

Today the DB has no historical injury reports, so this feature degrades
to all-zeros for backfilled training rows. Once daily ingest accumulates
real ESPN injury data going forward, retraining will let the model
learn the lift.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class TeammateStatusFeatures:
    team_n_out: int
    team_n_doubtful: int
    team_n_questionable: int
    team_minutes_lost_estimate: float

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


_QUESTIONABLE = ("QUESTIONABLE", "DAY_TO_DAY")


def compute_teammate_status(
    teammate_reports: pd.DataFrame,
    as_of_dt: datetime,
) -> TeammateStatusFeatures:
    """
    teammate_reports columns expected:
        player_id, report_datetime, status, l10_minutes

    One row per (teammate, report). We resolve the latest report per
    teammate strictly before `as_of_dt`, then aggregate counts and
    total expected minutes lost.
    """
    if teammate_reports is None or teammate_reports.empty:
        return TeammateStatusFeatures(0, 0, 0, 0.0)

    df = teammate_reports.copy()
    df["report_datetime"] = pd.to_datetime(df["report_datetime"])
    df = df[df["report_datetime"] < as_of_dt]
    if df.empty:
        return TeammateStatusFeatures(0, 0, 0, 0.0)

    df = df.sort_values("report_datetime")
    latest = df.groupby("player_id", as_index=False).tail(1)

    n_out = int((latest["status"] == "OUT").sum())
    n_doubt = int((latest["status"] == "DOUBTFUL").sum())
    n_quest = int(latest["status"].isin(_QUESTIONABLE).sum())

    # Minutes-lost: full credit for OUT, half-credit for DOUBTFUL, quarter
    # for QUESTIONABLE. Reflects expected probability of actually missing.
    weights = latest["status"].map({
        "OUT": 1.0, "DOUBTFUL": 0.5, "QUESTIONABLE": 0.25, "DAY_TO_DAY": 0.25,
    }).fillna(0.0)
    minutes = pd.to_numeric(latest.get("l10_minutes"), errors="coerce").fillna(0.0)
    minutes_lost = float((weights * minutes).sum())

    return TeammateStatusFeatures(
        team_n_out=n_out,
        team_n_doubtful=n_doubt,
        team_n_questionable=n_quest,
        team_minutes_lost_estimate=minutes_lost,
    )
