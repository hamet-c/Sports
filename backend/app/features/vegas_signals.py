"""
Vegas / market signals as features.

Derive implied team total from spread + total. The market knows things —
it's our single most predictive non-stat feature.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VegasFeatures:
    team_total: float | None
    opp_total: float | None
    total: float | None
    spread: float | None  # negative if favored
    is_favorite: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


def compute_vegas_signals(
    total: float | None,
    spread_for_team: float | None,
) -> VegasFeatures:
    """
    spread_for_team: positive means underdog, negative means favorite. Convention:
        team_total = total/2 - spread_for_team / 2
    Equivalent: favorite implied total = total/2 + |spread|/2.
    """
    if total is None or spread_for_team is None:
        return VegasFeatures(
            team_total=None,
            opp_total=None,
            total=total,
            spread=spread_for_team,
            is_favorite=None,
        )

    team_total = total / 2.0 - spread_for_team / 2.0
    opp_total = total - team_total
    return VegasFeatures(
        team_total=float(team_total),
        opp_total=float(opp_total),
        total=float(total),
        spread=float(spread_for_team),
        is_favorite=int(spread_for_team < 0),
    )
