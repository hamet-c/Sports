"""
Static player attributes: position one-hot + height + weight.

These don't change game-to-game and don't depend on `as_of`. They give
the model roster context that the form features can't infer (a 7-foot
center and a 6'2 guard with the same scoring average have very
different rebound/assist profiles).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlayerStaticFeatures:
    pos_guard: int | None
    pos_forward: int | None
    pos_center: int | None
    height_inches: int | None
    weight_lbs: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


_POSITION_LETTER_TO_KEY = {"G": "pos_guard", "F": "pos_forward", "C": "pos_center"}


def compute_player_static(
    position: str | None,
    height_inches: int | None,
    weight_lbs: int | None,
) -> PlayerStaticFeatures:
    """
    position: a single letter ("G"/"F"/"C") or None — bootstrap normalises
        commonplayerinfo's "Forward-Guard" style strings down to one letter.
    """
    if position is None:
        # Unknown position: leave one-hot all None so XGBoost treats them as
        # missing rather than learning "no position == guard".
        guard = forward = center = None
    else:
        key = _POSITION_LETTER_TO_KEY.get(position.upper())
        guard = 1 if key == "pos_guard" else 0
        forward = 1 if key == "pos_forward" else 0
        center = 1 if key == "pos_center" else 0

    return PlayerStaticFeatures(
        pos_guard=guard,
        pos_forward=forward,
        pos_center=center,
        height_inches=height_inches if height_inches else None,
        weight_lbs=weight_lbs if weight_lbs else None,
    )
