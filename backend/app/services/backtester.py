"""
Backtesting framework.

Walks through historical dates. For each date:
    1. Fetch completed games and their captured prop_lines.
    2. Build features using only data BEFORE that date (no leakage —
       FeatureBuilder enforces this with `as_of=game_date`).
    3. Generate predictions, compute edges, classify OVER/UNDER/PASS.
    4. Score against the actual stat value on PlayerGameStats.
    5. Aggregate ROI, hit rate, log loss, calibration buckets.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import log
from typing import Iterable

from loguru import logger
from sqlalchemy.orm import Session

from app.services.edge import american_to_decimal
from app.services.prediction_service import EdgeAnalysis, PredictionService


_STAT_COLUMN = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threes_made": "threes_made",
    "steals": "steals",
    "blocks": "blocks",
}


def _extract_actual(row, stat_type: str) -> float | None:
    if stat_type == "pra":
        if row.points is None or row.rebounds is None or row.assists is None:
            return None
        return float(row.points + row.rebounds + row.assists)
    col = _STAT_COLUMN.get(stat_type)
    if col is None:
        return None
    val = getattr(row, col, None)
    return None if val is None else float(val)


@dataclass
class BucketStats:
    n: int = 0
    hits: int = 0

    def hit_rate(self) -> float:
        return self.hits / self.n if self.n else 0.0


@dataclass
class BacktestResult:
    n_predictions: int = 0
    n_bets: int = 0
    n_wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0
    log_loss_sum: float = 0.0
    log_loss_n: int = 0
    by_stat: dict[str, "BacktestResult"] = field(default_factory=dict)
    calibration: dict[str, BucketStats] = field(default_factory=dict)

    @property
    def roi(self) -> float:
        return (self.total_returned - self.total_staked) / self.total_staked if self.total_staked else 0.0

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_bets if self.n_bets else 0.0

    @property
    def log_loss(self) -> float:
        return self.log_loss_sum / self.log_loss_n if self.log_loss_n else 0.0

    def to_dict(self) -> dict:
        return {
            "n_predictions": self.n_predictions,
            "n_bets": self.n_bets,
            "n_wins": self.n_wins,
            "win_rate": self.win_rate,
            "total_staked": self.total_staked,
            "total_returned": self.total_returned,
            "roi": self.roi,
            "log_loss": self.log_loss,
            "by_stat": {k: v.to_dict() for k, v in self.by_stat.items()},
            "calibration": {
                k: {"n": b.n, "hits": b.hits, "hit_rate": b.hit_rate()}
                for k, b in self.calibration.items()
            },
        }


class Backtester:
    def __init__(
        self,
        db: Session,
        prediction_service: PredictionService,
        edge_threshold: float = 0.05,
        stake_per_bet: float = 1.0,
    ) -> None:
        self.db = db
        self.svc = prediction_service
        self.edge_threshold = edge_threshold
        self.stake = stake_per_bet

    # --------------------------------- run ---------------------------------- #

    def run(self, start: date, end: date) -> BacktestResult:
        result = BacktestResult()
        current = start
        while current <= end:
            self._run_one_day(current, result)
            current += timedelta(days=1)
        logger.info(
            f"Backtest {start}..{end}  predictions={result.n_predictions} "
            f"bets={result.n_bets} win%={result.win_rate:.3f} ROI={result.roi:+.3f} "
            f"logloss={result.log_loss:.4f}"
        )
        return result

    def _run_one_day(self, day: date, result: BacktestResult) -> None:
        from app.db.models import Game, PlayerGameStats, PropLine

        games = (
            self.db.query(Game)
            .filter(Game.game_date == day)
            .filter(Game.is_completed.is_(True))
            .all()
        )
        if not games:
            return

        for game in games:
            prop_rows: Iterable[PropLine] = (
                self.db.query(PropLine)
                .filter(PropLine.game_id == game.id)
                .filter(PropLine.captured_at < self._tip_off_proxy(day))
                .all()
            )
            # Take the latest line per (player, stat, book) before tip-off.
            latest: dict[tuple[int, str, str], PropLine] = {}
            for p in prop_rows:
                key = (p.player_id, p.stat_type, p.book)
                if key not in latest or p.captured_at > latest[key].captured_at:
                    latest[key] = p

            for prop in latest.values():
                edge = self.svc.analyze_prop(
                    player_id=prop.player_id,
                    game_id=game.id,
                    as_of=day,
                    stat_type=prop.stat_type,
                    line=prop.line,
                    over_odds=prop.over_odds,
                    under_odds=prop.under_odds,
                    book=prop.book,
                )
                if edge is None:
                    continue
                actual_row = (
                    self.db.query(PlayerGameStats)
                    .filter(PlayerGameStats.player_id == prop.player_id)
                    .filter(PlayerGameStats.game_id == game.id)
                    .one_or_none()
                )
                if actual_row is None:
                    continue
                actual = _extract_actual(actual_row, prop.stat_type)
                if actual is None:
                    continue
                self._tally(result, edge, actual)

    # ---------------------------- bookkeeping ------------------------------ #

    def _tally(self, agg: BacktestResult, edge: EdgeAnalysis, actual: float) -> None:
        agg.n_predictions += 1
        stat_agg = agg.by_stat.setdefault(edge.stat_type, BacktestResult())
        stat_agg.n_predictions += 1

        # Log-loss against actual over/under outcome.
        actual_over = 1 if actual > edge.line else 0
        p = min(max(edge.over_probability, 1e-6), 1 - 1e-6)
        ll = -(actual_over * log(p) + (1 - actual_over) * log(1 - p))
        agg.log_loss_sum += ll
        agg.log_loss_n += 1
        stat_agg.log_loss_sum += ll
        stat_agg.log_loss_n += 1

        # Calibration buckets at 10% width.
        bucket = f"{int(edge.over_probability * 10) * 10:02d}-{int(edge.over_probability * 10) * 10 + 10:02d}"
        b = agg.calibration.setdefault(bucket, BucketStats())
        b.n += 1
        b.hits += actual_over

        # ROI tally on bets we'd have actually placed.
        if edge.recommendation == "PASS":
            return
        agg.n_bets += 1
        stat_agg.n_bets += 1
        agg.total_staked += self.stake
        stat_agg.total_staked += self.stake

        side_won = (
            (edge.recommendation == "OVER" and actual > edge.line)
            or (edge.recommendation == "UNDER" and actual < edge.line)
        )
        push = actual == edge.line
        if push:
            agg.total_returned += self.stake
            stat_agg.total_returned += self.stake
            return
        if side_won:
            odds = edge.over_odds if edge.recommendation == "OVER" else edge.under_odds
            decimal = american_to_decimal(odds)
            agg.n_wins += 1
            stat_agg.n_wins += 1
            agg.total_returned += self.stake * decimal
            stat_agg.total_returned += self.stake * decimal

    @staticmethod
    def _tip_off_proxy(day: date):
        """Use end-of-day as a permissive cutoff. In production replace with
        actual game tip-off timestamps and use them per-game."""
        from datetime import datetime, time
        return datetime.combine(day, time(23, 59, 59))
