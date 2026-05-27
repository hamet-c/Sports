"""
Master feature builder.

Composes feature group modules into a flat feature vector for one
(player, game). The model never sees DB rows directly — only features
built through this pipeline. Hard rule: training and inference use
identical inputs.

All sub-features take an `as_of` and only use data strictly before it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time

import pandas as pd
from sqlalchemy.orm import Session

from app.features.game_context import compute_game_context
from app.features.injury_features import compute_injury_features
from app.features.opponent_defense import compute_opponent_defense
from app.features.player_form import compute_player_form
from app.features.player_static import compute_player_static
from app.features.teammate_status import compute_teammate_status
from app.features.usage import compute_player_usage
from app.features.vegas_signals import compute_vegas_signals


# Stable canonical feature ordering. Training and inference both rely on this.
FEATURE_COLUMNS: tuple[str, ...] = (
    # form
    "pts_avg_5", "pts_avg_10", "pts_avg_season",
    "reb_avg_5", "reb_avg_10",
    "ast_avg_5", "ast_avg_10",
    "threes_avg_5", "threes_avg_10",
    "min_avg_5", "min_avg_10",
    "pts_ewma", "reb_ewma", "ast_ewma", "threes_ewma", "min_ewma",
    "pts_std_10", "reb_std_10", "ast_std_10",
    "games_played_l10", "games_played_season",
    # threes volume / efficiency / role (Phase 5.9)
    "threes_attempted_l10", "threes_pct_l10", "fg_share_threes_l10",
    # usage
    "usage_rate_l10", "started_pct_l10", "minutes_share_l10",
    # opponent defense
    "opp_def_rtg_l10", "opp_def_rtg_season",
    "opp_pts_allowed_l10",
    "opp_pace_l10", "opp_pace_season",
    "opp_threes_allowed_l10", "opp_3p_attempted_allowed_l10",
    "opp_games_played_l10",
    # opponent threes quality (Phase 5.9)
    "opp_3p_pct_allowed_l10", "opp_3pa_allowed_per_pace_l10",
    # context
    "is_home", "rest_days", "is_b2b",
    "games_in_last_3_days", "games_in_last_7_days",
    "is_playoff",
    # vegas
    "team_total", "opp_total", "total", "spread", "is_favorite",
    # injury (player being predicted)
    "status_ordinal", "is_questionable", "is_doubtful", "is_out",
    # roster / static
    "pos_guard", "pos_forward", "pos_center", "height_inches", "weight_lbs",
    # teammate availability
    "team_n_out", "team_n_doubtful", "team_n_questionable",
    "team_minutes_lost_estimate",
)


@dataclass
class FeatureVector:
    player_id: int
    game_id: int
    as_of: date
    features: dict[str, float | int | None]

    def to_pandas_row(self) -> pd.Series:
        return pd.Series({c: self.features.get(c) for c in FEATURE_COLUMNS})


class FeatureBuilder:
    """
    Builds one row of features for a (player, game). Caller supplies an
    `as_of` date — typically the game's date. Every underlying lookup
    filters strictly before that date so no future info leaks.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def build(
        self,
        player_id: int,
        game_id: int,
        as_of: date,
        market: dict[str, float | None] | None = None,
    ) -> FeatureVector:
        """
        market: optional pre-fetched vegas signals for the upcoming game,
            shape {"total": float, "spread_for_team": float}. If absent we
            attempt to load from GameMarket.
        """
        opponent_id, is_home, own_team_id, nba_id = self._game_context(game_id, player_id)

        player_log = self._load_player_game_log(player_id, before=as_of)

        features: dict[str, float | int | None] = {}

        # 1) form
        form = compute_player_form(player_log, as_of=as_of)
        features.update(asdict(form))

        # 2) usage (player-level)
        features.update(compute_player_usage(player_log, as_of=as_of))

        # 3) opponent defense
        opp_log = self._load_team_game_log(opponent_id, before=as_of) if opponent_id else pd.DataFrame()
        opp_def = compute_opponent_defense(opp_log, as_of=as_of)
        features.update(asdict(opp_def))

        # 4) game context
        game_dates = player_log["game_date"] if not player_log.empty else pd.Series(dtype="datetime64[ns]")
        ctx = compute_game_context(game_dates, as_of=as_of, is_home=is_home, nba_id=nba_id)
        features.update(asdict(ctx))

        # 5) vegas signals
        if market is None:
            market = self._load_market(game_id, is_home)
        vegas = compute_vegas_signals(
            total=market.get("total") if market else None,
            spread_for_team=market.get("spread_for_team") if market else None,
        )
        features.update(asdict(vegas))

        # 6) injury (player being predicted)
        injury_df = self._load_injury_reports(player_id)
        as_of_dt = datetime.combine(as_of, time.min)
        features.update(asdict(compute_injury_features(injury_df, as_of_dt=as_of_dt)))

        # 7) static roster
        player_obj = self._load_player(player_id)
        features.update(asdict(compute_player_static(
            position=player_obj.position if player_obj else None,
            height_inches=player_obj.height_inches if player_obj else None,
            weight_lbs=player_obj.weight_lbs if player_obj else None,
        )))

        # 8) teammate availability
        teammate_df = self._load_teammate_reports(
            player_id=player_id, team_id=own_team_id, before=as_of,
        )
        features.update(asdict(compute_teammate_status(
            teammate_reports=teammate_df, as_of_dt=as_of_dt,
        )))

        return FeatureVector(
            player_id=player_id, game_id=game_id, as_of=as_of, features=features,
        )

    # ------------------------------ DB loaders ------------------------------ #

    def _load_player_game_log(self, player_id: int, before: date) -> pd.DataFrame:
        from app.db.models import PlayerGameStats
        rows = (
            self.db.query(PlayerGameStats)
            .filter(PlayerGameStats.player_id == player_id)
            .filter(PlayerGameStats.game_date < before)
            .order_by(PlayerGameStats.game_date)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "game_date": r.game_date,
                "points": r.points,
                "rebounds": r.rebounds,
                "assists": r.assists,
                "threes_made": r.threes_made,
                "threes_attempted": r.threes_attempted,
                "field_goals_attempted": r.field_goals_attempted,
                "minutes": r.minutes,
                "usage_rate": r.usage_rate,
                "started": r.started,
            }
            for r in rows
        ])

    def _load_team_game_log(self, team_id: int, before: date) -> pd.DataFrame:
        from app.db.models import TeamGameStats
        rows = (
            self.db.query(TeamGameStats)
            .filter(TeamGameStats.team_id == team_id)
            .filter(TeamGameStats.game_date < before)
            .order_by(TeamGameStats.game_date)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "game_date": r.game_date,
                "points": r.points,
                "points_allowed": r.points_allowed,
                "pace": r.pace,
                "off_rating": r.off_rating,
                "def_rating": r.def_rating,
                "threes_made": r.threes_made,
                "threes_attempted": r.threes_attempted,
                "threes_allowed": r.threes_allowed,
                "threes_attempted_allowed": r.threes_attempted_allowed,
            }
            for r in rows
        ])

    def _load_injury_reports(self, player_id: int) -> pd.DataFrame:
        from app.db.models import InjuryReport
        rows = (
            self.db.query(InjuryReport)
            .filter(InjuryReport.player_id == player_id)
            .order_by(InjuryReport.report_datetime)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([
            {"report_datetime": r.report_datetime, "status": r.status}
            for r in rows
        ])

    def _game_context(
        self, game_id: int, player_id: int,
    ) -> tuple[int | None, bool | None, int | None, str | None]:
        """
        Returns (opponent_team_id, is_home, own_team_id, nba_id).

        For historical (training) rows we have a PlayerGameStats entry that
        records the player's team for THAT specific game (via
        opponent_team_id + is_home + the Game's home/away ids), which
        survives later trades. For future games we fall back to the
        player's current Player.team_id.

        `nba_id` is the external NBA game-id (e.g. "0042200401" — the prefix
        encodes season type). Prop_ingest stubs have nba_id=None, and the
        caller's playoff classifier falls back to a date heuristic for those.
        """
        from app.db.models import Game, Player, PlayerGameStats
        game = self.db.query(Game).filter(Game.id == game_id).one_or_none()
        if game is None:
            return None, None, None, None

        # Prefer the historical record if it exists.
        pgs = (
            self.db.query(PlayerGameStats)
            .filter(PlayerGameStats.player_id == player_id)
            .filter(PlayerGameStats.game_id == game_id)
            .one_or_none()
        )
        if pgs is not None and pgs.opponent_team_id is not None:
            opponent_id = pgs.opponent_team_id
            is_home = bool(pgs.is_home) if pgs.is_home is not None else None
            own_team_id = (
                game.home_team_id if opponent_id == game.away_team_id else game.away_team_id
            )
            return opponent_id, is_home, own_team_id, game.nba_id

        # Future game: derive from current roster.
        player = self.db.query(Player).filter(Player.id == player_id).one_or_none()
        if player is None or player.team_id is None:
            return None, None, None, game.nba_id
        is_home = player.team_id == game.home_team_id
        opponent_id = game.away_team_id if is_home else game.home_team_id
        return opponent_id, is_home, player.team_id, game.nba_id

    def _load_player(self, player_id: int):
        from app.db.models import Player
        return self.db.query(Player).filter(Player.id == player_id).one_or_none()

    def _load_teammate_reports(
        self, player_id: int, team_id: int | None, before: date,
    ) -> pd.DataFrame:
        """
        For every player on `team_id` (other than `player_id`) with at least
        one InjuryReport before `before`, return rows of
        (player_id, report_datetime, status, l10_minutes).

        l10_minutes is computed from PlayerGameStats — average of the
        teammate's last 10 games strictly before the as-of date. Used to
        weight the "minutes lost" estimate.

        Heavy queries are skipped when team_id is unknown — we early-return
        an empty frame so historical training rows (which currently have
        zero injury data anyway) cost nothing extra.
        """
        if team_id is None:
            return pd.DataFrame()
        from app.db.models import InjuryReport, Player, PlayerGameStats

        teammate_ids_q = (
            self.db.query(Player.id)
            .filter(Player.team_id == team_id)
            .filter(Player.id != player_id)
        )
        teammate_ids = [pid for (pid,) in teammate_ids_q.all()]
        if not teammate_ids:
            return pd.DataFrame()

        before_dt = datetime.combine(before, time.min)
        reports = (
            self.db.query(InjuryReport)
            .filter(InjuryReport.player_id.in_(teammate_ids))
            .filter(InjuryReport.report_datetime < before_dt)
            .all()
        )
        if not reports:
            return pd.DataFrame()

        # L10 minutes per teammate appearing in reports.
        reporting_ids = list({r.player_id for r in reports})
        rows_out = []
        for tid in reporting_ids:
            recent = (
                self.db.query(PlayerGameStats.minutes)
                .filter(PlayerGameStats.player_id == tid)
                .filter(PlayerGameStats.game_date < before)
                .order_by(PlayerGameStats.game_date.desc())
                .limit(10)
                .all()
            )
            mins = [m for (m,) in recent if m is not None]
            avg_min = sum(mins) / len(mins) if mins else 0.0
            for r in reports:
                if r.player_id != tid:
                    continue
                rows_out.append({
                    "player_id": r.player_id,
                    "report_datetime": r.report_datetime,
                    "status": r.status,
                    "l10_minutes": avg_min,
                })
        return pd.DataFrame(rows_out)

    def _load_market(self, game_id: int, is_home: bool | None) -> dict[str, float | None]:
        """Pull most recent GameMarket row for this game, average across books."""
        from app.db.models import GameMarket
        rows = (
            self.db.query(GameMarket)
            .filter(GameMarket.game_id == game_id)
            .order_by(GameMarket.captured_at.desc())
            .all()
        )
        if not rows:
            return {"total": None, "spread_for_team": None}
        # Latest snapshot per book, then average.
        latest_by_book: dict[str, GameMarket] = {}
        for r in rows:
            if r.book not in latest_by_book:
                latest_by_book[r.book] = r
        totals = [r.total for r in latest_by_book.values() if r.total is not None]
        spreads_home = [r.spread_home for r in latest_by_book.values() if r.spread_home is not None]
        if not totals or not spreads_home:
            return {"total": None, "spread_for_team": None}
        avg_total = sum(totals) / len(totals)
        avg_spread_home = sum(spreads_home) / len(spreads_home)
        spread_for_team = avg_spread_home if is_home else -avg_spread_home
        return {"total": float(avg_total), "spread_for_team": float(spread_for_team)}


# Convenience for callers that already have a built feature dict.
def feature_dict_to_dataframe(features: dict[str, float | int | None]) -> pd.DataFrame:
    """Render a single feature dict as a 1-row DataFrame in canonical column order.

    Coerces every column to float so XGBoost gets a numeric matrix; missing /
    None values become NaN, which XGBoost handles natively.
    """
    df = pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])
    return coerce_feature_frame(df)


def coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force all feature columns to float64. None / bool / object columns are
    converted; unparseable values become NaN. XGBoost requires numeric dtypes
    and treats NaN as a learnable missing-value branch.
    """
    out = df.copy()
    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = float("nan")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out[list(FEATURE_COLUMNS)]
