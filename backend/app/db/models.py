"""
Database schema.

Design notes:
- Normalized: players/teams/games separate, stats reference them.
- Time-aware: every stat row carries game date for "as of" filtering.
- Multi-sport-ready: `sport` column on Player and Team.
- Predictions persisted with their feature vector for audit + calibration.
- External NBA stats IDs preserved as `nba_id` so re-runs of bootstrap upsert
  rather than duplicate.
"""
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db.session import Base


class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    sport = Column(String(8), nullable=False, default="nba", index=True)
    nba_id = Column(Integer, unique=True, index=True)  # external NBA team id
    abbreviation = Column(String(4), nullable=False)
    full_name = Column(String(64), nullable=False)
    conference = Column(String(8))
    division = Column(String(16))
    __table_args__ = (UniqueConstraint("sport", "abbreviation"),)


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    sport = Column(String(8), nullable=False, default="nba", index=True)
    nba_id = Column(Integer, unique=True, index=True)
    full_name = Column(String(128), nullable=False, index=True)
    position = Column(String(8))
    height_inches = Column(Integer)
    weight_lbs = Column(Integer)
    team_id = Column(Integer, ForeignKey("teams.id"), index=True)
    is_active = Column(Boolean, default=True)
    team = relationship("Team")


class Game(Base):
    __tablename__ = "games"
    id = Column(Integer, primary_key=True)
    sport = Column(String(8), nullable=False, default="nba", index=True)
    nba_id = Column(String(16), unique=True, index=True)  # NBA GAME_ID is a string
    season = Column(String(8), nullable=False, index=True)
    game_date = Column(Date, nullable=False, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    is_completed = Column(Boolean, default=False, index=True)
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


class PlayerGameStats(Base):
    """One row per player per game. The factual record."""
    __tablename__ = "player_game_stats"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    opponent_team_id = Column(Integer, ForeignKey("teams.id"))
    is_home = Column(Boolean)

    minutes = Column(Float)
    points = Column(Integer)
    rebounds = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    threes_made = Column(Integer)
    threes_attempted = Column(Integer)
    field_goals_made = Column(Integer)
    field_goals_attempted = Column(Integer)
    free_throws_made = Column(Integer)
    free_throws_attempted = Column(Integer)
    plus_minus = Column(Float)
    usage_rate = Column(Float)
    started = Column(Boolean)

    __table_args__ = (
        UniqueConstraint("player_id", "game_id"),
        Index("ix_pgs_player_date", "player_id", "game_date"),
        Index("ix_pgs_game_date", "game_date"),
    )
    player = relationship("Player")
    game = relationship("Game")


class TeamGameStats(Base):
    """One row per team per game. Used for opponent defense features."""
    __tablename__ = "team_game_stats"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    opponent_team_id = Column(Integer, ForeignKey("teams.id"))
    is_home = Column(Boolean)

    points = Column(Integer)
    points_allowed = Column(Integer)
    pace = Column(Float)
    off_rating = Column(Float)
    def_rating = Column(Float)
    threes_made = Column(Integer)
    threes_attempted = Column(Integer)
    threes_allowed = Column(Integer)
    threes_attempted_allowed = Column(Integer)

    __table_args__ = (
        UniqueConstraint("team_id", "game_id"),
        Index("ix_tgs_team_date", "team_id", "game_date"),
    )


class InjuryReport(Base):
    __tablename__ = "injury_reports"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    report_datetime = Column(DateTime, nullable=False, index=True)
    status = Column(String(32), nullable=False)  # OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, AVAILABLE
    description = Column(String(256))
    __table_args__ = (Index("ix_injury_player_dt", "player_id", "report_datetime"),)


class PropLine(Base):
    __tablename__ = "prop_lines"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    book = Column(String(32), nullable=False)
    stat_type = Column(String(16), nullable=False)
    line = Column(Float, nullable=False)
    over_odds = Column(Integer, nullable=False)
    under_odds = Column(Integer, nullable=False)
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    __table_args__ = (
        Index("ix_prop_lookup", "player_id", "game_id", "stat_type", "book"),
        Index("ix_prop_game_stat", "game_id", "stat_type"),
    )


class GameMarket(Base):
    """Game-level Vegas signals (spread/total) for vegas-signal features."""
    __tablename__ = "game_markets"
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    book = Column(String(32), nullable=False)
    spread_home = Column(Float)
    total = Column(Float)
    home_moneyline = Column(Integer)
    away_moneyline = Column(Integer)
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    __table_args__ = (UniqueConstraint("game_id", "book", "captured_at"),)


class Prediction(Base):
    """Every model output is logged for audit + calibration."""
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    stat_type = Column(String(16), nullable=False)
    model_version = Column(String(32), nullable=False)

    predicted_mean = Column(Float, nullable=False)
    predicted_p10 = Column(Float)
    predicted_p25 = Column(Float)
    predicted_p50 = Column(Float)
    predicted_p75 = Column(Float)
    predicted_p90 = Column(Float)

    line = Column(Float)
    over_probability = Column(Float)
    expected_value_over = Column(Float)
    expected_value_under = Column(Float)

    features_json = Column(JSON)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    actual_value = Column(Float)
    __table_args__ = (Index("ix_pred_pg_stat", "player_id", "game_id", "stat_type"),)
