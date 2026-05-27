"""End-to-end FeatureBuilder smoke tests against an in-memory DB."""
from datetime import date

import pandas as pd

from app.db.models import Game, Player, PlayerGameStats, Team, TeamGameStats
from app.features.builder import FEATURE_COLUMNS, FeatureBuilder


def _seed(db) -> tuple[int, int, int]:
    home = Team(sport="nba", nba_id=1, abbreviation="HOM", full_name="Home")
    away = Team(sport="nba", nba_id=2, abbreviation="AWA", full_name="Away")
    db.add_all([home, away])
    db.flush()

    p = Player(sport="nba", nba_id=10, full_name="Test Player", team_id=home.id, is_active=True)
    db.add(p)
    db.flush()

    games = []
    for i, gd in enumerate(["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"]):
        g = Game(
            sport="nba",
            nba_id=f"00{i}",
            season="2023-24",
            game_date=date.fromisoformat(gd),
            home_team_id=home.id,
            away_team_id=away.id,
            is_completed=True,
        )
        db.add(g)
        db.flush()
        games.append(g)

    # Player rows for the first three; we'll predict the 4th.
    points_seq = [10, 20, 30]
    for g, pts in zip(games[:3], points_seq):
        db.add(
            PlayerGameStats(
                player_id=p.id,
                game_id=g.id,
                game_date=g.game_date,
                opponent_team_id=away.id,
                is_home=True,
                minutes=30.0,
                points=pts,
                rebounds=5,
                assists=4,
                threes_made=2,
                threes_attempted=5,
                started=True,
            )
        )
        db.add(
            TeamGameStats(
                team_id=away.id,
                game_id=g.id,
                game_date=g.game_date,
                opponent_team_id=home.id,
                is_home=False,
                points=110,
                points_allowed=100 + pts,
                pace=99.0,
                def_rating=112.0,
                threes_made=12,
                threes_attempted=30,
                threes_allowed=11,
                threes_attempted_allowed=28,
            )
        )
    db.commit()
    return p.id, games[-1].id, away.id


def test_builder_does_not_use_target_game(db):
    player_id, game_id, _ = _seed(db)
    fb = FeatureBuilder(db)
    fv = fb.build(player_id, game_id, as_of=date(2024, 1, 7))
    f = fv.features
    assert f["games_played_season"] == 3
    assert f["pts_avg_season"] == 20.0
    assert f["is_home"] == 1
    assert f["rest_days"] == 2  # 2024-01-05 -> 2024-01-07
    assert f["opp_def_rtg_l10"] == 112.0
    # All canonical columns present.
    for col in FEATURE_COLUMNS:
        assert col in f
