"""
One-time bootstrap: pull historical seasons of NBA data into the DB.

    python scripts/bootstrap_data.py --seasons 2022-23 2023-24 2024-25

Steps:
    1. Upsert all teams (static).
    2. Upsert active players + cache static historical players seen in game logs.
    3. Per season, per active player: pull regular-season game log,
       upsert PlayerGameStats and Game.
    4. Per season, per team: pull team game log, upsert TeamGameStats.

Designed to be re-runnable. Idempotent via nba_id-keyed upserts.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger
from tqdm import tqdm

from app.core.logging import configure_logging
from app.data.nba_api_client import nba_client
from app.db.models import (
    Game,
    Player,
    PlayerGameStats,
    Team,
    TeamGameStats,
)
from app.db.session import SessionLocal, init_db
from app.utils.upsert import upsert


def _parse_matchup(matchup: str) -> tuple[str, str, bool]:
    """
    NBA matchup field is like 'LAL vs. BOS' (home) or 'LAL @ BOS' (away).
    Returns (own_abbr, opp_abbr, is_home).
    """
    matchup = matchup or ""
    if "vs." in matchup:
        own, opp = [p.strip() for p in matchup.split("vs.")]
        return own, opp, True
    if "@" in matchup:
        own, opp = [p.strip() for p in matchup.split("@")]
        return own, opp, False
    return matchup, "", False


def upsert_teams(db) -> dict[int, int]:
    """Returns mapping nba_id -> our internal team id."""
    teams_df = nba_client.get_all_teams()
    rows = []
    for _, t in teams_df.iterrows():
        rows.append({
            "sport": "nba",
            "nba_id": int(t["id"]),
            "abbreviation": t["abbreviation"],
            "full_name": t["full_name"],
        })
    upsert(db, Team, rows, index_elements=["nba_id"])
    db.commit()
    return {t.nba_id: t.id for t in db.query(Team).all()}


def upsert_active_players(db, team_map: dict[int, int]) -> dict[int, int]:
    """Returns mapping nba_id -> our internal player id (active only)."""
    players_df = nba_client.get_active_players()
    rows = []
    for _, p in players_df.iterrows():
        rows.append({
            "sport": "nba",
            "nba_id": int(p["id"]),
            "full_name": p["full_name"],
            "is_active": bool(p.get("is_active", True)),
        })
    upsert(db, Player, rows, index_elements=["nba_id"])
    db.commit()
    return {p.nba_id: p.id for p in db.query(Player).all()}


def _ensure_game(
    db,
    *,
    nba_game_id: str,
    season: str,
    game_date: date,
    home_team_id: int,
    away_team_id: int,
) -> int:
    existing = db.query(Game).filter(Game.nba_id == nba_game_id).one_or_none()
    if existing is not None:
        return existing.id
    g = Game(
        sport="nba",
        nba_id=nba_game_id,
        season=season,
        game_date=game_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        is_completed=True,
    )
    db.add(g)
    db.flush()
    return g.id


def _delete_existing_pgs(db, player_id: int, game_ids: list[int]) -> None:
    if not game_ids:
        return
    db.query(PlayerGameStats).filter(
        PlayerGameStats.player_id == player_id,
        PlayerGameStats.game_id.in_(game_ids),
    ).delete(synchronize_session=False)


SEASON_TYPES = ("Regular Season", "Playoffs")


def ingest_player_season(
    db,
    *,
    player_id: int,
    nba_player_id: int,
    season: str,
    team_abbr_to_id: dict[str, int],
) -> int:
    """Pull a player's season game log (regular season + playoffs) and upsert
    into PlayerGameStats + Game.
    """
    frames = []
    for season_type in SEASON_TYPES:
        try:
            part = nba_client.get_player_game_log(
                nba_player_id, season=season, season_type=season_type,
            )
        except Exception as e:
            logger.warning(
                f"Game log failed: nba_player={nba_player_id} season={season} "
                f"type={season_type}: {e}"
            )
            continue
        if not part.empty:
            frames.append(part)
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return 0

    # Build games and player rows.
    new_pgs: list[dict[str, Any]] = []
    new_game_ids: list[int] = []
    for _, row in df.iterrows():
        own_abbr, opp_abbr, is_home = _parse_matchup(str(row.get("matchup", "")))
        own_id = team_abbr_to_id.get(own_abbr)
        opp_id = team_abbr_to_id.get(opp_abbr)
        if own_id is None or opp_id is None:
            continue
        gd = pd.to_datetime(row["game_date"]).date()
        nba_game_id = str(row["game_id_nba"])
        home_id = own_id if is_home else opp_id
        away_id = opp_id if is_home else own_id
        gid = _ensure_game(
            db,
            nba_game_id=nba_game_id,
            season=season,
            game_date=gd,
            home_team_id=home_id,
            away_team_id=away_id,
        )
        new_pgs.append({
            "player_id": player_id,
            "game_id": gid,
            "game_date": gd,
            "opponent_team_id": opp_id,
            "is_home": is_home,
            "minutes": row.get("minutes"),
            "points": row.get("points"),
            "rebounds": row.get("rebounds"),
            "assists": row.get("assists"),
            "steals": row.get("steals"),
            "blocks": row.get("blocks"),
            "turnovers": row.get("turnovers"),
            "threes_made": row.get("threes_made"),
            "threes_attempted": row.get("threes_attempted"),
            "field_goals_made": row.get("field_goals_made"),
            "field_goals_attempted": row.get("field_goals_attempted"),
            "free_throws_made": row.get("free_throws_made"),
            "free_throws_attempted": row.get("free_throws_attempted"),
            "plus_minus": row.get("plus_minus"),
        })
        new_game_ids.append(gid)

    if not new_pgs:
        return 0
    _delete_existing_pgs(db, player_id, new_game_ids)
    db.bulk_insert_mappings(PlayerGameStats, new_pgs)
    db.commit()
    return len(new_pgs)


def ingest_team_season(
    db,
    *,
    team_id: int,
    nba_team_id: int,
    season: str,
    team_abbr_to_id: dict[str, int],
) -> int:
    frames = []
    for season_type in SEASON_TYPES:
        try:
            part = nba_client.get_team_game_log(
                nba_team_id, season=season, season_type=season_type,
            )
        except Exception as e:
            logger.warning(
                f"Team game log failed: nba_team={nba_team_id} season={season} "
                f"type={season_type}: {e}"
            )
            continue
        if not part.empty:
            frames.append(part)
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return 0

    new_rows: list[dict[str, Any]] = []
    new_game_ids: list[int] = []
    # We need games already in DB; team logs lack matchup parsing here so we use
    # the GAME_ID linkage (already inserted from player ingest where possible).
    matchup_col = "matchup" if "matchup" in df.columns else "MATCHUP"
    for _, row in df.iterrows():
        nba_game_id = str(row["game_id_nba"])
        game = db.query(Game).filter(Game.nba_id == nba_game_id).one_or_none()
        if game is None:
            # Best-effort backfill from this row.
            own_abbr, opp_abbr, is_home = _parse_matchup(str(row.get(matchup_col, "")))
            own_id = team_abbr_to_id.get(own_abbr)
            opp_id = team_abbr_to_id.get(opp_abbr)
            if own_id is None or opp_id is None:
                continue
            gd = pd.to_datetime(row["game_date"]).date()
            home_id = own_id if is_home else opp_id
            away_id = opp_id if is_home else own_id
            gid = _ensure_game(
                db,
                nba_game_id=nba_game_id,
                season=season,
                game_date=gd,
                home_team_id=home_id,
                away_team_id=away_id,
            )
        else:
            gid = game.id
            gd = game.game_date
        own_abbr, opp_abbr, is_home = _parse_matchup(str(row.get(matchup_col, "")))
        opp_id = team_abbr_to_id.get(opp_abbr)
        new_rows.append({
            "team_id": team_id,
            "game_id": gid,
            "game_date": gd,
            "opponent_team_id": opp_id,
            "is_home": is_home,
            "points": row.get("points"),
            "threes_made": row.get("threes_made"),
            "threes_attempted": row.get("threes_attempted"),
        })
        new_game_ids.append(gid)

    if not new_rows:
        return 0
    db.query(TeamGameStats).filter(
        TeamGameStats.team_id == team_id,
        TeamGameStats.game_id.in_(new_game_ids),
    ).delete(synchronize_session=False)
    db.bulk_insert_mappings(TeamGameStats, new_rows)
    db.commit()
    return len(new_rows)


def _primary_position_letter(pos: str | None) -> str | None:
    """commonplayerinfo POSITION is "Guard", "Forward-Guard", "Center", etc.
    Reduce to a single primary letter G/F/C; None if unparseable."""
    if not pos:
        return None
    p = str(pos).strip().upper()
    if not p:
        return None
    first = p[0]
    return first if first in ("G", "F", "C") else None


def _height_to_inches(height) -> int | None:
    if height is None:
        return None
    s = str(height).strip()
    if "-" not in s:
        return None
    try:
        ft, inch = s.split("-", 1)
        return int(ft) * 12 + int(inch)
    except (ValueError, AttributeError):
        return None


def _weight_to_lbs(weight) -> int | None:
    if weight is None:
        return None
    s = str(weight).strip().split()
    if not s:
        return None
    try:
        return int(s[0])
    except ValueError:
        return None


def fill_player_static(db) -> tuple[int, int]:
    """
    Fill Player.position / team_id / height_inches / weight_lbs from
    nba_api commonplayerinfo. Idempotent — re-running just refreshes
    current values.

    Returns (n_updated, n_failed).
    """
    players = (
        db.query(Player)
        .filter(Player.is_active.is_(True))
        .filter(Player.nba_id.isnot(None))
        .all()
    )
    team_nba_to_id = {t.nba_id: t.id for t in db.query(Team).all()}
    updated = 0
    failed = 0
    for player in tqdm(players, desc="player info"):
        try:
            df = nba_client.get_common_player_info(player.nba_id)
        except Exception as e:
            logger.warning(f"commonplayerinfo failed for {player.full_name} ({player.nba_id}): {e}")
            failed += 1
            continue
        if df.empty:
            failed += 1
            continue
        row = df.iloc[0]
        # nba_api column casing varies; normalize once.
        cols = {c.upper(): c for c in df.columns}

        pos = _primary_position_letter(row.get(cols.get("POSITION", "POSITION")))
        height_in = _height_to_inches(row.get(cols.get("HEIGHT", "HEIGHT")))
        weight_lbs = _weight_to_lbs(row.get(cols.get("WEIGHT", "WEIGHT")))

        team_id = None
        nba_team_raw = row.get(cols.get("TEAM_ID", "TEAM_ID"))
        if nba_team_raw is not None and not pd.isna(nba_team_raw) and int(nba_team_raw) != 0:
            team_id = team_nba_to_id.get(int(nba_team_raw))

        if pos is not None:
            player.position = pos
        if height_in is not None:
            player.height_inches = height_in
        if weight_lbs is not None:
            player.weight_lbs = weight_lbs
        if team_id is not None:
            player.team_id = team_id
        updated += 1
        # Commit periodically so a crash doesn't lose all progress.
        if updated % 25 == 0:
            db.commit()
    db.commit()
    return updated, failed


def _safe_float(val) -> float | None:
    """Coerce a nba_api numeric cell to float, returning None for NaN/missing."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    if pd.isna(f):
        return None
    return f


def _parse_started(start_position) -> bool:
    """START_POSITION is 'G'/'F'/'C' for starters, '' for bench / DNP."""
    if start_position is None:
        return False
    return bool(str(start_position).strip())


def fill_advanced_box_scores(
    db, *, only_missing: bool = True, limit: int | None = None,
) -> tuple[int, int, int]:
    """
    For every completed Game, fetch BoxScoreAdvancedV3 and populate:

      PlayerGameStats: usage_rate (usagePercentage), started (position non-empty)
      TeamGameStats:   off_rating, def_rating, pace

    Idempotency: when ``only_missing=True`` (the default), skip games where
    both TeamGameStats rows already have ``def_rating`` set — those have been
    backfilled in a prior run. Pass ``only_missing=False`` to force a full
    refresh of every game.

    ``limit`` caps the number of *processed* games (skips don't count). Useful
    for smoke tests or chunked runs.

    Returns ``(processed, skipped, failed)``.
    """
    player_nba_to_id = {
        p.nba_id: p.id
        for p in db.query(Player).filter(Player.nba_id.isnot(None)).all()
    }
    team_nba_to_id = {t.nba_id: t.id for t in db.query(Team).all()}

    games = (
        db.query(Game)
        .filter(Game.is_completed.is_(True))
        .order_by(Game.game_date)
        .all()
    )

    processed = 0
    skipped = 0
    failed = 0

    for game in tqdm(games, desc="advanced box scores"):
        if limit is not None and processed >= limit:
            break
        if only_missing:
            tgs_rows = (
                db.query(TeamGameStats)
                .filter(TeamGameStats.game_id == game.id)
                .all()
            )
            if len(tgs_rows) == 2 and all(t.def_rating is not None for t in tgs_rows):
                skipped += 1
                continue

        try:
            player_df, team_df = nba_client.get_box_score_advanced(game.nba_id)
        except Exception as e:
            logger.warning(f"BoxScoreAdvancedV2 failed for game {game.nba_id}: {e}")
            failed += 1
            continue

        if not player_df.empty:
            for _, row in player_df.iterrows():
                nba_pid_raw = row.get("personId")
                if nba_pid_raw is None or pd.isna(nba_pid_raw):
                    continue
                pid = player_nba_to_id.get(int(nba_pid_raw))
                if pid is None:
                    continue
                pgs = (
                    db.query(PlayerGameStats)
                    .filter(PlayerGameStats.player_id == pid)
                    .filter(PlayerGameStats.game_id == game.id)
                    .one_or_none()
                )
                if pgs is None:
                    continue
                usage = _safe_float(row.get("usagePercentage"))
                if usage is not None:
                    pgs.usage_rate = usage
                pgs.started = _parse_started(row.get("position"))

        if not team_df.empty:
            for _, row in team_df.iterrows():
                nba_tid_raw = row.get("teamId")
                if nba_tid_raw is None or pd.isna(nba_tid_raw):
                    continue
                tid = team_nba_to_id.get(int(nba_tid_raw))
                if tid is None:
                    continue
                tgs = (
                    db.query(TeamGameStats)
                    .filter(TeamGameStats.team_id == tid)
                    .filter(TeamGameStats.game_id == game.id)
                    .one_or_none()
                )
                if tgs is None:
                    continue
                off = _safe_float(row.get("offensiveRating"))
                deff = _safe_float(row.get("defensiveRating"))
                pace = _safe_float(row.get("pace"))
                if off is not None:
                    tgs.off_rating = off
                if deff is not None:
                    tgs.def_rating = deff
                if pace is not None:
                    tgs.pace = pace

        processed += 1
        if processed % 50 == 0:
            db.commit()

    db.commit()
    return processed, skipped, failed


def fill_team_game_allowed(db) -> int:
    """
    Once both teams' rows exist for a game, fill in points_allowed and
    threes_allowed / threes_attempted_allowed from the opponent row.
    """
    rows_updated = 0
    games = db.query(Game).all()
    for game in games:
        sides = (
            db.query(TeamGameStats)
            .filter(TeamGameStats.game_id == game.id)
            .all()
        )
        if len(sides) != 2:
            continue
        a, b = sides
        if a.points is not None and b.points is not None:
            a.points_allowed = b.points
            b.points_allowed = a.points
        if a.threes_made is not None and b.threes_made is not None:
            a.threes_allowed = b.threes_made
            b.threes_allowed = a.threes_made
        if a.threes_attempted is not None and b.threes_attempted is not None:
            a.threes_attempted_allowed = b.threes_attempted
            b.threes_attempted_allowed = a.threes_attempted
        rows_updated += 2
    db.commit()
    return rows_updated


def bootstrap_seasons(seasons: list[str]) -> None:
    init_db()
    db = SessionLocal()
    try:
        logger.info("Loading teams")
        team_map_nba_to_id = upsert_teams(db)
        team_abbr_to_id = {t.abbreviation: t.id for t in db.query(Team).all()}

        logger.info("Loading active players")
        player_map_nba_to_id = upsert_active_players(db, team_map_nba_to_id)

        for season in seasons:
            logger.info(f"Season {season}: ingesting team game logs")
            for team in tqdm(db.query(Team).all(), desc=f"teams {season}"):
                ingest_team_season(
                    db,
                    team_id=team.id,
                    nba_team_id=team.nba_id,
                    season=season,
                    team_abbr_to_id=team_abbr_to_id,
                )

            logger.info(f"Season {season}: ingesting player game logs")
            players = db.query(Player).filter(Player.is_active.is_(True)).all()
            n_rows_total = 0
            for player in tqdm(players, desc=f"players {season}"):
                if player.nba_id is None:
                    continue
                n_rows_total += ingest_player_season(
                    db,
                    player_id=player.id,
                    nba_player_id=player.nba_id,
                    season=season,
                    team_abbr_to_id=team_abbr_to_id,
                )
            logger.info(f"Season {season}: ingested {n_rows_total} player-game rows")

        logger.info("Filling team allowed stats from opposite side")
        n = fill_team_game_allowed(db)
        logger.info(f"Updated {n} TeamGameStats with allowed columns")

        logger.info("Filling player static info (position/team/height/weight)")
        u, f = fill_player_static(db)
        logger.info(f"Player static fill: {u} updated, {f} failed")

        logger.info("Filling advanced box-score data (usage / started / def_rtg / pace)")
        p, s, fail = fill_advanced_box_scores(db, only_missing=True)
        logger.info(f"Advanced box-score fill: {p} processed, {s} skipped, {fail} failed")

        logger.info("Bootstrap complete")
    finally:
        db.close()


def bootstrap_player_static_only() -> None:
    """Run only the commonplayerinfo fill — useful when seasons already loaded."""
    init_db()
    db = SessionLocal()
    try:
        u, f = fill_player_static(db)
        logger.info(f"Player static fill: {u} updated, {f} failed")
    finally:
        db.close()


def bootstrap_advanced_only(force: bool = False) -> None:
    """Run only the BoxScoreAdvancedV2 backfill across all completed games."""
    init_db()
    db = SessionLocal()
    try:
        p, s, fail = fill_advanced_box_scores(db, only_missing=not force)
        logger.info(f"Advanced box-score fill: {p} processed, {s} skipped, {fail} failed")
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=["2023-24"])
    parser.add_argument(
        "--players-only",
        action="store_true",
        help="Skip game-log ingestion; only refresh player static info (position/team/height/weight).",
    )
    parser.add_argument(
        "--advanced-only",
        action="store_true",
        help="Skip game-log ingestion; only fetch BoxScoreAdvancedV2 for completed games (usage/started/def_rtg/pace).",
    )
    parser.add_argument(
        "--force-advanced",
        action="store_true",
        help="When combined with --advanced-only, re-fetch every game (default skips games already filled).",
    )
    args = parser.parse_args()
    if args.players_only:
        bootstrap_player_static_only()
    elif args.advanced_only:
        bootstrap_advanced_only(force=args.force_advanced)
    else:
        bootstrap_seasons(args.seasons)
