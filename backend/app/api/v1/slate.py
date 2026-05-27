"""Today's slate — every available prop with our projection, line, and edge."""
from datetime import date as date_type, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.db.models import Game, Player, PlayerGameStats, PropLine
from app.db.session import get_db
from app.models.registry import registry
from app.services.prediction_service import PredictionService

router = APIRouter()


class SlateProp(BaseModel):
    player_id: int
    player_name: str
    game_id: int
    game_date: date_type
    matchup: str
    team_abbr: str
    opp_abbr: str
    home_abbr: str
    away_abbr: str
    is_home: bool
    stat_type: str
    line: float
    over_odds: int
    under_odds: int
    book: str
    predicted_mean: float
    over_probability: float
    expected_value_over: float
    expected_value_under: float
    kelly_over: float
    kelly_under: float
    recommendation: str
    sharp_book_disagreement: bool
    book_favored_side: str


class SlateResponse(BaseModel):
    date: date_type
    props: list[SlateProp]


def _infer_player_team_ids(
    db: Session, player_ids: set[int],
) -> dict[int, int]:
    """
    Fallback team resolver: for each player, find the team they played for in
    their most recent PlayerGameStats row (derived from Game.home_team_id /
    Game.away_team_id and PGS.is_home). Used when Player.team_id is stale or
    NULL (e.g. fill_player_static never ran, or post-trade roster drift).

    Returns {player_id: inferred_team_id}; players with no PGS history are
    omitted.
    """
    if not player_ids:
        return {}
    # Pull every PGS row for the candidate players joined to its Game, then
    # pick the most recent (date, game_id) per player in Python. This avoids
    # composite tuple IN clauses, which SQLite doesn't support.
    rows = (
        db.query(PlayerGameStats, Game)
        .join(Game, Game.id == PlayerGameStats.game_id)
        .filter(PlayerGameStats.player_id.in_(player_ids))
        .all()
    )
    # player_id -> (game_date, game_id, team_id) for the latest row seen.
    best: dict[int, tuple[date_type, int, int]] = {}
    for pgs, game in rows:
        team_id = game.home_team_id if pgs.is_home else game.away_team_id
        if team_id is None:
            continue
        key = (pgs.game_date, pgs.game_id, team_id)
        prev = best.get(pgs.player_id)
        if prev is None or (key[0], key[1]) > (prev[0], prev[1]):
            best[pgs.player_id] = key
    return {pid: tid for pid, (_, _, tid) in best.items()}


def _resolve_player_team_id(
    game: Game, player_team_id: int | None, inferred_team_id: int | None,
) -> int | None:
    """
    Pick the team_id that actually appears in this game. Prefer
    Player.team_id when it matches a side; otherwise fall back to the
    inferred team from recent PlayerGameStats.
    """
    if player_team_id in (game.home_team_id, game.away_team_id):
        return player_team_id
    if inferred_team_id in (game.home_team_id, game.away_team_id):
        return inferred_team_id
    return None


def _team_context(
    game: Game, player_team_id: int | None,
) -> tuple[str, str, str, str, bool, str]:
    """Returns (team_abbr, opp_abbr, home_abbr, away_abbr, is_home, matchup_label)."""
    home = game.home_team.abbreviation if game.home_team else "?"
    away = game.away_team.abbreviation if game.away_team else "?"
    if player_team_id == game.home_team_id:
        return home, away, home, away, True, f"vs {away}"
    if player_team_id == game.away_team_id:
        return away, home, home, away, False, f"@ {home}"
    # Player's team_id doesn't match either side — likely a stale roster row
    # (post-trade) or a two-way player. The game itself is still well-defined,
    # so surface the game's home/away even though we can't say which side the
    # player is on. Falling back to is_home=False keeps the prop visible in
    # one of the columns rather than dropping it.
    return home, away, home, away, False, f"{away} @ {home}"


@router.get("/", response_model=SlateResponse)
def get_slate(
    target_date: date_type | None = None,
    min_edge: float = Query(0.0, ge=0.0, le=1.0),
    book: str | None = None,
    stat_types: list[str] | None = Query(None),
    db: Session = Depends(get_db),
) -> SlateResponse:
    target = target_date or date_type.today()

    games = (
        db.query(Game)
        .options(joinedload(Game.home_team), joinedload(Game.away_team))
        .filter(Game.game_date == target)
        .all()
    )
    if not games:
        return SlateResponse(date=target, props=[])

    game_ids = [g.id for g in games]
    game_by_id = {g.id: g for g in games}

    prop_query = db.query(PropLine).filter(PropLine.game_id.in_(game_ids))
    if book:
        prop_query = prop_query.filter(PropLine.book == book)
    if stat_types:
        prop_query = prop_query.filter(PropLine.stat_type.in_(stat_types))
    prop_rows = prop_query.all()

    # For each (player, game, stat, book) keep the most recent line.
    latest: dict[tuple[int, int, str, str], PropLine] = {}
    for p in prop_rows:
        key = (p.player_id, p.game_id, p.stat_type, p.book)
        if key not in latest or p.captured_at > latest[key].captured_at:
            latest[key] = p

    if not latest:
        return SlateResponse(date=target, props=[])

    # Bulk-load players referenced.
    player_ids = {p.player_id for p in latest.values()}
    players = {
        pl.id: pl
        for pl in db.query(Player).filter(Player.id.in_(player_ids)).all()
    }
    # Fallback team resolution for players whose Player.team_id is missing or
    # doesn't match the game (stale roster, two-way contracts, etc.).
    inferred_team_by_player = _infer_player_team_ids(db, player_ids)

    svc = PredictionService(
        db, registry,
        edge_threshold=settings.edge_threshold,
        over_edge_threshold=settings.edge_threshold_over,
    )
    out: list[SlateProp] = []
    cache: dict[tuple[int, int], dict] = {}  # (player_id, game_id) -> {stat: Distribution}

    for prop in latest.values():
        if prop.stat_type not in registry:
            continue
        player = players.get(prop.player_id)
        game = game_by_id.get(prop.game_id)
        if player is None or game is None:
            continue

        key = (prop.player_id, prop.game_id)
        if key not in cache:
            _, dists = svc.predict_player_game(
                prop.player_id, prop.game_id, target, stat_types=registry.all_stats(),
            )
            cache[key] = dists
        dist = cache[key].get(prop.stat_type)
        if dist is None:
            continue

        edge = svc.analyze_distribution(
            player_id=prop.player_id,
            game_id=prop.game_id,
            stat_type=prop.stat_type,
            distribution=dist,
            line=prop.line,
            over_odds=prop.over_odds,
            under_odds=prop.under_odds,
            book=prop.book,
        )
        max_ev = max(edge.expected_value_over, edge.expected_value_under)
        if max_ev < min_edge:
            continue

        resolved_team_id = _resolve_player_team_id(
            game, player.team_id, inferred_team_by_player.get(player.id),
        )
        team_abbr, opp_abbr, home_abbr, away_abbr, is_home, matchup = _team_context(
            game, resolved_team_id,
        )
        out.append(
            SlateProp(
                player_id=prop.player_id,
                player_name=player.full_name,
                game_id=prop.game_id,
                game_date=game.game_date,
                matchup=matchup,
                team_abbr=team_abbr,
                opp_abbr=opp_abbr,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                is_home=is_home,
                stat_type=prop.stat_type,
                line=prop.line,
                over_odds=prop.over_odds,
                under_odds=prop.under_odds,
                book=prop.book,
                predicted_mean=edge.predicted_mean,
                over_probability=edge.over_probability,
                expected_value_over=edge.expected_value_over,
                expected_value_under=edge.expected_value_under,
                kelly_over=edge.kelly_over,
                kelly_under=edge.kelly_under,
                recommendation=edge.recommendation,
                sharp_book_disagreement=edge.sharp_book_disagreement,
                book_favored_side=edge.book_favored_side,
            )
        )

    out.sort(
        key=lambda p: max(p.expected_value_over, p.expected_value_under),
        reverse=True,
    )
    return SlateResponse(date=target, props=out)


class SlateAnchorResponse(BaseModel):
    today: date_type
    latest_prop_date: date_type | None
    days_stale: int | None


@router.get("/anchor", response_model=SlateAnchorResponse)
def slate_anchor(db: Session = Depends(get_db)) -> SlateAnchorResponse:
    """
    Tells the frontend the most recent date for which we have captured
    prop_lines, so it can center a multi-day window even when daily ingest
    has stalled. ``days_stale`` is (today - latest); 0 means fresh.
    """
    from sqlalchemy import func
    today = date_type.today()
    latest = (
        db.query(func.max(Game.game_date))
        .join(PropLine, PropLine.game_id == Game.id)
        .scalar()
    )
    stale = (today - latest).days if latest is not None else None
    return SlateAnchorResponse(
        today=today,
        latest_prop_date=latest,
        days_stale=stale,
    )


# ============================ Recommendation record ============================ #


class RecRecordSide(BaseModel):
    n: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None


class RecRecordByStat(BaseModel):
    n: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None


class RecRecordResponse(BaseModel):
    start: date_type
    end: date_type
    n_recommendations: int
    wins: int
    losses: int
    pushes: int
    win_rate: float | None
    by_stat: dict[str, RecRecordByStat]
    over: RecRecordSide
    under: RecRecordSide
    note: str


def _empty_side() -> dict:
    return {"n": 0, "wins": 0, "losses": 0, "pushes": 0}


def _finalize_side(d: dict) -> RecRecordSide:
    settled = d["n"] - d["pushes"]
    return RecRecordSide(
        n=d["n"],
        wins=d["wins"],
        losses=d["losses"],
        pushes=d["pushes"],
        win_rate=(d["wins"] / settled) if settled > 0 else None,
    )


@router.get("/recommendation_record", response_model=RecRecordResponse)
def recommendation_record(
    days: int = Query(7, ge=1, le=60),
    db: Session = Depends(get_db),
) -> RecRecordResponse:
    """
    Grade the *current* model's recommendation against actual outcomes for every
    captured prop_line in the last ``days`` days.

    Caveat: predictions are computed on demand using today's model, not the
    model that was deployed when the prop was offered. So this is "how would
    this model have done?" not "how did we actually do?". To get real
    historical recs we would need a RecommendationLog table.
    """
    today = date_type.today()
    start = today - timedelta(days=days)
    # Only consider games that have finished, i.e. game_date is strictly before
    # today. Today's games may still be in progress.
    end = today - timedelta(days=1)

    # We deliberately don't filter on Game.is_completed — stub games created
    # by the prop_ingest flow never get that flag set. Instead we rely on the
    # PlayerGameStats join below to filter to games that actually finished
    # (a game with no PGS rows is treated as not yet playable).
    games = (
        db.query(Game)
        .options(joinedload(Game.home_team), joinedload(Game.away_team))
        .filter(Game.game_date >= start)
        .filter(Game.game_date <= end)
        .all()
    )
    if not games:
        return RecRecordResponse(
            start=start, end=end, n_recommendations=0, wins=0, losses=0, pushes=0,
            win_rate=None, by_stat={}, over=_finalize_side(_empty_side()),
            under=_finalize_side(_empty_side()),
            note="No games in window.",
        )
    # Prop ingest sometimes creates a stub Game row with no nba_id, separate
    # from the real Game row populated by bootstrap. Map each stub to its real
    # counterpart by (date, home, away) so we can pull actuals from the real
    # game's PlayerGameStats.
    stub_to_real: dict[int, int] = {}
    by_match: dict[tuple, int] = {}
    for g in games:
        if g.nba_id is not None:
            by_match[(g.game_date, g.home_team_id, g.away_team_id)] = g.id
    for g in games:
        if g.nba_id is None:
            real_id = by_match.get((g.game_date, g.home_team_id, g.away_team_id))
            if real_id is not None:
                stub_to_real[g.id] = real_id

    game_ids = [g.id for g in games]
    game_by_id = {g.id: g for g in games}

    prop_rows = (
        db.query(PropLine)
        .filter(PropLine.game_id.in_(game_ids))
        .all()
    )
    # Latest line per (player, game, stat, book) — same dedupe as the slate.
    latest: dict[tuple[int, int, str, str], PropLine] = {}
    for p in prop_rows:
        key = (p.player_id, p.game_id, p.stat_type, p.book)
        if key not in latest or p.captured_at > latest[key].captured_at:
            latest[key] = p
    if not latest:
        return RecRecordResponse(
            start=start, end=end, n_recommendations=0, wins=0, losses=0, pushes=0,
            win_rate=None, by_stat={}, over=_finalize_side(_empty_side()),
            under=_finalize_side(_empty_side()),
            note="No prop lines captured for this window.",
        )

    # Bulk-load actual outcomes for every real game we might need. The lookup
    # key is the *real* game id, not the stub the prop was attached to.
    real_game_ids = list({stub_to_real.get(gid, gid) for gid in game_ids})
    pgs_rows = (
        db.query(PlayerGameStats)
        .filter(PlayerGameStats.game_id.in_(real_game_ids))
        .all()
    )
    actuals_by_key: dict[tuple[int, int], PlayerGameStats] = {
        (r.player_id, r.game_id): r for r in pgs_rows
    }

    svc = PredictionService(
        db, registry,
        edge_threshold=settings.edge_threshold,
        over_edge_threshold=settings.edge_threshold_over,
    )
    # Distributions cached per (player, game) since they don't depend on book/line.
    dist_cache: dict[tuple[int, int], dict] = {}

    overall = _empty_side()
    side_over = _empty_side()
    side_under = _empty_side()
    by_stat_dict: dict[str, dict] = {}

    for prop in latest.values():
        if prop.stat_type not in registry:
            continue
        game = game_by_id.get(prop.game_id)
        if game is None:
            continue
        # Resolve the game we look up actuals against (the stub's real twin
        # if a stub, else itself). Predictions are computed against the stub
        # since that's what the prop was attached to — same player, same
        # date, so the feature builder produces identical features either way.
        real_game_id = stub_to_real.get(prop.game_id, prop.game_id)
        actual_row = actuals_by_key.get((prop.player_id, real_game_id))
        if actual_row is None:
            continue
        actual = getattr(actual_row, prop.stat_type, None)
        if actual is None:
            continue
        actual_f = float(actual)

        key = (prop.player_id, prop.game_id)
        if key not in dist_cache:
            try:
                _, dists = svc.predict_player_game(
                    prop.player_id, prop.game_id, game.game_date,
                    stat_types=registry.all_stats(),
                )
            except Exception:
                continue
            dist_cache[key] = dists
        dist = dist_cache[key].get(prop.stat_type)
        if dist is None:
            continue

        edge = svc.analyze_distribution(
            player_id=prop.player_id,
            game_id=prop.game_id,
            stat_type=prop.stat_type,
            distribution=dist,
            line=prop.line,
            over_odds=prop.over_odds,
            under_odds=prop.under_odds,
            book=prop.book,
        )
        rec = edge.recommendation
        if rec == "PASS":
            continue

        stat_bucket = by_stat_dict.setdefault(prop.stat_type, _empty_side())
        side_bucket = side_over if rec == "OVER" else side_under

        for bucket in (overall, stat_bucket, side_bucket):
            bucket["n"] += 1

        if actual_f == prop.line:
            for bucket in (overall, stat_bucket, side_bucket):
                bucket["pushes"] += 1
            continue

        won = (rec == "OVER" and actual_f > prop.line) or (
            rec == "UNDER" and actual_f < prop.line
        )
        result_key = "wins" if won else "losses"
        for bucket in (overall, stat_bucket, side_bucket):
            bucket[result_key] += 1

    overall_settled = overall["n"] - overall["pushes"]
    return RecRecordResponse(
        start=start,
        end=end,
        n_recommendations=overall["n"],
        wins=overall["wins"],
        losses=overall["losses"],
        pushes=overall["pushes"],
        win_rate=(overall["wins"] / overall_settled) if overall_settled > 0 else None,
        by_stat={
            stat: RecRecordByStat(
                n=d["n"],
                wins=d["wins"],
                losses=d["losses"],
                pushes=d["pushes"],
                win_rate=(d["wins"] / (d["n"] - d["pushes"])) if d["n"] - d["pushes"] > 0 else None,
            )
            for stat, d in by_stat_dict.items()
        },
        over=_finalize_side(side_over),
        under=_finalize_side(side_under),
        note="Recs computed with the currently-loaded model, not the production model that may have been live when each prop was offered.",
    )
