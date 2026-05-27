"""
Engine-aware upsert helper.

SQLite supports `ON CONFLICT DO UPDATE` (3.24+); SQLAlchemy 2.x exposes
this via `sqlalchemy.dialects.sqlite.insert(...).on_conflict_do_update`.
For Postgres the equivalent dialect is `dialects.postgresql.insert`.

Usage:
    upsert(db, Player, [{"nba_id": 203999, "full_name": "Nikola Jokic", ...}],
           index_elements=["nba_id"])
"""
from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


def _dialect_insert(engine: Engine):
    name = engine.dialect.name
    if name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _ins
        return _ins
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as _ins
        return _ins
    raise NotImplementedError(f"Upsert not implemented for dialect: {name}")


def upsert(
    db: Session,
    model: type,
    rows: Iterable[dict[str, Any]],
    index_elements: list[str],
    update_columns: list[str] | None = None,
) -> int:
    rows_list = list(rows)
    if not rows_list:
        return 0
    engine = db.get_bind()
    insert_stmt = _dialect_insert(engine)(model.__table__).values(rows_list)
    if update_columns is None:
        update_columns = [
            c.name for c in model.__table__.columns
            if c.name not in index_elements and c.name != "id"
        ]
    set_clause = {col: getattr(insert_stmt.excluded, col) for col in update_columns}
    stmt = insert_stmt.on_conflict_do_update(
        index_elements=index_elements,
        set_=set_clause,
    )
    db.execute(stmt)
    db.commit()
    return len(rows_list)
