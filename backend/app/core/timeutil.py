"""Time helpers.

The DB stores naive-UTC datetimes (100k+ existing rows serialized without an
offset); keep new writes naive so string ordering and parsing stay uniform.
"""
from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC now — replacement for the deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utcnow_iso_z() -> str:
    """UTC timestamp string like 2026-07-02T15:04:05Z for report metadata."""
    return utcnow().isoformat(timespec="seconds") + "Z"
