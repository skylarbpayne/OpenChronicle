"""SQLite store for typed extractor records."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS extractor_records (
  id TEXT PRIMARY KEY,
  extractor_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  confidence REAL NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  source_refs_json TEXT NOT NULL,
  links_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  superseded_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_extractor_records_kind_status
  ON extractor_records(kind, status);

CREATE INDEX IF NOT EXISTS idx_extractor_records_extractor_updated
  ON extractor_records(extractor_id, updated_at);
"""


@dataclass
class ExtractorRecord:
    id: str
    extractor_id: str
    kind: str
    status: str
    confidence: float
    summary: str
    payload: dict[str, Any]
    source_refs: list[dict[str, Any]]
    links: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("extractor record requires id")
        if not self.extractor_id:
            raise ValueError("extractor record requires extractor_id")
        if not self.kind:
            raise ValueError("extractor record requires kind")
        if not self.source_refs:
            raise ValueError("extractor record requires source_refs")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def upsert_record(conn: sqlite3.Connection, record: ExtractorRecord) -> None:
    ensure_schema(conn)
    conn.execute(
        """
        INSERT INTO extractor_records(
            id, extractor_id, kind, status, confidence, summary, payload_json,
            source_refs_json, links_json, created_at, updated_at, superseded_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            extractor_id=excluded.extractor_id,
            kind=excluded.kind,
            status=excluded.status,
            confidence=excluded.confidence,
            summary=excluded.summary,
            payload_json=excluded.payload_json,
            source_refs_json=excluded.source_refs_json,
            links_json=excluded.links_json,
            updated_at=excluded.updated_at,
            superseded_by=excluded.superseded_by
        """,
        _as_tuple(record),
    )


def supersede_record(conn: sqlite3.Connection, old_id: str, new_record: ExtractorRecord) -> None:
    ensure_schema(conn)
    upsert_record(conn, new_record)
    conn.execute(
        """
        UPDATE extractor_records
           SET status='superseded', superseded_by=?, updated_at=?
         WHERE id=?
        """,
        (new_record.id, new_record.updated_at, old_id),
    )


def get_record(conn: sqlite3.Connection, record_id: str) -> ExtractorRecord | None:
    ensure_schema(conn)
    row = conn.execute("SELECT * FROM extractor_records WHERE id=?", (record_id,)).fetchone()
    return _from_row(row) if row else None


def list_records(
    conn: sqlite3.Connection,
    *,
    kind: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
) -> list[ExtractorRecord]:
    ensure_schema(conn)
    clauses: list[str] = []
    args: list[Any] = []
    if kind is not None:
        clauses.append("kind=?")
        args.append(kind)
    if status is not None:
        clauses.append("status=?")
        args.append(status)
    if since is not None:
        clauses.append("updated_at >= ?")
        args.append(since)
    if until is not None:
        clauses.append("updated_at <= ?")
        args.append(until)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM extractor_records {where} ORDER BY updated_at DESC, id ASC LIMIT ?",
        (*args, limit),
    ).fetchall()
    return [_from_row(row) for row in rows]


def search_records(
    conn: sqlite3.Connection,
    *,
    query: str,
    kind: str | None = None,
    status: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
) -> list[ExtractorRecord]:
    """Search typed extractor records with optional entity-type filtering.

    This intentionally stays generic: entity specialization lives in `kind`, not
    in separate tools/functions for every extracted type.
    """
    ensure_schema(conn)
    terms = [t.casefold() for t in query.split() if t.strip()]
    clauses: list[str] = []
    args: list[Any] = []
    if kind is not None:
        clauses.append("kind=?")
        args.append(kind)
    if status is not None:
        clauses.append("status=?")
        args.append(status)
    if since is not None:
        clauses.append("updated_at >= ?")
        args.append(since)
    if until is not None:
        clauses.append("updated_at <= ?")
        args.append(until)
    haystack = "lower(summary || ' ' || payload_json || ' ' || source_refs_json || ' ' || links_json)"
    for term in terms:
        clauses.append(f"{haystack} LIKE ?")
        args.append(f"%{_escape_like(term)}%")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM extractor_records {where} ORDER BY updated_at DESC, id ASC LIMIT ?",
        (*args, limit),
    ).fetchall()
    return [_from_row(row) for row in rows]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _as_tuple(record: ExtractorRecord) -> tuple[Any, ...]:
    return (
        record.id,
        record.extractor_id,
        record.kind,
        record.status,
        float(record.confidence),
        record.summary,
        json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
        json.dumps(record.source_refs, ensure_ascii=False, sort_keys=True),
        json.dumps(record.links, ensure_ascii=False, sort_keys=True),
        record.created_at,
        record.updated_at,
        record.superseded_by,
    )


def _from_row(row: sqlite3.Row) -> ExtractorRecord:
    return ExtractorRecord(
        id=row["id"],
        extractor_id=row["extractor_id"],
        kind=row["kind"],
        status=row["status"],
        confidence=float(row["confidence"]),
        summary=row["summary"],
        payload=json.loads(row["payload_json"] or "{}"),
        source_refs=json.loads(row["source_refs_json"] or "[]"),
        links=json.loads(row["links_json"] or "[]"),
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
        superseded_by=row["superseded_by"],
    )
