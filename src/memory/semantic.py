"""Semantic memory — SQLite-backed facts and patterns.

Facts are promoted from episodic memory during the sleep/consolidation
cycle. Each time a pattern is observed again its confidence increases.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.memory.models import SemanticEntry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SemanticStore:
    def __init__(self, path: Path | str = "data/memory/semantic.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic (
                id           TEXT PRIMARY KEY,
                concept      TEXT NOT NULL,
                category     TEXT NOT NULL,
                evidence_count INTEGER DEFAULT 1,
                confidence   REAL    DEFAULT 0.5,
                created_at   TEXT    NOT NULL,
                updated_at   TEXT    NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON semantic(category)"
        )
        self._conn.commit()

    # ── public API ────────────────────────────────────────────────────────────

    def add_or_update(
        self,
        concept: str,
        category: str,
        confidence_delta: float = 0.1,
    ) -> SemanticEntry:
        """Add a new fact or strengthen an existing one."""
        now = _now()
        existing = self._find_exact(concept, category)
        if existing:
            new_conf = min(1.0, existing.confidence + confidence_delta)
            new_count = existing.evidence_count + 1
            self._conn.execute(
                "UPDATE semantic SET evidence_count=?, confidence=?, updated_at=? WHERE id=?",
                (new_count, new_conf, now, existing.id),
            )
            self._conn.commit()
            existing.evidence_count = new_count
            existing.confidence = new_conf
            existing.updated_at = now
            return existing

        entry = SemanticEntry(
            id=str(uuid.uuid4()),
            concept=concept,
            category=category,
            confidence=0.5,
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            "INSERT INTO semantic VALUES (?,?,?,?,?,?,?)",
            (
                entry.id,
                entry.concept,
                entry.category,
                entry.evidence_count,
                entry.confidence,
                entry.created_at,
                entry.updated_at,
            ),
        )
        self._conn.commit()
        return entry

    def retrieve(
        self,
        category: str | None = None,
        min_confidence: float = 0.3,
        limit: int = 10,
    ) -> list[SemanticEntry]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM semantic WHERE category=? AND confidence>=?"
                " ORDER BY confidence DESC LIMIT ?",
                (category, min_confidence, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM semantic WHERE confidence>=?"
                " ORDER BY confidence DESC LIMIT ?",
                (min_confidence, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def get_all(self) -> list[SemanticEntry]:
        rows = self._conn.execute(
            "SELECT * FROM semantic ORDER BY confidence DESC"
        ).fetchall()
        return [self._row(r) for r in rows]

    # ── internal ──────────────────────────────────────────────────────────────

    def _find_exact(self, concept: str, category: str) -> SemanticEntry | None:
        row = self._conn.execute(
            "SELECT * FROM semantic WHERE concept=? AND category=?",
            (concept, category),
        ).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(r: tuple) -> SemanticEntry:  # type: ignore[type-arg]
        return SemanticEntry(
            id=r[0],
            concept=r[1],
            category=r[2],
            evidence_count=r[3],
            confidence=r[4],
            created_at=r[5],
            updated_at=r[6],
        )

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM semantic").fetchone()[0])
