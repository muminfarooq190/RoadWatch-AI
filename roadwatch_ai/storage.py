from __future__ import annotations

import sqlite3
from pathlib import Path

from roadwatch_ai.models import ViolationEvent


class EventRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    vehicle_class TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    plate_confidence REAL NOT NULL,
                    speed_kph REAL NOT NULL,
                    speed_limit_kph REAL NOT NULL,
                    evidence_path TEXT NOT NULL,
                    alert_sent INTEGER NOT NULL
                )
                """
            )

    def add(self, event: ViolationEvent) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO violations (
                    occurred_at, track_id, vehicle_class, plate, plate_confidence,
                    speed_kph, speed_limit_kph, evidence_path, alert_sent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.occurred_at.isoformat(),
                    event.track_id,
                    event.vehicle_class,
                    event.plate,
                    event.plate_confidence,
                    event.speed_kph,
                    event.speed_limit_kph,
                    str(event.evidence_path),
                    int(event.alert_sent),
                ),
            )
            return int(cursor.lastrowid)
