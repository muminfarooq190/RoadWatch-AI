import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from roadwatch_ai.models import ViolationEvent
from roadwatch_ai.storage import EventRepository


def test_inserts_and_reads_complete_violation(tmp_path):
    database = tmp_path / "nested" / "events.db"
    repository = EventRepository(database)
    event = ViolationEvent(
        occurred_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
        track_id=12,
        vehicle_class="car",
        plate="JK01AB1234",
        plate_confidence=0.91,
        speed_kph=72.4,
        speed_limit_kph=50.0,
        evidence_path=Path("evidence.jpg"),
        alert_sent=True,
    )

    assert repository.add(event) == 1
    assert EventRepository(database).add(event) == 2

    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT * FROM violations WHERE id = 1").fetchone()
    assert row == (
        1,
        "2026-07-23T00:00:00+00:00",
        12,
        "car",
        "JK01AB1234",
        0.91,
        72.4,
        50.0,
        "evidence.jpg",
        1,
    )
