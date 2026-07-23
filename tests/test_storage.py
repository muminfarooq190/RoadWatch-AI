import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from roadwatch_ai.models import ViolationEvent
from roadwatch_ai.storage import EventRepository


class EventRepositoryTests(unittest.TestCase):
    def test_inserts_violation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = EventRepository(Path(directory) / "events.db")
            event_id = repository.add(
                ViolationEvent(
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
            )
            self.assertEqual(event_id, 1)


if __name__ == "__main__":
    unittest.main()
