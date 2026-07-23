from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from roadwatch_ai.evidence import save_evidence
from roadwatch_ai.models import BoundingBox, VehicleObservation


def observation(y1=20):
    return VehicleObservation(
        track_id=42,
        class_id=2,
        class_name="car",
        confidence=0.9,
        box=BoundingBox(10, y1, 150, 90),
    )


@pytest.mark.parametrize("y1", [20, 60])
def test_saves_annotated_evidence(tmp_path, y1):
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    occurred_at = datetime(2026, 7, 23, 10, 11, 12, 3456, tzinfo=timezone.utc)
    path = save_evidence(
        frame,
        observation(y1),
        plate="JK01AB1234",
        speed_kph=72.4,
        speed_limit_kph=50,
        occurred_at=occurred_at,
        directory=tmp_path / "evidence",
    )
    assert path.name == "20260723T101112_003456_track-42_JK01AB1234.jpg"
    assert path.is_file()
    saved = cv2.imread(str(path))
    assert saved is not None
    assert np.any(saved != 0)
    assert not np.any(frame != 0)


def test_raises_when_image_write_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="Failed to write evidence image"):
        save_evidence(
            np.zeros((100, 200, 3), dtype=np.uint8),
            observation(),
            plate="UNKNOWN",
            speed_kph=80,
            speed_limit_kph=50,
            occurred_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
            directory=tmp_path,
        )
