from __future__ import annotations

from datetime import datetime
from pathlib import Path

from roadwatch_ai.models import VehicleObservation


def save_evidence(
    frame: object,
    observation: VehicleObservation,
    *,
    plate: str,
    speed_kph: float,
    speed_limit_kph: float,
    occurred_at: datetime,
    directory: Path,
) -> Path:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    box = observation.box
    cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 3)
    caption = (
        f"{plate} | {speed_kph:.1f} km/h | limit {speed_limit_kph:.1f} | "
        f"track {observation.track_id}"
    )
    y = max(35, box.y1 - 12)
    cv2.putText(
        output,
        caption,
        (max(5, box.x1), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    filename = (
        f"{occurred_at.strftime('%Y%m%dT%H%M%S_%f')}_"
        f"track-{observation.track_id}_{plate}.jpg"
    )
    path = directory / filename
    if not cv2.imwrite(str(path), output):
        raise OSError(f"Failed to write evidence image: {path}")
    return path
