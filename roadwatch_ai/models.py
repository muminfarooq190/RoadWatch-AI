from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, float(self.y2))


@dataclass(frozen=True)
class VehicleObservation:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class SpeedMeasurement:
    track_id: int
    speed_kph: float
    elapsed_seconds: float
    measured_at_monotonic: float


@dataclass(frozen=True)
class PlateReading:
    text: str
    confidence: float
    crop_box: BoundingBox | None = None


@dataclass(frozen=True)
class ViolationEvent:
    occurred_at: datetime
    track_id: int
    vehicle_class: str
    plate: str
    plate_confidence: float
    speed_kph: float
    speed_limit_kph: float
    evidence_path: Path
    alert_sent: bool
