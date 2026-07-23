from __future__ import annotations

from dataclasses import dataclass

from roadwatch_ai.models import SpeedMeasurement

Point = tuple[float, float]
Line = tuple[Point, Point]


def denormalize_line(line: Line, width: int, height: int) -> Line:
    return (
        (line[0][0] * width, line[0][1] * height),
        (line[1][0] * width, line[1][1] * height),
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a: Point, b: Point, c: Point, d: Point, epsilon: float = 1e-9) -> bool:
    """Return whether movement segment a-b crosses finite calibration segment c-d."""
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if abs(o1) <= epsilon and abs(o2) <= epsilon and abs(o3) <= epsilon and abs(o4) <= epsilon:
        x_overlap = (
            max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0])) + epsilon
        )
        y_overlap = (
            max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1])) + epsilon
        )
        return x_overlap and y_overlap

    return (o1 * o2 <= epsilon) and (o3 * o4 <= epsilon)


@dataclass
class _TrackState:
    last_point: Point
    last_seen: float
    start_time: float | None = None
    completed: bool = False


class SpeedEstimator:
    def __init__(
        self,
        start_line: Line,
        end_line: Line,
        distance_meters: float,
        *,
        correction_factor: float = 1.0,
        stale_track_seconds: float = 10.0,
        minimum_travel_seconds: float = 0.10,
        maximum_travel_seconds: float = 10.0,
    ) -> None:
        self.start_line = start_line
        self.end_line = end_line
        self.distance_meters = distance_meters
        self.correction_factor = correction_factor
        self.stale_track_seconds = stale_track_seconds
        self.minimum_travel_seconds = minimum_travel_seconds
        self.maximum_travel_seconds = maximum_travel_seconds
        self._tracks: dict[int, _TrackState] = {}

    def observe(self, track_id: int, point: Point, timestamp: float) -> SpeedMeasurement | None:
        state = self._tracks.get(track_id)
        if state is None:
            self._tracks[track_id] = _TrackState(point, timestamp)
            self._purge_stale(timestamp)
            return None

        measurement: SpeedMeasurement | None = None
        crossed_start = segments_cross(state.last_point, point, *self.start_line)
        crossed_end = segments_cross(state.last_point, point, *self.end_line)

        if not state.completed:
            if state.start_time is None and crossed_start:
                state.start_time = timestamp
            elif state.start_time is not None and crossed_end:
                elapsed = timestamp - state.start_time
                state.completed = True
                if self.minimum_travel_seconds <= elapsed <= self.maximum_travel_seconds:
                    speed = self.distance_meters / elapsed * 3.6 * self.correction_factor
                    measurement = SpeedMeasurement(track_id, speed, elapsed, timestamp)

        state.last_point = point
        state.last_seen = timestamp
        self._purge_stale(timestamp)
        return measurement

    def _purge_stale(self, timestamp: float) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if timestamp - state.last_seen > self.stale_track_seconds
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]
