from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from roadwatch_ai.alerting import AlertGate, SmsNotifier
from roadwatch_ai.config import AppConfig
from roadwatch_ai.detection import VehicleTracker
from roadwatch_ai.evidence import save_evidence
from roadwatch_ai.models import VehicleObservation, ViolationEvent
from roadwatch_ai.plates import PlateReader
from roadwatch_ai.speed import SpeedEstimator, denormalize_line
from roadwatch_ai.storage import EventRepository

LOGGER = logging.getLogger(__name__)


def _video_timestamp(capture: object, source: int | str) -> float:
    import cv2

    if isinstance(source, str) and Path(source).is_file():
        milliseconds = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if milliseconds > 0:
            return milliseconds / 1000.0
    return time.monotonic()


def _draw_live_overlay(
    frame: object,
    observations: list[VehicleObservation],
    start_line: tuple[tuple[float, float], tuple[float, float]],
    end_line: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    import cv2

    cv2.line(
        frame,
        tuple(int(value) for value in start_line[0]),
        tuple(int(value) for value in start_line[1]),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        tuple(int(value) for value in end_line[0]),
        tuple(int(value) for value in end_line[1]),
        (255, 255, 0),
        3,
    )
    for observation in observations:
        box = observation.box
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{observation.class_name} #{observation.track_id}",
            (box.x1, max(20, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def run(
    config: AppConfig,
    *,
    dry_run: bool = False,
    display_override: bool | None = None,
    source_override: int | str | None = None,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is missing. Install the project with: pip install -e .") from exc

    source = config.camera.source if source_override is None else source_override
    display = config.runtime.display if display_override is None else display_override
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {source}")

    tracker = VehicleTracker(config.detection)
    plate_reader = PlateReader(config.detection)
    repository = EventRepository(config.storage.database_path)
    notifier = SmsNotifier(config.alerts, dry_run=dry_run)
    alert_gate = AlertGate(config.alerts.cooldown_seconds)

    estimator: SpeedEstimator | None = None
    start_line = None
    end_line = None
    frame_index = 0
    latest_observations: list[VehicleObservation] = []

    LOGGER.info("RoadWatch started; source=%s dry_run=%s display=%s", source, dry_run, display)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                LOGGER.info("Video source ended or frame capture failed")
                break
            frame_index += 1
            if frame_index % config.camera.process_every_n_frames != 0:
                if display:
                    cv2.imshow("RoadWatch AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            if estimator is None:
                height, width = frame.shape[:2]
                start_line = denormalize_line(config.speed.start_line, width, height)
                end_line = denormalize_line(config.speed.end_line, width, height)
                estimator = SpeedEstimator(
                    start_line,
                    end_line,
                    config.speed.distance_meters,
                    correction_factor=config.speed.correction_factor,
                    stale_track_seconds=config.speed.stale_track_seconds,
                    minimum_travel_seconds=config.speed.minimum_travel_seconds,
                    maximum_travel_seconds=config.speed.maximum_travel_seconds,
                )

            timestamp = _video_timestamp(capture, source)
            latest_observations = list(tracker.track(frame))
            for observation in latest_observations:
                measurement = estimator.observe(
                    observation.track_id, observation.box.bottom_center, timestamp
                )
                if measurement is None:
                    continue
                LOGGER.info(
                    "Track %s measured at %.1f km/h in %.3f seconds",
                    observation.track_id,
                    measurement.speed_kph,
                    measurement.elapsed_seconds,
                )
                if measurement.speed_kph <= config.speed.speed_limit_kph:
                    continue

                plate = plate_reader.read(frame, observation.box)
                occurred_at = datetime.now(timezone.utc)
                evidence_path = save_evidence(
                    frame,
                    observation,
                    plate=plate.text,
                    speed_kph=measurement.speed_kph,
                    speed_limit_kph=config.speed.speed_limit_kph,
                    occurred_at=occurred_at,
                    directory=config.storage.evidence_directory,
                )
                alert_key = (
                    plate.text
                    if plate.text != "UNKNOWN"
                    else f"unknown-track-{observation.track_id}"
                )
                alert_sent = False
                if alert_gate.allow(alert_key, timestamp):
                    alert_sent = notifier.send(
                        plate=plate.text,
                        speed_kph=measurement.speed_kph,
                        limit_kph=config.speed.speed_limit_kph,
                        occurred_at=occurred_at,
                        evidence_path=str(evidence_path),
                    )

                event = ViolationEvent(
                    occurred_at=occurred_at,
                    track_id=observation.track_id,
                    vehicle_class=observation.class_name,
                    plate=plate.text,
                    plate_confidence=plate.confidence,
                    speed_kph=measurement.speed_kph,
                    speed_limit_kph=config.speed.speed_limit_kph,
                    evidence_path=evidence_path,
                    alert_sent=alert_sent,
                )
                event_id = repository.add(event)
                LOGGER.warning(
                    "Overspeed event %s: plate=%s speed=%.1f km/h evidence=%s",
                    event_id,
                    plate.text,
                    measurement.speed_kph,
                    evidence_path,
                )

            if display:
                assert start_line is not None and end_line is not None
                _draw_live_overlay(frame, latest_observations, start_line, end_line)
                cv2.imshow("RoadWatch AI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()
    return 0
