from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Point = tuple[float, float]
Line = tuple[Point, Point]


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    width: int = 1920
    height: int = 1080
    process_every_n_frames: int = 1


@dataclass(frozen=True)
class DetectionConfig:
    vehicle_model: str = "yolo11n.pt"
    plate_model: str = "models/license_plate_detector.pt"
    require_plate_model: bool = False
    vehicle_confidence: float = 0.45
    plate_confidence: float = 0.40
    vehicle_classes: tuple[int, ...] = (2, 3, 5, 7)
    ocr_languages: tuple[str, ...] = ("en",)
    ocr_min_confidence: float = 0.30


@dataclass(frozen=True)
class SpeedConfig:
    start_line: Line = ((0.08, 0.58), (0.92, 0.58))
    end_line: Line = ((0.05, 0.82), (0.95, 0.82))
    distance_meters: float = 12.0
    speed_limit_kph: float = 50.0
    correction_factor: float = 1.0
    stale_track_seconds: float = 10.0
    minimum_travel_seconds: float = 0.10
    maximum_travel_seconds: float = 10.0


@dataclass(frozen=True)
class AlertsConfig:
    enabled: bool = False
    cooldown_seconds: float = 300.0
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    to_number: str = ""


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path = Path("data/roadwatch.db")
    evidence_directory: Path = Path("data/evidence")


@dataclass(frozen=True)
class RuntimeConfig:
    display: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _line(value: Any, name: str) -> Line:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two points")
    points: list[Point] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{name} points must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"{name} coordinates must be normalized between 0 and 1")
        points.append((x, y))
    if points[0] == points[1]:
        raise ValueError(f"{name} endpoints must be different")
    return (points[0], points[1])


def _source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def _positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install the project with: pip install -e ."
        ) from exc

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}. Copy config.example.yaml to config.yaml."
        )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    camera = raw.get("camera", {})
    detection = raw.get("detection", {})
    speed = raw.get("speed", {})
    alerts = raw.get("alerts", {})
    storage = raw.get("storage", {})
    runtime = raw.get("runtime", {})

    app_config = AppConfig(
        camera=CameraConfig(
            source=_source(camera.get("source", 0)),
            width=int(camera.get("width", 1920)),
            height=int(camera.get("height", 1080)),
            process_every_n_frames=int(camera.get("process_every_n_frames", 1)),
        ),
        detection=DetectionConfig(
            vehicle_model=str(detection.get("vehicle_model", "yolo11n.pt")),
            plate_model=str(detection.get("plate_model", "models/license_plate_detector.pt")),
            require_plate_model=bool(detection.get("require_plate_model", False)),
            vehicle_confidence=float(detection.get("vehicle_confidence", 0.45)),
            plate_confidence=float(detection.get("plate_confidence", 0.40)),
            vehicle_classes=tuple(
                int(value) for value in detection.get("vehicle_classes", [2, 3, 5, 7])
            ),
            ocr_languages=tuple(detection.get("ocr_languages", ["en"])),
            ocr_min_confidence=float(detection.get("ocr_min_confidence", 0.30)),
        ),
        speed=SpeedConfig(
            start_line=_line(
                speed.get("start_line", [[0.08, 0.58], [0.92, 0.58]]), "speed.start_line"
            ),
            end_line=_line(speed.get("end_line", [[0.05, 0.82], [0.95, 0.82]]), "speed.end_line"),
            distance_meters=float(speed.get("distance_meters", 12.0)),
            speed_limit_kph=float(speed.get("speed_limit_kph", 50.0)),
            correction_factor=float(speed.get("correction_factor", 1.0)),
            stale_track_seconds=float(speed.get("stale_track_seconds", 10.0)),
            minimum_travel_seconds=float(speed.get("minimum_travel_seconds", 0.10)),
            maximum_travel_seconds=float(speed.get("maximum_travel_seconds", 10.0)),
        ),
        alerts=AlertsConfig(
            enabled=bool(alerts.get("enabled", False)),
            cooldown_seconds=float(alerts.get("cooldown_seconds", 300.0)),
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            from_number=os.getenv("TWILIO_FROM_NUMBER", ""),
            to_number=os.getenv("ALERT_TO_NUMBER", ""),
        ),
        storage=StorageConfig(
            database_path=Path(storage.get("database_path", "data/roadwatch.db")),
            evidence_directory=Path(storage.get("evidence_directory", "data/evidence")),
        ),
        runtime=RuntimeConfig(
            display=bool(runtime.get("display", False)),
            log_level=str(runtime.get("log_level", "INFO")).upper(),
        ),
    )
    validate_config(app_config)
    return app_config


def validate_config(config: AppConfig) -> None:
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ValueError("camera width and height must be greater than zero")
    if config.camera.process_every_n_frames < 1:
        raise ValueError("camera.process_every_n_frames must be at least 1")
    for name, value in (
        ("detection.vehicle_confidence", config.detection.vehicle_confidence),
        ("detection.plate_confidence", config.detection.plate_confidence),
        ("detection.ocr_min_confidence", config.detection.ocr_min_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    _positive(config.speed.distance_meters, "speed.distance_meters")
    _positive(config.speed.speed_limit_kph, "speed.speed_limit_kph")
    _positive(config.speed.correction_factor, "speed.correction_factor")
    _positive(config.speed.stale_track_seconds, "speed.stale_track_seconds")
    _positive(config.speed.minimum_travel_seconds, "speed.minimum_travel_seconds")
    _positive(config.speed.maximum_travel_seconds, "speed.maximum_travel_seconds")
    if config.speed.minimum_travel_seconds >= config.speed.maximum_travel_seconds:
        raise ValueError("minimum_travel_seconds must be lower than maximum_travel_seconds")
    if config.alerts.enabled:
        missing = [
            name
            for name, value in (
                ("TWILIO_ACCOUNT_SID", config.alerts.account_sid),
                ("TWILIO_AUTH_TOKEN", config.alerts.auth_token),
                ("TWILIO_FROM_NUMBER", config.alerts.from_number),
                ("ALERT_TO_NUMBER", config.alerts.to_number),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"alerts are enabled but these environment values are missing: {missing}"
            )
