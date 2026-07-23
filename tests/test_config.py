import builtins
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from roadwatch_ai.config import (
    AlertsConfig,
    AppConfig,
    CameraConfig,
    DetectionConfig,
    SpeedConfig,
    _line,
    _positive,
    _source,
    load_config,
    validate_config,
)


def test_source_parses_indexes_and_urls():
    assert _source(2) == 2
    assert _source(" 12 ") == 12
    assert _source("rtsp://camera/live") == "rtsp://camera/live"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("bad", "must contain exactly two points"),
        ([[0, 0]], "must contain exactly two points"),
        ([[0, 0], "bad"], "points must contain x and y"),
        ([[0, 0], [1]], "points must contain x and y"),
        ([[-0.1, 0], [1, 1]], "coordinates must be normalized"),
        ([[0, 0], [1.1, 1]], "coordinates must be normalized"),
        ([[0.5, 0.5], [0.5, 0.5]], "endpoints must be different"),
    ],
)
def test_line_validation_rejects_invalid_geometry(value, message):
    with pytest.raises(ValueError, match=message):
        _line(value, "test.line")


def test_line_converts_numeric_values():
    assert _line([["0.1", 0.2], [0.9, "0.8"]], "line") == (
        (0.1, 0.2),
        (0.9, 0.8),
    )


def test_positive_returns_value_and_rejects_non_positive():
    assert _positive(1.5, "distance") == 1.5
    with pytest.raises(ValueError, match="distance must be greater than zero"):
        _positive(0, "distance")


def test_load_config_uses_defaults_for_empty_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    config = load_config(path)
    assert config == AppConfig()


def test_load_config_loads_dotenv_when_available(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}", encoding="utf-8")
    dotenv = ModuleType("dotenv")
    calls = []
    dotenv.load_dotenv = lambda: calls.append(True)
    with patch.dict(sys.modules, {"dotenv": dotenv}):
        assert load_config(path) == AppConfig()
    assert calls == [True]


def test_load_config_reads_all_sections_and_environment(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
camera:
  source: "3"
  width: 1280
  height: 720
  process_every_n_frames: 2
detection:
  vehicle_model: vehicle.pt
  plate_model: plate.pt
  require_plate_model: true
  vehicle_confidence: 0.6
  plate_confidence: 0.7
  vehicle_classes: [2, 7]
  ocr_languages: [en, hi]
  ocr_min_confidence: 0.4
speed:
  start_line: [[0.1, 0.2], [0.9, 0.2]]
  end_line: [[0.1, 0.8], [0.9, 0.8]]
  distance_meters: 15
  speed_limit_kph: 60
  correction_factor: 1.05
  stale_track_seconds: 12
  minimum_travel_seconds: 0.2
  maximum_travel_seconds: 8
alerts:
  enabled: true
  cooldown_seconds: 90
storage:
  database_path: custom/events.db
  evidence_directory: custom/evidence
runtime:
  display: true
  log_level: debug
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+1")
    monkeypatch.setenv("ALERT_TO_NUMBER", "+91")

    config = load_config(path)

    assert config.camera == CameraConfig(3, 1280, 720, 2)
    assert config.detection == DetectionConfig(
        vehicle_model="vehicle.pt",
        plate_model="plate.pt",
        require_plate_model=True,
        vehicle_confidence=0.6,
        plate_confidence=0.7,
        vehicle_classes=(2, 7),
        ocr_languages=("en", "hi"),
        ocr_min_confidence=0.4,
    )
    assert config.speed.distance_meters == 15
    assert config.speed.start_line[0] == (0.1, 0.2)
    assert config.alerts == AlertsConfig(
        enabled=True,
        cooldown_seconds=90,
        account_sid="AC",
        auth_token="token",
        from_number="+1",
        to_number="+91",
    )
    assert config.storage.database_path == Path("custom/events.db")
    assert config.runtime.display is True
    assert config.runtime.log_level == "DEBUG"


def test_load_config_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Copy config.example.yaml"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_reports_missing_yaml_dependency(tmp_path):
    real_import = builtins.__import__

    def fail_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_yaml):
        with pytest.raises(RuntimeError, match="PyYAML is required"):
            load_config(tmp_path / "config.yaml")


def test_load_config_allows_missing_dotenv(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{}", encoding="utf-8")
    real_import = builtins.__import__

    def fail_dotenv(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_dotenv):
        assert load_config(path) == AppConfig()


@pytest.mark.parametrize(
    "camera",
    [CameraConfig(width=0), CameraConfig(width=1, height=0)],
)
def test_validate_config_rejects_invalid_dimensions(camera):
    with pytest.raises(ValueError, match="camera width and height"):
        validate_config(replace(AppConfig(), camera=camera))


def test_validate_config_rejects_frame_interval():
    with pytest.raises(ValueError, match="at least 1"):
        validate_config(replace(AppConfig(), camera=CameraConfig(process_every_n_frames=0)))


@pytest.mark.parametrize(
    "detection",
    [
        DetectionConfig(vehicle_confidence=-0.1),
        DetectionConfig(plate_confidence=1.1),
        DetectionConfig(ocr_min_confidence=-0.1),
    ],
)
def test_validate_config_rejects_confidence_outside_unit_interval(detection):
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        validate_config(replace(AppConfig(), detection=detection))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_meters", 0),
        ("speed_limit_kph", 0),
        ("correction_factor", 0),
        ("stale_track_seconds", 0),
        ("minimum_travel_seconds", 0),
        ("maximum_travel_seconds", 0),
    ],
)
def test_validate_config_rejects_non_positive_speed_values(field, value):
    speed = replace(SpeedConfig(), **{field: value})
    with pytest.raises(ValueError, match=f"speed.{field} must be greater than zero"):
        validate_config(replace(AppConfig(), speed=speed))


def test_validate_config_rejects_reversed_time_bounds():
    with pytest.raises(ValueError, match="minimum_travel_seconds must be lower"):
        validate_config(
            replace(
                AppConfig(),
                speed=SpeedConfig(
                    minimum_travel_seconds=1,
                    maximum_travel_seconds=1,
                ),
            )
        )


def test_validate_config_lists_missing_alert_secrets():
    with pytest.raises(ValueError, match="TWILIO_AUTH_TOKEN") as error:
        validate_config(
            replace(
                AppConfig(),
                alerts=AlertsConfig(
                    enabled=True,
                    account_sid="AC",
                    from_number="+1",
                    to_number="+91",
                ),
            )
        )
    assert "TWILIO_ACCOUNT_SID" not in str(error.value)


def test_validate_config_accepts_enabled_alerts_with_secrets():
    validate_config(
        replace(
            AppConfig(),
            alerts=AlertsConfig(
                enabled=True,
                account_sid="AC",
                auth_token="token",
                from_number="+1",
                to_number="+91",
            ),
        )
    )
