import runpy
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from roadwatch_ai import cli
from roadwatch_ai.config import AppConfig, DetectionConfig, RuntimeConfig, SpeedConfig
from roadwatch_ai.plates import PlateValidationReport


def test_cli_source_parser():
    assert cli._source("4") == 4
    assert cli._source("sample.mp4") == "sample.mp4"


def test_parser_requires_command():
    with pytest.raises(SystemExit) as error:
        cli._parser().parse_args([])
    assert error.value.code == 2


def test_doctor_reports_missing_dependencies(tmp_path, capsys):
    with (
        patch.object(cli, "load_config", return_value=AppConfig()),
        patch.object(cli.importlib.util, "find_spec", return_value=None),
    ):
        assert cli._doctor(str(tmp_path / "config.yaml")) == 1
    captured = capsys.readouterr()
    assert "Missing packages: OpenCV, EasyOCR, Ultralytics" in captured.err
    assert "Install with: python -m pip install -e ." in captured.err


def test_doctor_requires_plate_model_when_configured(tmp_path, capsys):
    config = replace(
        AppConfig(),
        detection=DetectionConfig(
            plate_model=str(tmp_path / "missing.pt"), require_plate_model=True
        ),
    )
    with (
        patch.object(cli, "load_config", return_value=config),
        patch.object(cli.importlib.util, "find_spec", return_value=object()),
    ):
        assert cli._doctor("config.yaml") == 1
    assert "Plate model not found" in capsys.readouterr().err


def test_doctor_warns_for_optional_missing_plate_model(tmp_path, capsys):
    config = replace(
        AppConfig(),
        detection=DetectionConfig(plate_model=str(tmp_path / "missing.pt")),
    )
    with (
        patch.object(cli, "load_config", return_value=config),
        patch.object(cli.importlib.util, "find_spec", return_value=object()),
    ):
        assert cli._doctor("config.yaml") == 0
    captured = capsys.readouterr()
    assert "WARNING: Plate model not found" in captured.out
    assert "Configuration and Python dependencies are valid." in captured.out


def test_doctor_reports_valid_operational_values(tmp_path, capsys):
    model = tmp_path / "plate.pt"
    model.write_bytes(b"model")
    config = replace(
        AppConfig(),
        detection=DetectionConfig(plate_model=str(model)),
        speed=SpeedConfig(speed_limit_kph=60, distance_meters=15),
    )
    with (
        patch.object(cli, "load_config", return_value=config),
        patch.object(cli.importlib.util, "find_spec", return_value=object()),
    ):
        assert cli._doctor("config.yaml") == 0
    output = capsys.readouterr().out
    assert "Camera source: 0" in output
    assert "Speed limit: 60.0 km/h" in output
    assert "Measured line distance: 15.00 m" in output
    assert "SMS enabled: False" in output


def test_main_runs_doctor():
    config = AppConfig()
    with (
        patch.object(cli, "load_config", return_value=config) as load,
        patch.object(cli, "_doctor", return_value=7) as doctor,
    ):
        assert cli.main(["doctor", "--config", "custom.yaml"]) == 7
    load.assert_called_once_with("custom.yaml")
    doctor.assert_called_once_with("custom.yaml")


def test_main_runs_plate_validation():
    with (
        patch.object(cli, "load_config", return_value=AppConfig()),
        patch.object(cli, "_validate_plates", return_value=6) as validate,
    ):
        assert (
            cli.main(
                [
                    "validate-plates",
                    "--config",
                    "custom.yaml",
                    "--directory",
                    "samples",
                    "--minimum-accuracy",
                    "0.95",
                ]
            )
            == 6
        )
    validate.assert_called_once_with("custom.yaml", "samples", 0.95)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            ["run", "--config", "c.yaml", "--dry-run"],
            {
                "dry_run": True,
                "display_override": None,
                "source_override": None,
            },
        ),
        (
            ["run", "--source", "2", "--display"],
            {
                "dry_run": False,
                "display_override": True,
                "source_override": 2,
            },
        ),
        (
            ["run", "--source", "video.mp4"],
            {
                "dry_run": False,
                "display_override": None,
                "source_override": "video.mp4",
            },
        ),
    ],
)
def test_main_runs_monitor(arguments, expected):
    run = Mock(return_value=9)
    with (
        patch.object(
            cli,
            "load_config",
            return_value=replace(AppConfig(), runtime=RuntimeConfig(log_level="DEBUG")),
        ),
        patch("roadwatch_ai.app.run", run),
        patch.object(cli.logging, "basicConfig") as logging_config,
    ):
        assert cli.main(arguments) == 9
    run.assert_called_once()
    assert run.call_args.kwargs == expected
    logging_config.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        RuntimeError("dependency"),
        ValueError("invalid"),
    ],
)
def test_main_returns_user_error_for_expected_failures(error, capsys):
    with patch.object(cli, "load_config", side_effect=error):
        assert cli.main(["doctor"]) == 2
    assert f"RoadWatch error: {error}" in capsys.readouterr().err


def test_package_main_module_exits_with_cli_status():
    with patch.object(cli, "main", return_value=5):
        with pytest.raises(SystemExit) as error:
            runpy.run_module("roadwatch_ai.__main__", run_name="__main__")
    assert error.value.code == 5


def test_package_main_module_does_nothing_when_imported_under_other_name():
    namespace = runpy.run_module("roadwatch_ai.__main__", run_name="roadwatch_ai.__main_test__")
    assert "main" in namespace


def test_validate_plates_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        cli._validate_plates("config.yaml", "samples", 1.1)


@pytest.mark.parametrize(
    ("report", "threshold", "expected_status"),
    [
        (PlateValidationReport(10, 9, 0, ()), 0.9, 0),
        (
            PlateValidationReport(
                10,
                8,
                1,
                (("JK01AB1234__bad.jpg", "JK01AB1234", "UNKNOWN"),),
            ),
            0.9,
            1,
        ),
    ],
)
def test_validate_plates_reports_accuracy(report, threshold, expected_status, capsys):
    reader = object()
    with (
        patch.object(cli, "load_config", return_value=AppConfig()),
        patch("roadwatch_ai.plates.PlateReader", return_value=reader),
        patch("roadwatch_ai.plates.validate_plate_directory", return_value=report) as validate,
    ):
        assert cli._validate_plates("config.yaml", "samples", threshold) == expected_status
    validate.assert_called_once_with(reader, "samples")
    output = capsys.readouterr().out
    assert f"{report.correct}/{report.total}" in output
    if report.mistakes:
        assert "MISMATCH JK01AB1234__bad.jpg" in output
