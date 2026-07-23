import builtins
import sys
from dataclasses import replace
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import pytest

from roadwatch_ai import app
from roadwatch_ai.config import AppConfig, CameraConfig, RuntimeConfig, StorageConfig
from roadwatch_ai.models import (
    BoundingBox,
    PlateReading,
    SpeedMeasurement,
    VehicleObservation,
)


class FakeCapture:
    def __init__(self, frames=(), *, opened=True, milliseconds=0):
        self.frames = list(frames)
        self.opened = opened
        self.milliseconds = milliseconds
        self.set_calls = []
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        return False, None

    def get(self, prop):
        return self.milliseconds

    def release(self):
        self.released = True


def fake_cv2(capture, wait_keys=()):
    module = ModuleType("cv2")
    module.CAP_PROP_FRAME_WIDTH = 3
    module.CAP_PROP_FRAME_HEIGHT = 4
    module.CAP_PROP_POS_MSEC = 0
    module.FONT_HERSHEY_SIMPLEX = 0
    module.LINE_AA = 16
    module.capture_source = None
    module.imshow_calls = []
    module.line_calls = []
    module.rectangle_calls = []
    module.put_text_calls = []
    module.destroyed = False
    keys = list(wait_keys)

    def video_capture(source):
        module.capture_source = source
        return capture

    module.VideoCapture = video_capture
    module.imshow = lambda *args: module.imshow_calls.append(args)
    module.waitKey = lambda delay: keys.pop(0) if keys else 0
    module.line = lambda *args: module.line_calls.append(args)
    module.rectangle = lambda *args: module.rectangle_calls.append(args)
    module.putText = lambda *args: module.put_text_calls.append(args)

    def destroy():
        module.destroyed = True

    module.destroyAllWindows = destroy
    return module


def observation(track_id, y1=10):
    return VehicleObservation(
        track_id=track_id,
        class_id=2,
        class_name="car",
        confidence=0.9,
        box=BoundingBox(10, y1, 100, 80),
    )


def test_video_timestamp_prefers_video_position_for_files(tmp_path):
    source = tmp_path / "traffic.mp4"
    source.write_bytes(b"video")
    capture = FakeCapture(milliseconds=1250)
    assert app._video_timestamp(capture, str(source)) == 1.25


@pytest.mark.parametrize("source", [0, "not-a-file.mp4"])
def test_video_timestamp_uses_monotonic_for_live_sources(source):
    with patch.object(app.time, "monotonic", return_value=123.4):
        assert app._video_timestamp(FakeCapture(), source) == 123.4


def test_video_timestamp_uses_monotonic_for_video_at_initial_position(tmp_path):
    source = tmp_path / "traffic.mp4"
    source.write_bytes(b"video")
    with patch.object(app.time, "monotonic", return_value=55.0):
        assert app._video_timestamp(FakeCapture(milliseconds=0), str(source)) == 55.0


def test_draw_live_overlay_draws_lines_and_vehicle():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2 = fake_cv2(FakeCapture())
    with patch.dict(sys.modules, {"cv2": cv2}):
        app._draw_live_overlay(
            frame,
            [observation(7, y1=10)],
            ((1.9, 2.1), (100.8, 2.1)),
            ((2.2, 90.9), (101.7, 90.9)),
        )
    assert len(cv2.line_calls) == 2
    assert cv2.line_calls[0][1:3] == ((1, 2), (100, 2))
    assert len(cv2.rectangle_calls) == 1
    assert cv2.put_text_calls[0][1] == "car #7"
    assert cv2.put_text_calls[0][2] == (10, 20)


def test_run_reports_missing_opencv():
    real_import = builtins.__import__

    def fail_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_cv2):
        with pytest.raises(RuntimeError, match="OpenCV is missing"):
            app.run(AppConfig())


def test_run_rejects_unopened_camera():
    capture = FakeCapture(opened=False)
    cv2 = fake_cv2(capture)
    with patch.dict(sys.modules, {"cv2": cv2}):
        with pytest.raises(RuntimeError, match="Could not open camera"):
            app.run(AppConfig(), source_override="bad-stream")
    assert cv2.capture_source == "bad-stream"
    assert capture.set_calls == [(3, 1920), (4, 1080)]


def test_run_processes_measurements_records_events_and_suppresses_alert(tmp_path):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame), (True, frame)])
    cv2 = fake_cv2(capture)
    observations = [observation(index) for index in range(1, 5)]
    tracker = Mock()
    tracker.track.side_effect = [observations, []]
    estimator = Mock()
    estimator.observe.side_effect = [
        None,
        SpeedMeasurement(2, 40.0, 1.0, 10.0),
        SpeedMeasurement(3, 70.0, 1.0, 10.0),
        SpeedMeasurement(4, 80.0, 0.8, 10.0),
    ]
    plate_reader = Mock()
    plate_reader.read.side_effect = [
        PlateReading("JK01AB1234", 0.95),
        PlateReading("UNKNOWN", 0.0),
    ]
    repository = Mock()
    repository.add.side_effect = [101, 102]
    notifier = Mock()
    notifier.send.return_value = True
    gate = Mock()
    gate.allow.side_effect = [True, False]
    evidence = tmp_path / "evidence.jpg"
    config = replace(
        AppConfig(),
        storage=StorageConfig(tmp_path / "events.db", tmp_path / "evidence"),
    )

    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(app, "VehicleTracker", return_value=tracker),
        patch.object(app, "PlateReader", return_value=plate_reader),
        patch.object(app, "EventRepository", return_value=repository),
        patch.object(app, "SmsNotifier", return_value=notifier),
        patch.object(app, "AlertGate", return_value=gate),
        patch.object(app, "SpeedEstimator", return_value=estimator) as estimator_type,
        patch.object(app, "save_evidence", return_value=evidence) as save,
        patch.object(app.time, "monotonic", return_value=10.0),
    ):
        assert app.run(config, dry_run=True) == 0

    assert capture.released
    assert not cv2.destroyed
    start_line, end_line, distance = estimator_type.call_args.args[:3]
    assert np.asarray(start_line) == pytest.approx(np.asarray(((16.0, 58.0), (184.0, 58.0))))
    assert np.asarray(end_line) == pytest.approx(np.asarray(((10.0, 82.0), (190.0, 82.0))))
    assert distance == 12.0
    assert plate_reader.read.call_count == 2
    assert save.call_count == 2
    assert gate.allow.call_args_list[0].args == ("JK01AB1234", 10.0)
    assert gate.allow.call_args_list[1].args == ("unknown-track-4", 10.0)
    notifier.send.assert_called_once()
    first_event, second_event = [call.args[0] for call in repository.add.call_args_list]
    assert first_event.plate == "JK01AB1234"
    assert first_event.alert_sent is True
    assert second_event.plate == "UNKNOWN"
    assert second_event.alert_sent is False


@pytest.mark.parametrize("processed_key", [0, ord("q")])
def test_run_display_path_draws_overlay_and_releases_window(processed_key):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame)])
    cv2 = fake_cv2(capture, wait_keys=[processed_key])
    tracker = Mock()
    tracker.track.return_value = [observation(1)]
    estimator = Mock()
    estimator.observe.return_value = None
    config = replace(AppConfig(), runtime=RuntimeConfig(display=False))

    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(app, "VehicleTracker", return_value=tracker),
        patch.object(app, "PlateReader", return_value=Mock()),
        patch.object(app, "EventRepository", return_value=Mock()),
        patch.object(app, "SmsNotifier", return_value=Mock()),
        patch.object(app, "AlertGate", return_value=Mock()),
        patch.object(app, "SpeedEstimator", return_value=estimator),
        patch.object(app, "_draw_live_overlay") as overlay,
    ):
        assert app.run(config, display_override=True) == 0

    overlay.assert_called_once()
    assert len(cv2.imshow_calls) == 1
    assert cv2.destroyed
    assert capture.released


@pytest.mark.parametrize("skip_key", [0, ord("q")])
def test_run_skips_configured_frames_and_handles_display_quit(skip_key):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame)])
    cv2 = fake_cv2(capture, wait_keys=[skip_key])
    tracker = Mock()
    config = replace(
        AppConfig(),
        camera=CameraConfig(process_every_n_frames=2),
        runtime=RuntimeConfig(display=True),
    )

    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(app, "VehicleTracker", return_value=tracker),
        patch.object(app, "PlateReader", return_value=Mock()),
        patch.object(app, "EventRepository", return_value=Mock()),
        patch.object(app, "SmsNotifier", return_value=Mock()),
        patch.object(app, "AlertGate", return_value=Mock()),
        patch.object(app, "SpeedEstimator") as estimator_type,
    ):
        assert app.run(config) == 0

    tracker.track.assert_not_called()
    estimator_type.assert_not_called()
    assert len(cv2.imshow_calls) == 1


def test_run_skips_frame_without_display():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    capture = FakeCapture([(True, frame)])
    cv2 = fake_cv2(capture)
    config = replace(
        AppConfig(),
        camera=CameraConfig(process_every_n_frames=2),
        runtime=RuntimeConfig(display=False),
    )
    tracker = Mock()
    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(app, "VehicleTracker", return_value=tracker),
        patch.object(app, "PlateReader", return_value=Mock()),
        patch.object(app, "EventRepository", return_value=Mock()),
        patch.object(app, "SmsNotifier", return_value=Mock()),
        patch.object(app, "AlertGate", return_value=Mock()),
        patch.object(app, "SpeedEstimator") as estimator_type,
    ):
        assert app.run(config) == 0
    tracker.track.assert_not_called()
    estimator_type.assert_not_called()
    assert cv2.imshow_calls == []
