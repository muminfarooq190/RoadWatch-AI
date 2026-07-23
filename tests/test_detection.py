import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from roadwatch_ai.config import DetectionConfig
from roadwatch_ai.detection import VehicleTracker
from roadwatch_ai.models import BoundingBox, VehicleObservation


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results = []
        self.calls = []

    def track(self, frame, **options):
        self.calls.append((frame, options))
        return self.results


def install_ultralytics():
    module = ModuleType("ultralytics")
    instances = []

    def yolo(model_name):
        model = FakeModel(model_name)
        instances.append(model)
        return model

    module.YOLO = yolo
    return patch.dict(sys.modules, {"ultralytics": module}), instances


def test_tracker_reports_missing_ultralytics():
    real_import = builtins.__import__

    def fail_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_ultralytics):
        with pytest.raises(RuntimeError, match="Ultralytics is missing"):
            VehicleTracker(DetectionConfig())


def test_tracker_configures_model_and_returns_empty_results():
    context, instances = install_ultralytics()
    with context:
        tracker = VehicleTracker(
            DetectionConfig(
                vehicle_model="vehicle.pt",
                vehicle_confidence=0.6,
                vehicle_classes=(2, 7),
            )
        )
    model = instances[0]
    assert model.model_name == "vehicle.pt"
    assert tracker.track("frame") == ()
    assert model.calls[0] == (
        "frame",
        {
            "persist": True,
            "tracker": "bytetrack.yaml",
            "conf": 0.6,
            "classes": [2, 7],
            "verbose": False,
        },
    )


@pytest.mark.parametrize(
    "boxes",
    [None, SimpleNamespace(id=None)],
)
def test_tracker_ignores_results_without_track_ids(boxes):
    context, instances = install_ultralytics()
    with context:
        tracker = VehicleTracker(DetectionConfig())
    instances[0].results = [SimpleNamespace(boxes=boxes)]
    assert tracker.track("frame") == ()


def test_tracker_maps_yolo_tensors_to_observations():
    context, instances = install_ultralytics()
    with context:
        tracker = VehicleTracker(DetectionConfig())
    boxes = SimpleNamespace(
        xyxy=FakeTensor([[1.9, 2.1, 101.8, 202.7], [5, 6, 50, 60]]),
        id=FakeTensor([11, 12]),
        cls=FakeTensor([2, 7]),
        conf=FakeTensor([0.91, 0.82]),
    )
    instances[0].results = [SimpleNamespace(boxes=boxes, names={2: "car", 7: "truck"})]

    assert tracker.track("frame") == [
        VehicleObservation(11, 2, "car", 0.91, BoundingBox(1, 2, 101, 202)),
        VehicleObservation(12, 7, "truck", 0.82, BoundingBox(5, 6, 50, 60)),
    ]
