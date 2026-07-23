from __future__ import annotations

from collections.abc import Iterable

from roadwatch_ai.config import DetectionConfig
from roadwatch_ai.models import BoundingBox, VehicleObservation


class VehicleTracker:
    def __init__(self, config: DetectionConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install the project with: pip install -e .") from exc

        self._model = YOLO(config.vehicle_model)
        self._confidence = config.vehicle_confidence
        self._classes = list(config.vehicle_classes)

    def track(self, frame: object) -> Iterable[VehicleObservation]:
        results = self._model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self._confidence,
            classes=self._classes,
            verbose=False,
        )
        if not results:
            return ()

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return ()

        names = result.names
        observations: list[VehicleObservation] = []
        for xyxy, track_id, class_id, confidence in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.id.int().cpu().tolist(),
            boxes.cls.int().cpu().tolist(),
            boxes.conf.cpu().tolist(),
            strict=True,
        ):
            observations.append(
                VehicleObservation(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=str(names[int(class_id)]),
                    confidence=float(confidence),
                    box=BoundingBox(*(int(value) for value in xyxy)),
                )
            )
        return observations
