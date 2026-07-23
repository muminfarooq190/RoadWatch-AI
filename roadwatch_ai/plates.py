from __future__ import annotations

import logging
import re
from pathlib import Path

from roadwatch_ai.config import DetectionConfig
from roadwatch_ai.models import BoundingBox, PlateReading

LOGGER = logging.getLogger(__name__)

STANDARD_INDIAN_PLATE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BHARAT_SERIES_PLATE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def looks_like_indian_plate(text: str) -> bool:
    normalized = normalize_plate_text(text)
    return bool(
        STANDARD_INDIAN_PLATE.fullmatch(normalized)
        or BHARAT_SERIES_PLATE.fullmatch(normalized)
    )


class PlateReader:
    def __init__(self, config: DetectionConfig) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("EasyOCR is missing. Install the project with: pip install -e .") from exc

        self._config = config
        self._reader = easyocr.Reader(list(config.ocr_languages), gpu=False)
        self._plate_model = None
        model_path = Path(config.plate_model)
        if model_path.is_file():
            from ultralytics import YOLO

            self._plate_model = YOLO(str(model_path))
        elif config.require_plate_model:
            raise FileNotFoundError(
                f"Required plate detector not found: {model_path}. "
                "Provide a location-specific YOLO plate model."
            )
        else:
            LOGGER.warning(
                "Plate detector %s is absent; using an unreliable lower-vehicle-crop fallback",
                model_path,
            )

    def read(self, frame: object, vehicle_box: BoundingBox) -> PlateReading:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        vehicle_box = BoundingBox(
            max(0, vehicle_box.x1),
            max(0, vehicle_box.y1),
            min(width, vehicle_box.x2),
            min(height, vehicle_box.y2),
        )
        vehicle_crop = frame[
            vehicle_box.y1 : vehicle_box.y2, vehicle_box.x1 : vehicle_box.x2
        ]
        if vehicle_crop.size == 0:
            return PlateReading("UNKNOWN", 0.0)

        plate_crop, crop_box = self._find_plate(vehicle_crop, vehicle_box)
        if plate_crop.size == 0:
            return PlateReading("UNKNOWN", 0.0)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        results = self._reader.readtext(
            np.asarray(gray),
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        candidates: list[tuple[str, float]] = []
        ordered_parts: list[tuple[float, float, str, float]] = []
        for box, text, confidence in results:
            normalized = normalize_plate_text(text)
            if confidence >= self._config.ocr_min_confidence and 7 <= len(normalized) <= 11:
                candidates.append((normalized, float(confidence)))
            if confidence >= self._config.ocr_min_confidence and normalized:
                centre_x = sum(float(point[0]) for point in box) / len(box)
                centre_y = sum(float(point[1]) for point in box) / len(box)
                ordered_parts.append((centre_y, centre_x, normalized, float(confidence)))

        if len(ordered_parts) > 1:
            ordered_parts.sort(key=lambda item: (round(item[0] / 20), item[1]))
            combined = "".join(item[2] for item in ordered_parts)
            if 7 <= len(combined) <= 11:
                mean_confidence = sum(item[3] for item in ordered_parts) / len(ordered_parts)
                candidates.append((combined, mean_confidence))

        if not candidates:
            return PlateReading("UNKNOWN", 0.0, crop_box)

        candidates.sort(
            key=lambda item: (looks_like_indian_plate(item[0]), item[1]), reverse=True
        )
        text, confidence = candidates[0]
        return PlateReading(text, confidence, crop_box)

    def _find_plate(
        self, vehicle_crop: object, vehicle_box: BoundingBox
    ) -> tuple[object, BoundingBox]:
        if self._plate_model is None:
            height, width = vehicle_crop.shape[:2]
            y1 = int(height * 0.50)
            x1 = int(width * 0.10)
            x2 = int(width * 0.90)
            crop_box = BoundingBox(
                vehicle_box.x1 + x1,
                vehicle_box.y1 + y1,
                vehicle_box.x1 + x2,
                vehicle_box.y2,
            )
            return vehicle_crop[y1:height, x1:x2], crop_box

        results = self._plate_model.predict(
            vehicle_crop, conf=self._config.plate_confidence, verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return vehicle_crop[0:0, 0:0], vehicle_box

        boxes = results[0].boxes
        best_index = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = (
            int(value) for value in boxes.xyxy[best_index].cpu().tolist()
        )
        crop_box = BoundingBox(
            vehicle_box.x1 + x1,
            vehicle_box.y1 + y1,
            vehicle_box.x1 + x2,
            vehicle_box.y1 + y2,
        )
        return vehicle_crop[y1:y2, x1:x2], crop_box
