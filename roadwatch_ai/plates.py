from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roadwatch_ai.config import DetectionConfig
from roadwatch_ai.models import BoundingBox, PlateReading

LOGGER = logging.getLogger(__name__)

STANDARD_INDIAN_PLATE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BHARAT_SERIES_PLATE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
_LETTER_SUBSTITUTIONS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}
_DIGIT_SUBSTITUTIONS = {
    "B": "8",
    "D": "0",
    "G": "6",
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "S": "5",
    "Z": "2",
}
_UNSET = object()


@dataclass(frozen=True)
class PlateValidationReport:
    total: int
    correct: int
    unknown: int
    mistakes: tuple[tuple[str, str, str], ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def looks_like_indian_plate(text: str) -> bool:
    normalized = normalize_plate_text(text)
    return bool(
        STANDARD_INDIAN_PLATE.fullmatch(normalized) or BHARAT_SERIES_PLATE.fullmatch(normalized)
    )


def _coerce_characters(text: str, *, letters: bool) -> tuple[str, int] | None:
    substitutions = _LETTER_SUBSTITUTIONS if letters else _DIGIT_SUBSTITUTIONS
    expected = str.isalpha if letters else str.isdigit
    output: list[str] = []
    corrections = 0
    for character in text:
        if expected(character):
            output.append(character)
        elif character in substitutions:
            output.append(substitutions[character])
            corrections += 1
        else:
            return None
    return "".join(output), corrections


def _canonical_candidates(text: str) -> list[tuple[str, int]]:
    normalized = normalize_plate_text(text)
    candidates: list[tuple[str, int]] = []

    if 9 <= len(normalized) <= 10:
        year = _coerce_characters(normalized[:2], letters=False)
        marker = _coerce_characters(normalized[2:4], letters=True)
        serial = _coerce_characters(normalized[4:8], letters=False)
        suffix = _coerce_characters(normalized[8:], letters=True)
        if year and marker and serial and suffix and marker[0] == "BH":
            value = year[0] + marker[0] + serial[0] + suffix[0]
            corrections = year[1] + marker[1] + serial[1] + suffix[1]
            candidates.append((value, corrections))

    for district_length in (1, 2):
        for series_length in (1, 2, 3):
            if len(normalized) != 2 + district_length + series_length + 4:
                continue
            district_end = 2 + district_length
            series_end = district_end + series_length
            state = _coerce_characters(normalized[:2], letters=True)
            district = _coerce_characters(normalized[2:district_end], letters=False)
            series = _coerce_characters(normalized[district_end:series_end], letters=True)
            serial = _coerce_characters(normalized[series_end:], letters=False)
            if state and district and series and serial:
                if int(district[0]) == 0:
                    continue
                value = state[0] + district[0] + series[0] + serial[0]
                corrections = state[1] + district[1] + series[1] + serial[1]
                candidates.append((value, corrections))
    return candidates


def canonicalize_indian_plate(text: str) -> str | None:
    """Normalize a plate and repair position-specific OCR confusions."""
    candidates = _canonical_candidates(text)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[0][0]


def _box_center(box: Any) -> tuple[float, float]:
    return (
        sum(float(point[0]) for point in box) / len(box),
        sum(float(point[1]) for point in box) / len(box),
    )


def select_plate_candidate(
    ocr_batches: list[list[tuple[Any, str, float]]],
    min_confidence: float,
) -> tuple[str, float] | None:
    """Choose the strongest valid Indian plate from one or more OCR passes."""
    scored: dict[str, float] = {}
    for results in ocr_batches:
        parts: list[tuple[float, float, str, float]] = []
        raw_candidates: list[tuple[str, float]] = []
        for box, text, confidence in results:
            confidence = float(confidence)
            normalized = normalize_plate_text(text)
            if confidence < min_confidence or not normalized:
                continue
            raw_candidates.append((normalized, confidence))
            centre_x, centre_y = _box_center(box)
            parts.append((centre_y, centre_x, normalized, confidence))

        if len(parts) > 1:
            parts.sort(key=lambda item: (round(item[0] / 20), item[1]))
            raw_candidates.append(
                (
                    "".join(item[2] for item in parts),
                    sum(item[3] for item in parts) / len(parts),
                )
            )

        for raw_text, confidence in raw_candidates:
            candidates = _canonical_candidates(raw_text)
            if not candidates:
                continue
            canonical, corrections = min(candidates, key=lambda item: (item[1], item[0]))
            adjusted_confidence = max(0.0, confidence - corrections * 0.05)
            scored[canonical] = max(scored.get(canonical, 0.0), adjusted_confidence)

    if not scored:
        return None
    return max(scored.items(), key=lambda item: (item[1], item[0]))


class PlateReader:
    def __init__(
        self,
        config: DetectionConfig,
        *,
        reader: object | None = None,
        plate_model: object = _UNSET,
    ) -> None:
        self._config = config
        if reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "EasyOCR is missing. Install the project with: pip install -e ."
                ) from exc
            self._reader = easyocr.Reader(list(config.ocr_languages), gpu=False)
        else:
            self._reader = reader

        if plate_model is not _UNSET:
            self._plate_model = plate_model
        else:
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
                    "Plate detector %s is absent; using OpenCV candidate detection",
                    model_path,
                )

    def read(self, frame: object, vehicle_box: BoundingBox) -> PlateReading:

        height, width = frame.shape[:2]
        vehicle_box = BoundingBox(
            max(0, vehicle_box.x1),
            max(0, vehicle_box.y1),
            min(width, vehicle_box.x2),
            min(height, vehicle_box.y2),
        )
        vehicle_crop = frame[vehicle_box.y1 : vehicle_box.y2, vehicle_box.x1 : vehicle_box.x2]
        if vehicle_crop.size == 0:
            return PlateReading("UNKNOWN", 0.0)

        plate_crop, crop_box = self._find_plate(vehicle_crop, vehicle_box)
        if plate_crop.size == 0:
            return PlateReading("UNKNOWN", 0.0)

        return self.read_crop(plate_crop, crop_box)

    def read_crop(self, plate_crop: object, crop_box: BoundingBox | None = None) -> PlateReading:
        """Read a pre-cropped plate image, useful for validation datasets."""
        import cv2
        import numpy as np

        if plate_crop.size == 0:
            return PlateReading("UNKNOWN", 0.0, crop_box)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        thresholded = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        options = {
            "detail": 1,
            "paragraph": False,
            "allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        }
        ocr_batches = [
            self._reader.readtext(np.asarray(image), **options) for image in (enhanced, thresholded)
        ]
        selected = select_plate_candidate(ocr_batches, self._config.ocr_min_confidence)
        if selected is None:
            return PlateReading("UNKNOWN", 0.0, crop_box)
        text, confidence = selected
        return PlateReading(text, confidence, crop_box)

    def _find_plate(
        self, vehicle_crop: object, vehicle_box: BoundingBox
    ) -> tuple[object, BoundingBox]:
        if self._plate_model is None:
            return self._find_plate_with_opencv(vehicle_crop, vehicle_box)

        results = self._plate_model.predict(
            vehicle_crop, conf=self._config.plate_confidence, verbose=False
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return vehicle_crop[0:0, 0:0], vehicle_box

        boxes = results[0].boxes
        best_index = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = (int(value) for value in boxes.xyxy[best_index].cpu().tolist())
        crop_box = BoundingBox(
            vehicle_box.x1 + x1,
            vehicle_box.y1 + y1,
            vehicle_box.x1 + x2,
            vehicle_box.y1 + y2,
        )
        return vehicle_crop[y1:y2, x1:x2], crop_box

    @staticmethod
    def _find_plate_with_opencv(
        vehicle_crop: object, vehicle_box: BoundingBox
    ) -> tuple[object, BoundingBox]:
        import cv2

        height, width = vehicle_crop.shape[:2]
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 70, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        image_area = max(1, width * height)
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
            if candidate_height == 0:
                continue
            aspect_ratio = candidate_width / candidate_height
            area_ratio = candidate_width * candidate_height / image_area
            if not 2.0 <= aspect_ratio <= 6.5 or not 0.01 <= area_ratio <= 0.35:
                continue
            rectangularity = cv2.contourArea(contour) / max(1, candidate_width * candidate_height)
            vertical_position = (y + candidate_height / 2) / max(1, height)
            score = area_ratio * (0.5 + rectangularity) * (0.5 + vertical_position)
            candidates.append((score, (x, y, candidate_width, candidate_height)))

        if candidates:
            _, (x, y, candidate_width, candidate_height) = max(candidates)
            padding_x = max(2, int(candidate_width * 0.04))
            padding_y = max(2, int(candidate_height * 0.10))
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(width, x + candidate_width + padding_x)
            y2 = min(height, y + candidate_height + padding_y)
        else:
            y1 = int(height * 0.50)
            x1 = int(width * 0.10)
            x2 = int(width * 0.90)
            y2 = height

        crop_box = BoundingBox(
            vehicle_box.x1 + x1,
            vehicle_box.y1 + y1,
            vehicle_box.x1 + x2,
            vehicle_box.y1 + y2,
        )
        return vehicle_crop[y1:y2, x1:x2], crop_box


def validate_plate_directory(reader: PlateReader, directory: str | Path) -> PlateValidationReport:
    """Measure exact-match OCR accuracy on labelled, cropped plate images."""
    import cv2

    directory = Path(directory)
    paths = sorted(
        path for pattern in ("*.jpg", "*.jpeg", "*.png") for path in directory.glob(pattern)
    )
    if not paths:
        raise ValueError(f"No JPG or PNG plate images found in: {directory}")

    correct = 0
    unknown = 0
    mistakes: list[tuple[str, str, str]] = []
    for path in paths:
        label = path.stem.split("__", 1)[0]
        expected = canonicalize_indian_plate(label)
        if expected is None:
            raise ValueError(
                f"Invalid expected plate in filename {path.name}; use PLATENUMBER__sample.jpg"
            )
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read plate image: {path}")
        actual = reader.read_crop(image).text
        if actual == expected:
            correct += 1
        else:
            unknown += int(actual == "UNKNOWN")
            mistakes.append((path.name, expected, actual))

    return PlateValidationReport(
        total=len(paths),
        correct=correct,
        unknown=unknown,
        mistakes=tuple(mistakes),
    )
