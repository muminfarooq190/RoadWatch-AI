import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from roadwatch_ai.config import DetectionConfig
from roadwatch_ai.models import BoundingBox
from roadwatch_ai.plates import (
    PlateReader,
    PlateValidationReport,
    _coerce_characters,
    canonicalize_indian_plate,
    looks_like_indian_plate,
    normalize_plate_text,
    select_plate_candidate,
    validate_plate_directory,
)

OCR_BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


class FakeReader:
    def __init__(self, *batches):
        self.batches = list(batches)
        self.images = []

    def readtext(self, image, **options):
        self.images.append((image, options))
        return self.batches.pop(0) if self.batches else []


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def argmax(self):
        return FakeTensor(int(np.argmax(self.value)))

    def item(self):
        return self.value

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def __getitem__(self, index):
        return FakeTensor(self.value[index])


class FakeBoxes:
    def __init__(self, xyxy, confidence):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(confidence)

    def __len__(self):
        return len(self.xyxy.value)


class FakePlateModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, crop, **options):
        self.calls.append((crop, options))
        return self.results


def _vehicle_image() -> np.ndarray:
    image = np.zeros((240, 480, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 145), (370, 205), (255, 255, 255), -1)
    cv2.rectangle(image, (120, 145), (370, 205), (0, 0, 0), 3)
    cv2.putText(
        image,
        "JK01AB1234",
        (135, 187),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return image


def test_normalizes_and_recognizes_supported_formats():
    assert normalize_plate_text(" dl-01 ab 1234 ") == "DL01AB1234"
    assert looks_like_indian_plate("JK01AB1234")
    assert looks_like_indian_plate("22BH1234AA")
    assert not looks_like_indian_plate("DELIVERY")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DL0IABI234", "DL01AB1234"),
        ("jk-01-ab-1234", "JK01AB1234"),
        ("2Z8H1234AA", "22BH1234AA"),
        ("DL1CAA1111", "DL1CAA1111"),
        ("DL0AB1234", None),
        ("DELIVERY", None),
        ("@@@", None),
    ],
)
def test_canonicalizes_common_ocr_confusions(raw, expected):
    assert canonicalize_indian_plate(raw) == expected


def test_character_coercion_rejects_impossible_characters():
    assert _coerce_characters("AB", letters=True) == ("AB", 0)
    assert _coerce_characters("08", letters=True) == ("OB", 2)
    assert _coerce_characters("12", letters=False) == ("12", 0)
    assert _coerce_characters("OZ", letters=False) == ("02", 2)
    assert _coerce_characters("@", letters=True) is None
    assert _coerce_characters("@", letters=False) is None


def test_selects_combined_valid_plate_and_penalizes_corrections():
    batches = [
        [
            (OCR_BOX, "", 0.99),
            (OCR_BOX, "NOISE", 0.10),
            ([[0, 0], [30, 0], [30, 20], [0, 20]], "JK01", 0.92),
            ([[40, 0], [110, 0], [110, 20], [40, 20]], "ABI234", 0.90),
        ],
        [(OCR_BOX, "DL01AB1234", 0.80)],
    ]
    selected = select_plate_candidate(batches, min_confidence=0.30)
    assert selected is not None
    assert selected[0] == "JK01AB1234"
    assert selected[1] == pytest.approx(0.86)


def test_selects_best_duplicate_and_returns_none_without_valid_plate():
    duplicate_batches = [
        [(OCR_BOX, "JK01AB1234", 0.70)],
        [(OCR_BOX, "JK01AB1234", 0.93)],
    ]
    assert select_plate_candidate(duplicate_batches, 0.3) == ("JK01AB1234", 0.93)
    assert select_plate_candidate([[(OCR_BOX, "DELIVERY", 0.99)]], 0.3) is None


def test_reader_uses_injected_dependencies_without_importing_easyocr():
    reader = FakeReader([], [])
    model = FakePlateModel([])
    plate_reader = PlateReader(DetectionConfig(), reader=reader, plate_model=model)
    assert plate_reader._reader is reader
    assert plate_reader._plate_model is model


def test_reader_reports_missing_easyocr():
    real_import = builtins.__import__

    def fail_easyocr(name, *args, **kwargs):
        if name == "easyocr":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_easyocr):
        with pytest.raises(RuntimeError, match="EasyOCR is missing"):
            PlateReader(DetectionConfig(), plate_model=None)


def test_reader_constructs_easyocr_reader():
    easyocr = ModuleType("easyocr")
    created = []

    def reader_factory(languages, gpu):
        created.append((languages, gpu))
        return FakeReader()

    easyocr.Reader = reader_factory
    with patch.dict(sys.modules, {"easyocr": easyocr}):
        plate_reader = PlateReader(DetectionConfig(ocr_languages=("en", "hi")), plate_model=None)
    assert created == [(["en", "hi"], False)]
    assert isinstance(plate_reader._reader, FakeReader)


def test_reader_requires_configured_plate_model(tmp_path):
    config = DetectionConfig(plate_model=str(tmp_path / "missing.pt"), require_plate_model=True)
    with pytest.raises(FileNotFoundError, match="Required plate detector"):
        PlateReader(config, reader=FakeReader())


def test_reader_loads_existing_yolo_plate_model(tmp_path):
    model_path = tmp_path / "plate.pt"
    model_path.write_bytes(b"weights")
    loaded = []
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = lambda path: loaded.append(path) or "MODEL"
    with patch.dict(sys.modules, {"ultralytics": ultralytics}):
        plate_reader = PlateReader(
            DetectionConfig(plate_model=str(model_path)), reader=FakeReader()
        )
    assert loaded == [str(model_path)]
    assert plate_reader._plate_model == "MODEL"


def test_reader_uses_opencv_fallback_and_reads_synthetic_plate(tmp_path, caplog):
    config = DetectionConfig(plate_model=str(tmp_path / "missing.pt"))
    ocr = FakeReader(
        [(OCR_BOX, "JK0IABI234", 0.96)],
        [(OCR_BOX, "NOT A PLATE", 0.99)],
    )
    plate_reader = PlateReader(config, reader=ocr)
    frame = _vehicle_image()

    result = plate_reader.read(frame, BoundingBox(0, 0, 480, 240))

    assert result.text == "JK01AB1234"
    assert result.confidence == pytest.approx(0.86)
    assert result.crop_box is not None
    assert result.crop_box.y1 >= 100
    assert len(ocr.images) == 2
    assert ocr.images[0][1]["allowlist"].endswith("0123456789")
    assert "OpenCV candidate detection" in caplog.text


def test_reader_clamps_vehicle_box_and_returns_unknown_for_empty_crop():
    reader = PlateReader(DetectionConfig(), reader=FakeReader(), plate_model=None)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert reader.read(frame, BoundingBox(30, 30, 40, 40)).text == "UNKNOWN"


def test_read_crop_handles_empty_and_unrecognized_images():
    reader = PlateReader(
        DetectionConfig(),
        reader=FakeReader([], []),
        plate_model=None,
    )
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    box = BoundingBox(1, 2, 3, 4)
    assert reader.read_crop(empty, box).crop_box == box

    crop = np.zeros((40, 160, 3), dtype=np.uint8)
    result = reader.read_crop(crop, box)
    assert result.text == "UNKNOWN"
    assert result.confidence == 0.0
    assert result.crop_box == box


@pytest.mark.parametrize(
    "results",
    [
        [],
        [SimpleNamespace(boxes=None)],
        [SimpleNamespace(boxes=FakeBoxes([], []))],
    ],
)
def test_yolo_plate_detector_returns_empty_when_no_box(results):
    model = FakePlateModel(results)
    reader = PlateReader(DetectionConfig(), reader=FakeReader(), plate_model=model)
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    vehicle_box = BoundingBox(10, 20, 210, 120)

    plate_crop, crop_box = reader._find_plate(crop, vehicle_box)

    assert plate_crop.size == 0
    assert crop_box == vehicle_box


def test_yolo_plate_detector_selects_highest_confidence_box():
    boxes = FakeBoxes([[1, 2, 50, 20], [20, 30, 180, 80]], [0.4, 0.9])
    model = FakePlateModel([SimpleNamespace(boxes=boxes)])
    reader = PlateReader(
        DetectionConfig(plate_confidence=0.55),
        reader=FakeReader(),
        plate_model=model,
    )
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    plate_crop, crop_box = reader._find_plate(crop, BoundingBox(10, 20, 210, 120))

    assert plate_crop.shape == (50, 160, 3)
    assert crop_box == BoundingBox(30, 50, 190, 100)
    assert model.calls[0][1] == {"conf": 0.55, "verbose": False}


def test_reader_returns_unknown_when_yolo_finds_no_plate():
    model = FakePlateModel([])
    reader = PlateReader(DetectionConfig(), reader=FakeReader(), plate_model=model)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert reader.read(frame, BoundingBox(0, 0, 200, 100)).text == "UNKNOWN"


def test_opencv_fallback_uses_lower_crop_without_candidate(monkeypatch):
    monkeypatch.setattr(cv2, "findContours", lambda *args: ([], None))
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    plate_crop, crop_box = PlateReader._find_plate_with_opencv(crop, BoundingBox(10, 20, 210, 120))
    assert plate_crop.shape == (50, 160, 3)
    assert crop_box == BoundingBox(30, 70, 190, 120)


def test_opencv_fallback_filters_invalid_contours(monkeypatch):
    contours = ["flat", "narrow", "tiny", "valid"]
    rectangles = {
        "flat": (0, 0, 40, 0),
        "narrow": (0, 0, 20, 40),
        "tiny": (0, 0, 30, 10),
        "valid": (10, 40, 120, 30),
    }
    monkeypatch.setattr(cv2, "findContours", lambda *args: (contours, None))
    monkeypatch.setattr(cv2, "boundingRect", lambda contour: rectangles[contour])
    monkeypatch.setattr(cv2, "contourArea", lambda contour: 2_000.0)
    crop = np.zeros((100, 200, 3), dtype=np.uint8)

    plate_crop, crop_box = PlateReader._find_plate_with_opencv(crop, BoundingBox(5, 7, 205, 107))

    assert plate_crop.shape == (36, 128, 3)
    assert crop_box == BoundingBox(11, 44, 139, 80)


def _write_image(path):
    assert cv2.imwrite(str(path), np.zeros((30, 100, 3), dtype=np.uint8))


def test_validates_labelled_plate_directory(tmp_path):
    _write_image(tmp_path / "JK01AB1234__front.jpg")
    _write_image(tmp_path / "MH12CD5678__night.jpeg")
    _write_image(tmp_path / "DL01AB1234__rain.png")
    readings = iter(
        [
            SimpleNamespace(text="XX01YY9999"),
            SimpleNamespace(text="JK01AB1234"),
            SimpleNamespace(text="UNKNOWN"),
        ]
    )
    reader = SimpleNamespace(read_crop=lambda image: next(readings))

    report = validate_plate_directory(reader, tmp_path)

    assert report.total == 3
    assert report.correct == 1
    assert report.unknown == 1
    assert report.accuracy == pytest.approx(1 / 3)
    assert {mistake[0] for mistake in report.mistakes} == {
        "DL01AB1234__rain.png",
        "MH12CD5678__night.jpeg",
    }


def test_plate_directory_validation_requires_images(tmp_path):
    with pytest.raises(ValueError, match="No JPG or PNG"):
        validate_plate_directory(SimpleNamespace(), tmp_path)


def test_plate_directory_validation_rejects_invalid_filename(tmp_path):
    _write_image(tmp_path / "DELIVERY.jpg")
    with pytest.raises(ValueError, match="Invalid expected plate"):
        validate_plate_directory(SimpleNamespace(), tmp_path)


def test_plate_directory_validation_rejects_unreadable_image(tmp_path):
    (tmp_path / "JK01AB1234__broken.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Could not read plate image"):
        validate_plate_directory(SimpleNamespace(), tmp_path)


def test_validation_report_accuracy_for_perfect_result():
    report = PlateValidationReport(2, 2, 0, ())
    assert report.accuracy == 1.0
