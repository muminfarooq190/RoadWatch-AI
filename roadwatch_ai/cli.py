from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

from roadwatch_ai.config import load_config


def _source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roadwatch-ai",
        description="Fixed-camera vehicle speed, plate OCR, evidence, and SMS alerts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the traffic monitoring pipeline")
    run_parser.add_argument("--config", default="config.yaml")
    run_parser.add_argument("--source", type=_source)
    run_parser.add_argument("--dry-run", action="store_true", help="never send SMS")
    run_parser.add_argument("--display", action="store_true", help="show the annotated video")

    doctor_parser = subparsers.add_parser("doctor", help="validate config and dependencies")
    doctor_parser.add_argument("--config", default="config.yaml")
    return parser


def _doctor(config_path: str) -> int:
    config = load_config(config_path)
    modules = {
        "cv2": "OpenCV",
        "easyocr": "EasyOCR",
        "ultralytics": "Ultralytics",
        "yaml": "PyYAML",
        "dotenv": "python-dotenv",
        "twilio": "Twilio",
    }
    missing = [
        package for module, package in modules.items() if importlib.util.find_spec(module) is None
    ]
    if missing:
        print(f"Missing packages: {', '.join(missing)}", file=sys.stderr)
        print("Install with: python -m pip install -e .", file=sys.stderr)
        return 1

    plate_model = Path(config.detection.plate_model)
    if not plate_model.is_file():
        message = f"Plate model not found: {plate_model}"
        if config.detection.require_plate_model:
            print(message, file=sys.stderr)
            return 1
        print(f"WARNING: {message}; fallback OCR is not deployment-grade")
    print("Configuration and Python dependencies are valid.")
    print(f"Camera source: {config.camera.source}")
    print(f"Speed limit: {config.speed.speed_limit_kph:.1f} km/h")
    print(f"Measured line distance: {config.speed.distance_meters:.2f} m")
    print(f"SMS enabled: {config.alerts.enabled}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        logging.basicConfig(
            level=getattr(logging, config.runtime.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        if args.command == "doctor":
            return _doctor(args.config)

        from roadwatch_ai.app import run

        return run(
            config,
            dry_run=args.dry_run,
            display_override=True if args.display else None,
            source_override=args.source,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"RoadWatch error: {exc}", file=sys.stderr)
        return 2
