# RoadWatch AI

RoadWatch AI is a fixed-camera traffic monitoring prototype that:

- detects and tracks cars, motorcycles, buses, and trucks;
- estimates speed from two physically measured road lines;
- reads Indian registration plates locally;
- saves an evidence image and an SQLite event record; and
- sends an SMS when the measured speed exceeds the configured limit.

> [!IMPORTANT]
> This is not a certified speed-enforcement device. Camera-only speed accuracy depends on
> installation geometry, a measured road distance, stable video timestamps, sufficient frame
> rate, and field validation. Do not issue fines from this prototype. Use certified radar/LIDAR
> and comply with local law for enforcement.

## Supported hardware

Use a Raspberry Pi 5, NVIDIA Jetson, or a laptop/mini-PC with a fixed USB, CSI, RTSP, or IP
camera. A normal Arduino cannot run YOLO tracking and optical character recognition. It can
only be used as an optional sensor or trigger beside the computer running this application.

Recommended minimums:

- 1080p fixed camera, 25 FPS or better
- Raspberry Pi 5 (8 GB) for a small model, or a Jetson for higher throughput
- camera mounted rigidly with an unobstructed view of both calibration lines
- night illumination if operation after dark is required

## How speed measurement works

Paint, tape, or identify two lines across the same lane and measure the road-surface distance
between them. Put their normalized image coordinates in `config.yaml`. For each tracked
vehicle, RoadWatch records when the bottom-centre point crosses line 1 and line 2:

```text
speed (km/h) = measured distance (m) / elapsed time (s) × 3.6
```

This is substantially more defensible than guessing speed from bounding-box size. It is still
only as accurate as the calibration, timestamps, tracking stability, and camera frame rate.

## Quick start

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/muminfarooq190/RoadWatch-AI.git
cd RoadWatch-AI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp config.example.yaml config.yaml
cp .env.example .env
python -m roadwatch_ai doctor --config config.yaml
python -m roadwatch_ai run --config config.yaml --dry-run --display
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The first run may download the configured Ultralytics vehicle model. Set `camera.source` to:

- `0` for the first local camera;
- a video path such as `samples/traffic.mp4`; or
- an RTSP/HTTP camera URL.

## Plate detector

The repository does not pretend that OCR over an entire vehicle crop is reliable. For a real
deployment, provide a YOLO licence-plate detector trained for the plates and camera angle in
your location:

```yaml
detection:
  plate_model: models/license_plate_detector.pt
  require_plate_model: true
```

Without that model, RoadWatch uses the lower portion of the vehicle crop as an explicitly
labelled fallback. It is useful for pipeline testing, not dependable plate identification.
EasyOCR then reads the detected plate locally. No plate-recognition API or recurring API bill
is required.

## Camera calibration

1. Fix the camera permanently. Any later movement invalidates calibration.
2. Choose two visible lines across the lane, separated along the direction of travel.
3. Measure the road-surface distance between the lines with a tape or survey tool.
4. Capture a frame and note its pixel width and height.
5. Convert each endpoint to normalized coordinates:
   `normalized_x = pixel_x / frame_width`, `normalized_y = pixel_y / frame_height`.
6. Put both line endpoints and the measured distance in `config.yaml`.
7. Run with `--display`; the lines must visually sit on the measured road marks.
8. Validate against many passes at known speeds in both daylight and expected night conditions.
9. Adjust `speed.correction_factor` only from a documented validation set, never to make one
   convenient sample look correct.

Example for two horizontal lines:

```yaml
speed:
  start_line: [[0.08, 0.58], [0.92, 0.58]]
  end_line: [[0.05, 0.82], [0.95, 0.82]]
  distance_meters: 12.0
  speed_limit_kph: 50.0
  correction_factor: 1.0
```

The start line must be crossed before the end line. Reverse the two in the configuration when
traffic travels in the opposite direction.

## SMS configuration

Twilio credentials and the destination number belong in `.env`, never in committed source:

```dotenv
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
ALERT_TO_NUMBER=+91...
```

Then enable alerts in `config.yaml`:

```yaml
alerts:
  enabled: true
```

Keep `--dry-run` enabled until detection, evidence, and event history have been checked. An SMS
contains the recognized plate, measured speed, configured limit, timestamp, and local evidence
path. It does not upload the image.

## Commands

```bash
# Validate config and installed packages
python -m roadwatch_ai doctor --config config.yaml

# Safe test: no SMS is sent
python -m roadwatch_ai run --config config.yaml --dry-run --display

# Headless production run
python -m roadwatch_ai run --config config.yaml

# Override the camera source for one run
python -m roadwatch_ai run --config config.yaml --source sample.mp4 --dry-run

# Run dependency-free core tests
python -m unittest discover -s tests -v
```

Press `q` to stop a displayed run.

## Raspberry Pi service

After confirming the project manually:

```bash
sudo cp deploy/roadwatch-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now roadwatch-ai
sudo systemctl status roadwatch-ai
```

Edit the user and paths in `deploy/roadwatch-ai.service` first. OpenCV, Torch, Ultralytics, and
EasyOCR can be slow to install on ARM. A 64-bit Raspberry Pi OS and a model exported to an
accelerated format such as NCNN are strongly recommended for useful throughput.

## Event data

By default:

- evidence images: `data/evidence/`
- SQLite database: `data/roadwatch.db`
- log: standard output/systemd journal

The database stores timestamp, track ID, plate, OCR confidence, speed, limit, evidence path,
and whether an alert was sent. Retention is your responsibility. Number plates are personal
data in many jurisdictions; restrict access and delete data when no longer needed.

## Project layout

```text
roadwatch_ai/
  app.py          video pipeline
  detection.py    YOLO vehicle tracking
  speed.py        line-crossing speed estimator
  plates.py       plate crop and OCR
  alerting.py     SMS and duplicate-alert gate
  evidence.py     annotated evidence images
  storage.py      SQLite event history
  config.py       typed YAML and environment config
tests/            dependency-free unit tests
deploy/           systemd service template
```

## Accuracy checklist

Before relying on an installation:

- [ ] fixed, vibration-free camera
- [ ] both road lines accurately surveyed
- [ ] 25 FPS or higher with stable timestamps
- [ ] sufficient pixels on every plate
- [ ] location-specific plate detector
- [ ] daylight, rain, glare, occlusion, and night validation
- [ ] at least 50–100 known-speed validation passes per lane/direction
- [ ] false alert and duplicate alert review
- [ ] legal/privacy review and retention policy

If legal-grade accuracy is required, stop here and use certified radar/LIDAR hardware. Software
cannot turn an uncalibrated consumer camera into a certified enforcement instrument.

## Licence

MIT. Model weights may have their own licences; check them before commercial use.
