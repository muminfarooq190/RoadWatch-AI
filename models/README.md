# Model weights

Put a location-specific YOLO licence-plate detector at:

```text
models/license_plate_detector.pt
```

Weights are intentionally not committed because they are large and may carry a separate
licence. Set `detection.require_plate_model: true` before treating plate recognition as an
operational requirement.
