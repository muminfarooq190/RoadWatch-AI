import pytest

from roadwatch_ai.speed import SpeedEstimator, denormalize_line, segments_cross


def test_denormalizes_line_coordinates():
    assert denormalize_line(((0.1, 0.2), (0.9, 0.8)), 200, 100) == (
        (20.0, 20.0),
        (180.0, 80.0),
    )


@pytest.mark.parametrize(
    ("movement", "line", "expected"),
    [
        (((5, 0), (5, 10)), ((0, 5), (10, 5)), True),
        (((20, 0), (20, 10)), ((0, 5), (10, 5)), False),
        (((20, 5), (30, 5)), ((0, 5), (10, 5)), False),
        (((5, 5), (15, 5)), ((0, 5), (10, 5)), True),
        (((0, 0), (10, 10)), ((10, 10), (20, 0)), True),
        (((0, 0), (2, 2)), ((0, 3), (2, 5)), False),
    ],
)
def test_segment_crossing_geometry(movement, line, expected):
    assert segments_cross(*movement, *line) is expected


def make_estimator(**overrides):
    options = {
        "correction_factor": 1.0,
        "stale_track_seconds": 10.0,
        "minimum_travel_seconds": 0.1,
        "maximum_travel_seconds": 5.0,
    }
    options.update(overrides)
    return SpeedEstimator(
        ((0, 10), (100, 10)),
        ((0, 30), (100, 30)),
        distance_meters=10,
        **options,
    )


def test_measures_corrected_speed_once_after_ordered_crossings():
    estimator = make_estimator(correction_factor=1.1)
    assert estimator.observe(7, (50, 0), 0.0) is None
    assert estimator.observe(7, (50, 15), 1.0) is None
    result = estimator.observe(7, (50, 35), 2.0)
    assert result is not None
    assert result.track_id == 7
    assert result.speed_kph == pytest.approx(39.6)
    assert result.elapsed_seconds == pytest.approx(1.0)
    assert result.measured_at_monotonic == 2.0
    assert estimator.observe(7, (50, 5), 3.0) is None
    assert estimator.observe(7, (50, 35), 4.0) is None


def test_ignores_reverse_crossing_order():
    estimator = make_estimator()
    estimator.observe(8, (50, 40), 0.0)
    estimator.observe(8, (50, 20), 1.0)
    assert estimator.observe(8, (50, 0), 2.0) is None


@pytest.mark.parametrize("end_time", [1.05, 7.0])
def test_rejects_elapsed_time_outside_bounds(end_time):
    estimator = make_estimator()
    estimator.observe(9, (50, 0), 0.0)
    estimator.observe(9, (50, 15), 1.0)
    assert estimator.observe(9, (50, 35), end_time) is None


@pytest.mark.parametrize("end_time", [1.1, 6.0])
def test_accepts_elapsed_time_on_bounds(end_time):
    estimator = make_estimator()
    estimator.observe(11, (50, 0), 0.0)
    estimator.observe(11, (50, 15), 1.0)
    assert estimator.observe(11, (50, 35), end_time) is not None


def test_purges_only_stale_tracks():
    estimator = make_estimator(stale_track_seconds=2.0)
    estimator.observe(1, (50, 0), 0.0)
    estimator.observe(2, (50, 0), 1.5)
    estimator.observe(2, (50, 1), 2.1)
    assert 1 not in estimator._tracks
    assert 2 in estimator._tracks
