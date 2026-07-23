import unittest

from roadwatch_ai.speed import SpeedEstimator, segments_cross


class SegmentCrossingTests(unittest.TestCase):
    def test_crosses_finite_line(self) -> None:
        self.assertTrue(segments_cross((5, 0), (5, 10), (0, 5), (10, 5)))

    def test_does_not_cross_outside_line_endpoints(self) -> None:
        self.assertFalse(segments_cross((20, 0), (20, 10), (0, 5), (10, 5)))

    def test_collinear_but_disjoint_segments_do_not_cross(self) -> None:
        self.assertFalse(segments_cross((20, 5), (30, 5), (0, 5), (10, 5)))


class SpeedEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = SpeedEstimator(
            ((0, 10), (100, 10)),
            ((0, 30), (100, 30)),
            distance_meters=10,
            minimum_travel_seconds=0.1,
            maximum_travel_seconds=5,
        )

    def test_measures_speed_after_ordered_crossings(self) -> None:
        self.assertIsNone(self.estimator.observe(7, (50, 0), 0.0))
        self.assertIsNone(self.estimator.observe(7, (50, 15), 1.0))
        result = self.estimator.observe(7, (50, 35), 2.0)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.speed_kph, 36.0)
        self.assertAlmostEqual(result.elapsed_seconds, 1.0)

    def test_ignores_reverse_order(self) -> None:
        self.estimator.observe(8, (50, 40), 0.0)
        self.estimator.observe(8, (50, 20), 1.0)
        result = self.estimator.observe(8, (50, 0), 2.0)
        self.assertIsNone(result)

    def test_rejects_impossible_elapsed_time(self) -> None:
        self.estimator.observe(9, (50, 0), 0.0)
        self.estimator.observe(9, (50, 15), 1.0)
        result = self.estimator.observe(9, (50, 35), 7.0)
        self.assertIsNone(result)

    def test_emits_only_once_per_track(self) -> None:
        self.estimator.observe(10, (50, 0), 0.0)
        self.estimator.observe(10, (50, 15), 1.0)
        self.assertIsNotNone(self.estimator.observe(10, (50, 35), 2.0))
        self.assertIsNone(self.estimator.observe(10, (50, 5), 3.0))
        self.assertIsNone(self.estimator.observe(10, (50, 35), 4.0))


if __name__ == "__main__":
    unittest.main()
