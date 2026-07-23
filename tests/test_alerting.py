import unittest

from roadwatch_ai.alerting import AlertGate


class AlertGateTests(unittest.TestCase):
    def test_suppresses_duplicate_during_cooldown(self) -> None:
        gate = AlertGate(60)
        self.assertTrue(gate.allow("JK01AB1234", 100.0))
        self.assertFalse(gate.allow("JK01AB1234", 130.0))
        self.assertTrue(gate.allow("JK01AB1234", 161.0))

    def test_tracks_different_plates_independently(self) -> None:
        gate = AlertGate(60)
        self.assertTrue(gate.allow("JK01AB1234", 100.0))
        self.assertTrue(gate.allow("DL01AB1234", 101.0))


if __name__ == "__main__":
    unittest.main()
