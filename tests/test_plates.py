import unittest

from roadwatch_ai.plates import looks_like_indian_plate, normalize_plate_text


class PlateTextTests(unittest.TestCase):
    def test_normalizes_spacing_and_punctuation(self) -> None:
        self.assertEqual(normalize_plate_text("dl-01 ab 1234"), "DL01AB1234")

    def test_recognizes_standard_indian_plate(self) -> None:
        self.assertTrue(looks_like_indian_plate("JK01AB1234"))

    def test_recognizes_bharat_series(self) -> None:
        self.assertTrue(looks_like_indian_plate("22BH1234AA"))

    def test_rejects_arbitrary_ocr_text(self) -> None:
        self.assertFalse(looks_like_indian_plate("DELIVERY"))


if __name__ == "__main__":
    unittest.main()
