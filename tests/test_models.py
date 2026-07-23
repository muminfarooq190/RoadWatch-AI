from roadwatch_ai.models import BoundingBox


def test_bounding_box_bottom_center():
    assert BoundingBox(10, 20, 30, 50).bottom_center == (20.0, 50.0)
