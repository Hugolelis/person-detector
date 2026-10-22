from src.redzone import RedZone

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def test_point_inside_zone():
    zone = RedZone(SQUARE)
    assert zone.contains((50, 50))


def test_point_outside_zone():
    zone = RedZone(SQUARE)
    assert not zone.contains((150, 50))


def test_point_on_border_counts_as_inside():
    zone = RedZone(SQUARE)
    assert zone.contains((0, 50))


def test_count_is_zero_without_detections():
    zone = RedZone(SQUARE)
    assert zone.count_people([]) == 0


def test_person_with_feet_inside_is_counted():
    zone = RedZone(SQUARE)
    detections = [(40, 10, 60, 50, 0.9)]
    assert zone.count_people(detections) == 1


def test_person_with_feet_outside_is_not_counted_even_if_body_is_inside():
    zone = RedZone(SQUARE)
    detections = [(40, 50, 60, 120, 0.9)]
    assert zone.count_people(detections) == 0


def test_person_partially_outside_is_counted_if_feet_are_inside():
    zone = RedZone(SQUARE)
    detections = [(40, -50, 60, 50, 0.9)]
    assert zone.count_people(detections) == 1


def test_only_people_inside_are_counted():
    zone = RedZone(SQUARE)
    detections = [
        (40, 10, 60, 50, 0.9),
        (140, 10, 160, 50, 0.8),
        (10, 20, 30, 90, 0.7),
    ]
    assert zone.count_people(detections) == 2
