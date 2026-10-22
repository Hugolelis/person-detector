import pytest
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


def test_box_fully_inside_has_full_overlap():
    zone = RedZone(SQUARE)
    assert zone.box_overlap((20, 20, 60, 60, 0.9)) == 1.0


def test_box_fully_outside_has_no_overlap():
    zone = RedZone(SQUARE)
    assert zone.box_overlap((150, 20, 190, 60, 0.9)) == 0.0
    assert zone.box_overlap((20, 150, 60, 190, 0.9)) == 0.0
    assert zone.box_overlap((-80, 20, -20, 60, 0.9)) == 0.0


def test_overlap_is_the_fraction_of_the_box_area_inside():
    zone = RedZone(SQUARE)
    assert zone.box_overlap((50, 20, 150, 60, 0.9)) == pytest.approx(0.5, abs=0.02)
    assert zone.box_overlap((50, 50, 150, 150, 0.9)) == pytest.approx(0.25, abs=0.02)


def test_overlap_counts_the_part_hanging_outside_of_the_zone_bounds():
    zone = RedZone(SQUARE)
    assert zone.box_overlap((-50, 20, 50, 60, 0.9)) == pytest.approx(0.5, abs=0.02)


def test_overlap_follows_a_diagonal_edge():
    triangle = RedZone([[0, 0], [100, 0], [0, 100]])
    overlap = triangle.box_overlap((0, 0, 100, 100, 0.9))
    assert 0.48 < overlap < 0.53


def test_degenerate_box_has_no_overlap():
    zone = RedZone(SQUARE)
    assert zone.box_overlap((50, 50, 50, 80, 0.9)) == 0.0


def test_person_is_counted_when_enough_of_the_box_is_inside():
    zone = RedZone(SQUARE, min_overlap_percent=40)
    assert zone.count_people([(50, 20, 150, 60, 0.9)]) == 1


def test_person_is_not_counted_below_the_threshold():
    zone = RedZone(SQUARE, min_overlap_percent=60)
    assert zone.count_people([(50, 20, 150, 60, 0.9)]) == 0


def test_the_threshold_comes_from_the_percentage():
    detections = [(80, 20, 130, 60, 0.9)]
    assert RedZone(SQUARE, min_overlap_percent=30).count_people(detections) == 1
    assert RedZone(SQUARE, min_overlap_percent=50).count_people(detections) == 0


def test_zero_percent_counts_anyone_touching_but_not_anyone_outside():
    zone = RedZone(SQUARE, min_overlap_percent=0)
    assert zone.count_people([(95, 20, 135, 60, 0.9)]) == 1
    assert zone.count_people([(150, 20, 190, 60, 0.9)]) == 0


def test_hundred_percent_needs_the_whole_box_inside():
    zone = RedZone(SQUARE, min_overlap_percent=100)
    assert zone.count_people([(10, 10, 60, 60, 0.9)]) == 1
    assert zone.count_people([(60, 10, 110, 60, 0.9)]) == 0


def test_only_people_inside_are_counted():
    zone = RedZone(SQUARE)
    detections = [
        (40, 10, 60, 50, 0.9),
        (140, 10, 160, 50, 0.8),
        (10, 20, 30, 90, 0.7),
    ]
    assert zone.count_people(detections) == 2
