from src.closest_tracker import ClosestTracker, iou

A = (10, 10, 50, 90, 0.9)
B = (100, 10, 140, 90, 0.9)


def ranking(*indexes):
    return [(index, 1.0 - 0.1 * position) for position, index in enumerate(indexes)]


def feed(tracker, detections, indexes, frames):
    return [tracker.update(detections, ranking(*indexes)) for _ in range(frames)]


def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    assert iou(A, A) == 1.0
    assert iou(A, B) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 5 / 15


def test_a_new_closest_person_is_confirmed_after_the_configured_frames():
    tracker = ClosestTracker(confirm_frames=5)
    results = feed(tracker, [A, B], [0, 1], frames=6)
    assert results == [None, None, None, None, 0, 0]


def test_one_frame_confirms_immediately():
    assert ClosestTracker(confirm_frames=1).update([A, B], ranking(0, 1)) == 0


def test_a_short_flicker_does_not_change_the_closest_person():
    tracker = ClosestTracker(confirm_frames=5)
    feed(tracker, [A, B], [0, 1], frames=5)
    flicker = feed(tracker, [A, B], [1, 0], frames=4)
    assert flicker == [0, 0, 0, 0]
    assert tracker.update([A, B], ranking(0, 1)) == 0


def test_alternating_rankings_never_switch():
    tracker = ClosestTracker(confirm_frames=3)
    feed(tracker, [A, B], [0, 1], frames=3)
    for _ in range(6):
        assert tracker.update([A, B], ranking(1, 0)) == 0
        assert tracker.update([A, B], ranking(1, 0)) == 0
        assert tracker.update([A, B], ranking(0, 1)) == 0


def test_a_lasting_change_switches_after_the_configured_frames():
    tracker = ClosestTracker(confirm_frames=5)
    feed(tracker, [A, B], [0, 1], frames=5)
    results = feed(tracker, [A, B], [1, 0], frames=5)
    assert results == [0, 0, 0, 0, 1]


def test_the_confirmed_person_is_followed_through_small_moves_and_reordering():
    tracker = ClosestTracker(confirm_frames=2)
    feed(tracker, [A, B], [0, 1], frames=2)
    moved = (14, 12, 54, 92, 0.9)
    # The detector now lists the other person first, but the closest one is the same.
    assert tracker.update([B, moved], ranking(1, 0)) == 1


def test_the_closest_person_is_dropped_after_being_missing_for_the_confirmation():
    tracker = ClosestTracker(confirm_frames=3)
    feed(tracker, [A, B], [0, 1], frames=3)
    results = [tracker.update([B], []) for _ in range(4)]
    assert results == [None, None, None, None]
    assert tracker.confirmed is None


def test_a_brief_miss_keeps_the_confirmed_person():
    tracker = ClosestTracker(confirm_frames=3)
    feed(tracker, [A, B], [0, 1], frames=3)
    assert tracker.update([B], ranking(0)) is None
    assert tracker.update([A, B], ranking(0, 1)) == 0


def test_without_ranking_nothing_changes():
    tracker = ClosestTracker(confirm_frames=2)
    assert tracker.update([], []) is None
