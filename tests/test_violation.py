from src.enums.violation_event import ViolationEvent
from src.violation import ViolationDetector

STARTED = ViolationEvent.STARTED
ENDED = ViolationEvent.ENDED


def make_detector(start=3, end=3, cooldown=2, limit=3):
    return ViolationDetector(
        person_limit=limit,
        frames_to_confirm_start=start,
        frames_to_confirm_end=end,
        cooldown_frames=cooldown,
    )


def feed(detector, counts):
    return [detector.update(count) for count in counts]


def test_count_at_the_limit_is_not_a_violation():
    events = feed(make_detector(), [3, 3, 3, 3])
    assert events == [None, None, None, None]


def test_violation_starts_after_enough_consecutive_frames():
    events = feed(make_detector(), [4, 4, 4])
    assert events == [None, None, STARTED]


def test_interruption_resets_the_start_counter():
    events = feed(make_detector(), [4, 4, 2, 4, 4, 4])
    assert events == [None, None, None, None, None, STARTED]


def test_violation_ends_after_enough_normal_frames():
    events = feed(make_detector(), [4, 4, 4, 2, 2, 2])
    assert events == [None, None, STARTED, None, None, ENDED]


def test_flicker_inside_a_violation_does_not_end_it():
    events = feed(make_detector(), [4, 4, 4, 2, 4, 2, 2, 2])
    assert events == [None, None, STARTED, None, None, None, None, ENDED]


def test_cooldown_delays_the_next_violation():
    counts = [4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4]
    with_cooldown = feed(make_detector(cooldown=2), counts)
    without_cooldown = feed(make_detector(cooldown=0), counts)
    assert with_cooldown.index(STARTED, 3) == 10
    assert without_cooldown.index(STARTED, 3) == 8


def test_in_violation_flag_follows_the_state():
    detector = make_detector()
    feed(detector, [4, 4])
    assert not detector.in_violation
    detector.update(4)
    assert detector.in_violation