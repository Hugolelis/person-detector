import pytest
from src.gui.fps_meter import FpsMeter, describe_speed


def test_current_fps_follows_the_arrival_rate():
    meter = FpsMeter()
    for index in range(5):
        meter.tick(index * 0.2)
    assert meter.current_fps() == pytest.approx(5.0)


def test_current_fps_is_unknown_with_a_single_frame():
    meter = FpsMeter()
    meter.tick(0.0)
    assert meter.current_fps() is None


def test_old_frames_leave_the_window():
    meter = FpsMeter(window_seconds=2.0)
    for index in range(10):
        meter.tick(index * 1.0)
    for index in range(1, 31):
        meter.tick(10 + index * 0.1)
    assert meter.current_fps() == pytest.approx(10.0)


def test_slow_processing_still_gives_a_reading():
    meter = FpsMeter(window_seconds=2.0)
    meter.tick(0.0)
    meter.tick(5.0)
    assert meter.current_fps() == pytest.approx(0.2)


def test_average_ignores_time_spent_paused():
    meter = FpsMeter()
    meter.tick(0.0)
    meter.tick(1.0)
    meter.pause(1.5)
    meter.resume(11.5)
    meter.tick(12.0)
    meter.tick(13.0)
    assert meter.average_fps() == pytest.approx(3 / 3.0)


def test_average_does_not_count_the_time_before_the_first_frame():
    meter = FpsMeter()
    meter.pause(0.0)
    meter.resume(30.0)
    meter.tick(40.0)
    meter.tick(41.0)
    assert meter.average_fps() == pytest.approx(1.0)


def test_reset_clears_everything():
    meter = FpsMeter()
    meter.tick(0.0)
    meter.tick(1.0)
    meter.reset()
    assert meter.frames == 0
    assert meter.current_fps() is None
    assert meter.average_fps() is None


def test_speed_is_described_relative_to_real_time():
    assert describe_speed(6.0, 24.0) == "6.0 fps (0.25x real time)"
    assert describe_speed(6.0, None) == "6.0 fps"
