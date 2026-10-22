import numpy as np
from src.chart import _moving_median


def test_single_frame_spike_is_removed():
    values = np.array([4, 4, 4, 9, 4, 4, 4])
    assert list(_moving_median(values, 3)) == [4, 4, 4, 4, 4, 4, 4]


def test_length_is_preserved():
    assert len(_moving_median(np.arange(10), 5)) == 10


def test_even_window_is_accepted():
    assert len(_moving_median(np.arange(10), 4)) == 10


def test_constant_series_is_unchanged():
    values = np.full(8, 3)
    assert list(_moving_median(values, 5)) == [3] * 8


def test_window_of_one_changes_nothing():
    values = np.array([1, 5, 2])
    assert list(_moving_median(values, 1)) == [1, 5, 2]
