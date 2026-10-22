from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from src.reporting.chart import (
    ALERT_COLOR,
    OK_COLOR,
    build_bars,
    choose_bin_seconds,
    draw_occupancy,
    set_integer_yaxis,
)


def test_bin_size_grows_with_the_video_duration():
    assert choose_bin_seconds(15) == 0.5
    assert choose_bin_seconds(60) == 2
    assert choose_bin_seconds(10**6) == 600


def test_each_bar_is_the_median_of_its_interval():
    starts, heights = build_bars(
        [0.0, 0.1, 0.2, 1.0, 1.1, 1.2], [4, 9, 4, 2, 2, 3], bin_seconds=1
    )
    assert list(starts) == [0, 1]
    assert list(heights) == [4, 2]


def test_a_single_frame_spike_does_not_change_the_bar():
    _, heights = build_bars([0.0, 0.1, 0.2, 0.3, 0.4], [4, 4, 9, 4, 4], 1)
    assert list(heights) == [4]


def test_median_between_two_counts_rounds_half_up():
    _, heights = build_bars([0.0, 0.1], [2, 3], bin_seconds=1)
    assert list(heights) == [3]


def test_empty_intervals_are_skipped():
    starts, _ = build_bars([0.0, 5.0], [1, 1], bin_seconds=1)
    assert list(starts) == [0, 5]


def test_bars_above_the_limit_are_red_and_the_rest_blue():
    figure = Figure()
    axes = figure.subplots()
    draw_occupancy(axes, [0.0, 1.0, 2.0], [1, 5, 2], limit=3, bin_seconds=1)
    colors = [tuple(bar.get_facecolor()) for bar in axes.patches]
    assert colors == [to_rgba(OK_COLOR), to_rgba(ALERT_COLOR), to_rgba(OK_COLOR)]


def test_time_runs_along_the_horizontal_axis():
    figure = Figure()
    axes = figure.subplots()
    draw_occupancy(axes, [0.0, 1.0, 2.0], [1, 5, 2], limit=3, bin_seconds=1)
    assert axes.get_xlabel() == "Time (s)"
    assert axes.get_ylabel() == "People in zone"
    assert not axes.xaxis_inverted()
    assert axes.get_xlim() == (0, 3)


def test_limit_is_drawn_as_a_horizontal_line():
    figure = Figure()
    axes = figure.subplots()
    draw_occupancy(axes, [0.0, 1.0], [1, 5], limit=3, bin_seconds=1)
    assert any(list(line.get_ydata()) == [3, 3] for line in axes.lines)


def test_x_axis_ticks_are_consecutive_integers():
    figure = Figure()
    axes = figure.subplots()
    set_integer_yaxis(axes, [1, 5, 2], limit=3)
    assert list(axes.get_yticks()) == [0, 1, 2, 3, 4, 5, 6]


def test_x_axis_always_reaches_above_the_limit():
    figure = Figure()
    axes = figure.subplots()
    set_integer_yaxis(axes, [], limit=3)
    assert max(axes.get_yticks()) >= 3
    assert axes.get_ylim()[1] > 3


def test_x_axis_skips_ticks_when_counts_are_large():
    figure = Figure()
    axes = figure.subplots()
    set_integer_yaxis(axes, [60], limit=3)
    assert len(axes.get_yticks()) <= 13
