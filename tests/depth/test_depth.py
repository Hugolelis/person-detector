import sys
import time

import numpy as np
import pytest
from src.depth.estimator import (
    DepthEstimator,
    normalize_depth,
    prepare_input,
)
from src.depth.runner import DepthRunner
from src.depth.view import (
    HIGHLIGHT_COLOR,
    colorize_depth,
    person_depth,
    rank_by_closeness,
    render_depth_view,
)
from src.errors.depth import DepthModelError


def depth_with_square(value, box, size=(120, 160), background=0.1):
    depth = np.full(size, background, dtype=np.float32)
    x1, y1, x2, y2 = box
    depth[y1:y2, x1:x2] = value
    return depth


def test_input_sides_are_multiples_of_the_patch_size():
    array = prepare_input(np.zeros((720, 1280, 3), dtype=np.uint8), input_size=266)
    channels, height, width = array.shape
    assert channels == 3
    assert height == 266
    assert width % 14 == 0 and width > height
    assert array.dtype == np.float32


def test_portrait_frames_keep_their_orientation():
    array = prepare_input(np.zeros((1280, 720, 3), dtype=np.uint8), input_size=266)
    assert array.shape[2] == 266
    assert array.shape[1] > array.shape[2]


def test_input_is_normalized_like_imagenet():
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    array = prepare_input(white, input_size=28)
    assert array[0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)


def test_depth_is_scaled_to_zero_one_with_one_as_closest():
    depth = np.linspace(5, 50, 10000).reshape(100, 100)
    normalized = normalize_depth(depth)
    assert normalized.min() == 0 and normalized.max() == 1
    assert normalized[-1, -1] > normalized[0, 0]


def test_a_flat_depth_map_does_not_divide_by_zero():
    assert normalize_depth(np.full((10, 10), 3.0)).max() == 0


def test_person_depth_ignores_background_around_the_box():
    box = (40, 20, 80, 100)
    depth = depth_with_square(0.9, (50, 60, 70, 100))
    assert person_depth(depth, box) == pytest.approx(0.9, abs=0.01)


def test_person_depth_is_none_for_a_box_outside_the_map():
    assert person_depth(np.zeros((50, 50), np.float32), (200, 200, 240, 260)) is None


def test_people_are_ranked_from_closest_to_farthest():
    depth = depth_with_square(0.9, (100, 40, 140, 110))
    depth[40:110, 10:50] = 0.4
    detections = [(10, 40, 50, 110, 0.9), (100, 40, 140, 110, 0.8)]
    ranking = rank_by_closeness(depth, detections)
    assert [index for index, _ in ranking] == [1, 0]


def test_ranking_is_empty_without_detections():
    assert rank_by_closeness(np.zeros((10, 10), np.float32), []) == []


def test_depth_view_marks_the_closest_person():
    depth = depth_with_square(0.9, (100, 40, 140, 110))
    detections = [(100, 40, 140, 110, 0.8)]
    ranking = rank_by_closeness(depth, detections)
    view = render_depth_view(depth, detections, ranking)
    assert view.shape == (120, 160, 3)
    assert tuple(view[75, 100]) == HIGHLIGHT_COLOR


def test_colorized_depth_is_a_bgr_image():
    assert colorize_depth(np.zeros((12, 16), np.float32)).shape == (12, 16, 3)


class FakeEstimator:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.frames = []

    def estimate(self, frame):
        time.sleep(self.delay)
        self.frames.append(frame)
        return np.full(frame.shape[:2], 0.5, dtype=np.float32)


def frame():
    return np.zeros((12, 16, 3), dtype=np.uint8)


def test_runner_only_estimates_every_stride_th_frame():
    estimator = FakeEstimator()
    runner = DepthRunner(estimator, stride=3, asynchronous=False)
    for _ in range(7):
        runner.submit(frame())
    assert len(estimator.frames) == 3


def test_latest_is_none_before_the_first_estimate():
    runner = DepthRunner(FakeEstimator(delay=0.2), stride=1)
    try:
        runner.submit(frame())
        assert runner.latest() is None
    finally:
        runner.close()


def test_runner_copies_the_frame_so_drawing_on_it_is_safe():
    estimator = FakeEstimator()
    runner = DepthRunner(estimator, stride=1, asynchronous=False)
    original = frame()
    runner.submit(original)
    original[:] = 255
    assert estimator.frames[0].max() == 0


def test_background_runner_publishes_the_latest_map_without_blocking():
    runner = DepthRunner(FakeEstimator(delay=0.1), stride=1)
    try:
        started = time.perf_counter()
        runner.submit(frame())
        assert time.perf_counter() - started < 0.05
        runner.wait_until_idle()
        assert runner.latest().shape == (12, 16)
    finally:
        runner.close()


def test_frames_are_dropped_while_an_estimate_is_running():
    estimator = FakeEstimator(delay=0.2)
    runner = DepthRunner(estimator, stride=1)
    try:
        for _ in range(5):
            runner.submit(frame())
        runner.wait_until_idle()
    finally:
        runner.close()
    assert len(estimator.frames) == 1


def test_estimator_errors_surface_as_a_depth_error():
    class Broken:
        def estimate(self, frame):
            raise RuntimeError("boom")

    runner = DepthRunner(Broken(), stride=1, asynchronous=False)
    runner.submit(frame())
    with pytest.raises(DepthModelError, match="boom"):
        runner.latest()


def test_close_stops_the_background_thread():
    runner = DepthRunner(FakeEstimator(), stride=1)
    runner.close()
    assert not runner._thread.is_alive()


def test_missing_transformers_gives_install_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(DepthModelError, match="poetry add transformers"):
        DepthEstimator()
