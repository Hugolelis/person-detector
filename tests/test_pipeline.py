import csv

import numpy as np
import pytest
from src.enums.violation_event import ViolationEvent
from src.pipeline import Pipeline

DEFAULT_BOXES = [(10, 10, 20, 40, 0.9), (60, 10, 70, 40, 0.8)]


class FakeDetector:
    instances = []
    boxes = DEFAULT_BOXES

    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.seen_shapes = []
        FakeDetector.instances.append(self)

    def detect(self, frame):
        self.calls += 1
        self.seen_shapes.append(frame.shape)
        return list(FakeDetector.boxes)


def make_config(stride=1, processing_width=None, min_overlap_percent=None):
    detection = {"model": "fake.pt", "confidence_threshold": 0.4}
    if stride != 1:
        detection["stride"] = stride
    config = {
        "roi": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "person_limit": 1,
        "violation": {
            "frames_to_confirm_start": 2,
            "frames_to_confirm_end": 2,
            "cooldown_frames": 0,
        },
        "clip": {"seconds_before": 0.2, "seconds_after": 0.2},
        "detection": detection,
    }
    if min_overlap_percent is not None:
        config["counting"] = {"min_overlap_percent": min_overlap_percent}
    if processing_width is not None:
        config["processing"] = {"width": processing_width}
    return config


@pytest.fixture(autouse=True)
def fake_detector(monkeypatch):
    FakeDetector.instances = []
    FakeDetector.boxes = DEFAULT_BOXES
    monkeypatch.setattr("src.pipeline.PersonDetector", FakeDetector)


def make_frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_process_counts_people_inside_the_zone(tmp_path):
    with Pipeline(make_config(), fps=10, run_dir=tmp_path) as pipeline:
        result = pipeline.process(make_frame(), 0)
    assert result.count == 2
    assert len(result.detections) == 2
    assert result.event is None
    assert result.annotated_frame.shape == (120, 160, 3)


def test_violation_starts_after_the_configured_number_of_frames(tmp_path):
    with Pipeline(make_config(), fps=10, run_dir=tmp_path) as pipeline:
        first = pipeline.process(make_frame(), 0)
        second = pipeline.process(make_frame(), 1)
    assert first.event is None
    assert second.event is ViolationEvent.STARTED
    assert second.in_violation


def test_clip_and_event_log_are_written(tmp_path):
    with Pipeline(make_config(), fps=10, run_dir=tmp_path) as pipeline:
        for index in range(6):
            pipeline.process(make_frame(), index)
    assert (tmp_path / "clips" / "violation_001.mp4").exists()
    with open(tmp_path / "csv" / "events.csv", newline="") as file:
        events = list(csv.DictReader(file))
    assert len(events) == 1
    assert events[0]["start_frame"] == "1"
    assert events[0]["peak_count"] == "2"


def test_series_has_one_row_per_processed_frame(tmp_path):
    with Pipeline(make_config(), fps=10, run_dir=tmp_path) as pipeline:
        for index in range(5):
            pipeline.process(make_frame(), index)
    with open(tmp_path / "csv" / "series.csv", newline="") as file:
        assert len(list(csv.DictReader(file))) == 5


def test_stride_runs_the_detector_only_every_nth_frame(tmp_path):
    with Pipeline(make_config(stride=3), fps=10, run_dir=tmp_path) as pipeline:
        for index in range(7):
            pipeline.process(make_frame(), index)
    assert FakeDetector.instances[0].calls == 3


def test_stride_still_counts_every_frame(tmp_path):
    with Pipeline(make_config(stride=3), fps=10, run_dir=tmp_path) as pipeline:
        results = [pipeline.process(make_frame(), index) for index in range(4)]
    assert [result.count for result in results] == [2, 2, 2, 2]


def test_large_frames_are_downscaled_once_before_everything_else(tmp_path):
    big_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    config = make_config(processing_width=200)
    with Pipeline(config, fps=10, run_dir=tmp_path, frame_width=400) as pipeline:
        result = pipeline.process(big_frame, 0)
    assert FakeDetector.instances[0].seen_shapes == [(150, 200, 3)]
    assert result.annotated_frame.shape == (150, 200, 3)


def test_zone_is_scaled_together_with_the_frame(tmp_path):
    FakeDetector.boxes = [(150, 10, 160, 50, 0.9)]
    big_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    config = make_config(processing_width=200)
    config["roi"] = [[0, 0], [200, 0], [200, 200], [0, 200]]
    with Pipeline(config, fps=10, run_dir=tmp_path, frame_width=400) as pipeline:
        result = pipeline.process(big_frame, 0)
    assert result.count == 0


def test_frames_already_small_enough_are_left_alone(tmp_path):
    config = make_config(processing_width=1280)
    with Pipeline(config, fps=10, run_dir=tmp_path, frame_width=160) as pipeline:
        pipeline.process(make_frame(), 0)
    assert FakeDetector.instances[0].seen_shapes == [(120, 160, 3)]


def test_a_width_of_zero_disables_the_downscale(tmp_path):
    big_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    config = make_config(processing_width=0)
    with Pipeline(config, fps=10, run_dir=tmp_path, frame_width=400) as pipeline:
        result = pipeline.process(big_frame, 0)
    assert result.annotated_frame.shape == (300, 400, 3)


def test_measured_times_are_logged_instead_of_frame_over_fps(tmp_path):
    with Pipeline(make_config(), fps=30, run_dir=tmp_path) as pipeline:
        for index in range(6):
            pipeline.process(make_frame(), index, time_s=index * 0.5)
    with open(tmp_path / "csv" / "series.csv", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [float(row["timestamp"]) for row in rows] == [0, 0.5, 1, 1.5, 2, 2.5]
    with open(tmp_path / "csv" / "events.csv", newline="") as file:
        events = list(csv.DictReader(file))
    assert events[0]["start_time"] == "0.5"
    assert events[0]["end_time"] == "2.5"


def test_overlap_percentage_from_the_config_decides_who_is_counted(tmp_path):
    # Half of this box is inside the 100x100 zone.
    FakeDetector.boxes = [(50, 10, 150, 50, 0.9)]
    counts = {}
    for percent in (40, 60):
        run_dir = tmp_path / str(percent)
        config = make_config(min_overlap_percent=percent)
        with Pipeline(config, fps=10, run_dir=run_dir) as pipeline:
            counts[percent] = pipeline.process(make_frame(), 0).count
    assert counts == {40: 1, 60: 0}


class FakeDepth:
    """Stands in for DepthRunner: everything right of x=50 is closer."""

    def __init__(self):
        self.submitted = []
        self.closed = False

    def submit(self, frame):
        self.submitted.append(frame.copy())

    def latest(self):
        depth = np.full((120, 160), 0.2, dtype=np.float32)
        depth[:, 50:] = 0.9
        return depth

    def close(self):
        self.closed = True


def test_without_depth_there_is_no_depth_frame_or_closest_person(tmp_path):
    with Pipeline(make_config(), fps=10, run_dir=tmp_path) as pipeline:
        result = pipeline.process(make_frame(), 0)
    assert result.depth_frame is None
    assert result.closest is None


def depth_config(confirm_frames):
    config = make_config()
    config["depth"] = {"closest_confirm_frames": confirm_frames}
    return config


def test_the_closest_person_is_found_from_the_depth_map(tmp_path):
    config = depth_config(confirm_frames=1)
    with Pipeline(config, fps=10, run_dir=tmp_path, depth=FakeDepth()) as pipeline:
        result = pipeline.process(make_frame(), 0)
    # DEFAULT_BOXES: the second person (x 60-70) stands in the closer part.
    assert result.closest == 1
    assert result.depth_frame.shape == (120, 160, 3)


def test_the_closest_person_needs_five_frames_by_default(tmp_path):
    with Pipeline(
        make_config(), fps=10, run_dir=tmp_path, depth=FakeDepth()
    ) as pipeline:
        closest = [pipeline.process(make_frame(), index).closest for index in range(6)]
    assert closest == [None, None, None, None, 1, 1]


def test_the_confirmation_frames_come_from_the_config(tmp_path):
    config = depth_config(confirm_frames=3)
    with Pipeline(config, fps=10, run_dir=tmp_path, depth=FakeDepth()) as pipeline:
        closest = [pipeline.process(make_frame(), index).closest for index in range(4)]
    assert closest == [None, None, 1, 1]


def test_the_depth_model_receives_the_clean_frame(tmp_path):
    depth = FakeDepth()
    with Pipeline(make_config(), fps=10, run_dir=tmp_path, depth=depth) as pipeline:
        pipeline.process(make_frame(), 0)
    assert depth.submitted[0].max() == 0


def test_closing_the_pipeline_closes_the_depth_runner(tmp_path):
    depth = FakeDepth()
    with Pipeline(make_config(), fps=10, run_dir=tmp_path, depth=depth):
        pass
    assert depth.closed


def test_depth_is_off_unless_the_config_enables_it(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(
        "src.pipeline.DepthEstimator", lambda *args: built.append(args) or FakeDepth()
    )
    config = make_config()
    config["depth"] = {"enabled": False}
    with Pipeline(config, fps=10, run_dir=tmp_path) as pipeline:
        assert pipeline.depth is None
    assert built == []


def test_the_config_can_enable_depth_with_its_settings(tmp_path, monkeypatch):
    built = []

    class FakeEstimator:
        def __init__(self, *args):
            built.append(args)

        def estimate(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

    monkeypatch.setattr("src.pipeline.DepthEstimator", FakeEstimator)
    config = make_config()
    config["depth"] = {"enabled": True, "model": "some/model", "input_size": 140}
    with Pipeline(config, fps=10, run_dir=tmp_path) as pipeline:
        assert pipeline.depth is not None
    assert built == [("some/model", 140)]
