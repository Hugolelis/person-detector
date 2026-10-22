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


def make_config(stride=1, processing_width=None):
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
