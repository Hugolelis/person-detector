import csv

import numpy as np
import pytest
from src.enums.violation_event import ViolationEvent
from src.pipeline import Pipeline


class FakeDetector:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = 0
        FakeDetector.instances.append(self)

    def detect(self, frame):
        self.calls += 1
        return [(10, 10, 20, 40, 0.9), (60, 10, 70, 40, 0.8)]


def make_config(stride=1):
    detection = {"model": "fake.pt", "confidence_threshold": 0.4}
    if stride != 1:
        detection["stride"] = stride
    return {
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


@pytest.fixture(autouse=True)
def fake_detector(monkeypatch):
    FakeDetector.instances = []
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
