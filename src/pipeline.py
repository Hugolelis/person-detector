from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
from src.chart import plot_occupancy
from src.clip_recorder import ClipRecorder
from src.enums.violation_event import ViolationEvent
from src.metrics_logger import MetricsLogger
from src.overlay import draw_overlay
from src.person_detector import DEFAULT_THREADS, PersonDetector
from src.redzone import RedZone
from src.video_reader import VideoReader
from src.violation import ViolationDetector

DEFAULT_WORKING_WIDTH = 1280


@dataclass
class FrameResult:
    detections: list
    count: int
    event: ViolationEvent | None
    in_violation: bool
    annotated_frame: object


class Pipeline:
    def __init__(self, config, fps, run_dir, frame_width=None):
        self.run_dir = Path(run_dir)
        self.limit = config["person_limit"]
        violation = config["violation"]
        clip = config["clip"]
        detection = config["detection"]

        working_width = config.get("processing", {}).get("width", DEFAULT_WORKING_WIDTH)
        self.scale = 1.0
        self.working_width = None
        if frame_width and working_width and frame_width > working_width:
            self.scale = working_width / frame_width
            self.working_width = working_width

        roi = [[round(x * self.scale), round(y * self.scale)] for x, y in config["roi"]]
        self.zone = RedZone(roi)
        self.detector = PersonDetector(
            detection["model"],
            detection["confidence_threshold"],
            detection.get("threads", DEFAULT_THREADS),
        )
        self.detection_stride = max(1, detection.get("stride", 1))
        self.last_detections = []
        self.violation_detector = ViolationDetector(
            person_limit=self.limit,
            frames_to_confirm_start=violation["frames_to_confirm_start"],
            frames_to_confirm_end=violation["frames_to_confirm_end"],
            cooldown_frames=violation["cooldown_frames"],
        )
        self.logger = MetricsLogger(self.run_dir / "csv", fps)
        recorder_width = self.working_width or frame_width
        self.recorder = ClipRecorder(
            fps,
            clip["seconds_before"],
            clip["seconds_after"],
            **({"width": recorder_width} if recorder_width else {}),
        )
        self.violation_count = 0

    def process(self, frame, frame_index):
        if self.working_width:
            size = (self.working_width, round(frame.shape[0] * self.scale))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        if frame_index % self.detection_stride == 0:
            self.last_detections = self.detector.detect(frame)
        detections = self.last_detections
        count = self.zone.count_people(detections)
        event = self.violation_detector.update(count)
        in_violation = self.violation_detector.in_violation

        annotated = draw_overlay(
            frame, detections, self.zone, count, self.limit, in_violation
        )
        self.recorder.add_frame(annotated)

        if event is ViolationEvent.STARTED:
            self.violation_count += 1
            clip_path = (
                self.run_dir / "clips" / f"violation_{self.violation_count:03d}.mp4"
            )
            self.recorder.start(clip_path)
            self.logger.start_violation(frame_index, clip_path)

        self.logger.log_frame(frame_index, count, in_violation)

        if event is ViolationEvent.ENDED:
            self.logger.end_violation(frame_index)

        return FrameResult(detections, count, event, in_violation, annotated)

    def plot_chart(self):
        plot_occupancy(
            self.run_dir / "csv" / "series.csv",
            self.run_dir / "charts" / "occupancy.png",
            self.limit,
        )

    def close(self):
        self.recorder.close()
        self.logger.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def run_pipeline(video_path, config, run_dir):
    with (
        VideoReader(video_path) as reader,
        Pipeline(config, reader.fps, run_dir, frame_width=reader.width) as pipeline,
    ):
        started = perf_counter()
        frames = 0
        for frame_index, frame in enumerate(reader.frames()):
            result = pipeline.process(frame, frame_index)
            frames += 1
            if frame_index % 50 == 0:
                print(f"frame {frame_index}: {result.count} people in zone")
        elapsed = perf_counter() - started

    pipeline.plot_chart()
    if frames:
        print(
            f"Processed {frames} frames in {elapsed:.1f} s ({frames / elapsed:.1f} fps)"
        )
