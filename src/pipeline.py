from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
from src.depth.closest_tracker import DEFAULT_CONFIRM_FRAMES, ClosestTracker
from src.depth.estimator import (
    DEFAULT_DEPTH_MODEL,
    DEFAULT_INPUT_SIZE,
    DepthEstimator,
)
from src.depth.runner import DepthRunner
from src.depth.view import (
    colorize_depth,
    confirmed_first,
    draw_ranking,
    rank_by_closeness,
)
from src.detection.person_detector import DEFAULT_THREADS, PersonDetector
from src.detection.redzone import DEFAULT_MIN_OVERLAP_PERCENT, RedZone
from src.detection.violation import ViolationDetector
from src.enums.violation_event import ViolationEvent
from src.reporting.chart import plot_occupancy
from src.reporting.metrics_logger import MetricsLogger
from src.video.clip_recorder import ClipRecorder
from src.video.overlay import draw_overlay
from src.video.reader import VideoReader

DEFAULT_WORKING_WIDTH = 1280


@dataclass
class FrameResult:
    detections: list
    count: int
    event: ViolationEvent | None
    in_violation: bool
    annotated_frame: object
    depth_frame: object = None
    closest: int | None = None


class Pipeline:
    def __init__(self, config, fps, run_dir, frame_width=None, depth=None):
        self.run_dir = Path(run_dir)
        self.fps = fps
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
        counting = config.get("counting", {})
        self.zone = RedZone(
            roi, counting.get("min_overlap_percent", DEFAULT_MIN_OVERLAP_PERCENT)
        )
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
        depth_settings = config.get("depth", {})
        self.depth = depth or self._build_depth(depth_settings)
        self.closest_tracker = ClosestTracker(
            depth_settings.get("closest_confirm_frames", DEFAULT_CONFIRM_FRAMES)
        )
        self._depth_map = None
        self._depth_background = None

    @staticmethod
    def _build_depth(settings):
        if not settings.get("enabled", False):
            return None
        estimator = DepthEstimator(
            settings.get("model", DEFAULT_DEPTH_MODEL),
            settings.get("input_size", DEFAULT_INPUT_SIZE),
        )
        return DepthRunner(estimator, settings.get("stride", 10))

    def _rank_by_depth(self, frame, detections):
        """Submit the clean frame and rank people on the latest depth map, if any."""
        self.depth.submit(frame)
        depth_map = self.depth.latest()
        if depth_map is None:
            return None, []
        if depth_map is not self._depth_map:
            self._depth_map = depth_map
            self._depth_background = colorize_depth(depth_map)
        return depth_map, rank_by_closeness(depth_map, detections)

    def process(self, frame, frame_index, time_s=None):
        """Process one frame. time_s is its time in seconds in the stream.

        Without it the time is frame_index / fps, which is right for video files.
        Live sources pass the measured time because they run slower than their fps.
        """
        if time_s is None:
            time_s = frame_index / self.fps
        if self.working_width:
            size = (self.working_width, round(frame.shape[0] * self.scale))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        if frame_index % self.detection_stride == 0:
            self.last_detections = self.detector.detect(frame)
        detections = self.last_detections
        count = self.zone.count_people(detections)
        event = self.violation_detector.update(count)
        in_violation = self.violation_detector.in_violation

        closest = None
        depth_frame = None
        if self.depth:
            depth_map, ranking = self._rank_by_depth(frame, detections)
            if depth_map is not None:
                closest = self.closest_tracker.update(detections, ranking)
                depth_frame = draw_ranking(
                    self._depth_background.copy(),
                    detections,
                    confirmed_first(ranking, closest),
                    closest,
                )

        annotated = draw_overlay(
            frame, detections, self.zone, count, self.limit, in_violation, closest
        )
        self.recorder.add_frame(annotated, time_s)

        if event is ViolationEvent.STARTED:
            self.violation_count += 1
            clip_path = (
                self.run_dir / "clips" / f"violation_{self.violation_count:03d}.mp4"
            )
            self.recorder.start(clip_path)
            self.logger.start_violation(frame_index, clip_path, time_s)

        self.logger.log_frame(frame_index, count, in_violation, time_s)

        if event is ViolationEvent.ENDED:
            self.logger.end_violation(frame_index, time_s)

        return FrameResult(
            detections, count, event, in_violation, annotated, depth_frame, closest
        )

    def plot_chart(self):
        plot_occupancy(
            self.run_dir / "csv" / "series.csv",
            self.run_dir / "charts" / "occupancy.png",
            self.limit,
        )

    def close(self):
        if self.depth:
            self.depth.close()
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
