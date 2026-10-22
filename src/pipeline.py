from pathlib import Path

from src.chart import plot_occupancy
from src.clip_recorder import ClipRecorder
from src.enums.violation_event import ViolationEvent
from src.metrics_logger import MetricsLogger
from src.overlay import draw_overlay
from src.person_detector import PersonDetector
from src.redzone import RedZone
from src.video_reader import VideoReader
from src.violation import ViolationDetector


def run_pipeline(video_path, config, run_dir):
    run_dir = Path(run_dir)
    limit = config["person_limit"]
    violation = config["violation"]
    clip = config["clip"]
    detection = config["detection"]

    zone = RedZone(config["roi"])
    detector = PersonDetector(detection["model"], detection["confidence_threshold"])
    violation_detector = ViolationDetector(
        person_limit=limit,
        frames_to_confirm_start=violation["frames_to_confirm_start"],
        frames_to_confirm_end=violation["frames_to_confirm_end"],
        cooldown_frames=violation["cooldown_frames"],
    )

    with (
        VideoReader(video_path) as reader,
        MetricsLogger(run_dir / "csv", reader.fps) as logger,
    ):
        recorder = ClipRecorder(
            reader.fps, clip["seconds_before"], clip["seconds_after"]
        )
        violation_count = 0

        for frame_index, frame in enumerate(reader.frames()):
            detections = detector.detect(frame)
            count = zone.count_people(detections)
            event = violation_detector.update(count)
            in_violation = violation_detector.in_violation

            annotated = draw_overlay(
                    frame, detections, zone, count, limit, in_violation
                )
            recorder.add_frame(annotated)

            if event is ViolationEvent.STARTED:
                violation_count += 1
                clip_path = run_dir / "clips" / f"violation_{violation_count:03d}.mp4"
                recorder.start(clip_path)
                logger.start_violation(frame_index, clip_path)

            logger.log_frame(frame_index, count, in_violation)

            if event is ViolationEvent.ENDED:
                logger.end_violation(frame_index)

            if frame_index % 50 == 0:
                print(f"frame {frame_index}: {count} people in zone")

    plot_occupancy(
        run_dir / "csv" / "series.csv",
        run_dir / "charts" / "occupancy.png",
        limit,
    )