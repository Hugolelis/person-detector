"""Offline study of the detection and debounce parameters.

The small model is run once on every frame at a low confidence and the result is
cached, so every confidence threshold and debounce setting can be evaluated
without running the network again. A larger model, run on a sample of frames, is
used as a reference to estimate how far each confidence threshold is from the
"true" count.

Run from the project root:
    poetry run python -m scripts.tune_parameters --video videos/teste.mp4
"""

import argparse
import json
from pathlib import Path

from src.config_loader import load_config
from src.enums.violation_event import ViolationEvent
from src.person_detector import PersonDetector
from src.redzone import RedZone
from src.video_reader import VideoReader
from src.violation import ViolationDetector

CONFIDENCES = (0.25, 0.3, 0.4, 0.5, 0.6)
DEBOUNCE_FRAMES = (3, 5, 10, 20)
COLLECTION_CONFIDENCE = 0.15
REFERENCE_CONFIDENCE = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--reference-model", default="yolov8m.pt")
    parser.add_argument("--sample-step", type=int, default=3)
    parser.add_argument("--cache", default="/tmp/tune_parameters_cache.json")
    return parser.parse_args()


def collect(video, model, confidence, only_every=1):
    detector = PersonDetector(model, confidence)
    detections = {}
    with VideoReader(video) as reader:
        fps = reader.fps
        for index, frame in enumerate(reader.frames()):
            if index % only_every == 0:
                detections[index] = detector.detect(frame)
                if len(detections) % 20 == 0:
                    print(f"  {model}: {len(detections)} frames", flush=True)
    return fps, detections


def load_or_collect(args):
    small_cache = Path(f"{args.cache}.small.json")
    reference_cache = Path(f"{args.cache}.reference.json")

    if small_cache.exists():
        print(f"Using cached detections from {small_cache}")
        small_data = json.loads(small_cache.read_text())
    else:
        print(f"Running {args.model} on every frame...", flush=True)
        fps, small = collect(args.video, args.model, COLLECTION_CONFIDENCE)
        small_data = {"fps": fps, "detections": small}
        small_cache.write_text(json.dumps(small_data))

    if reference_cache.exists():
        print(f"Using cached detections from {reference_cache}")
        reference_data = json.loads(reference_cache.read_text())
    else:
        print(
            f"Running {args.reference_model} every {args.sample_step} frames...",
            flush=True,
        )
        _, reference = collect(
            args.video, args.reference_model, REFERENCE_CONFIDENCE, args.sample_step
        )
        reference_data = {"detections": reference}
        reference_cache.write_text(json.dumps(reference_data))

    def with_int_keys(detections):
        return {int(k): [tuple(box) for box in v] for k, v in detections.items()}

    return (
        small_data["fps"],
        with_int_keys(small_data["detections"]),
        with_int_keys(reference_data["detections"]),
    )


def counts(zone, detections_by_frame, min_confidence, frames):
    return [
        zone.count_people([d for d in detections_by_frame[i] if d[4] >= min_confidence])
        for i in frames
    ]


def violation_summary(series, limit, frames_to_confirm, cooldown):
    detector = ViolationDetector(limit, frames_to_confirm, frames_to_confirm, cooldown)
    started = 0
    frames_in_violation = 0
    for count in series:
        if detector.update(count) is ViolationEvent.STARTED:
            started += 1
        frames_in_violation += detector.in_violation
    return started, frames_in_violation


def main():
    args = parse_args()
    config = load_config(args.config)
    zone = RedZone(config["roi"])
    limit = config["person_limit"]
    violation = config["violation"]
    fps, small, reference = load_or_collect(args)

    all_frames = sorted(small)
    sampled = sorted(reference)
    reference_counts = counts(zone, reference, REFERENCE_CONFIDENCE, sampled)

    print(f"\nVideo: {args.video} | {len(all_frames)} frames @ {fps:.1f} fps")
    print(
        f"Zone limit: {limit} | reference: {args.reference_model} "
        f"on {len(sampled)} frames"
    )
    print("\n== Confidence threshold (counts inside the zone) ==")
    print("conf   mean   jitter  MAE_vs_ref  bias_vs_ref   (bias > 0: over-counts)")
    for confidence in CONFIDENCES:
        series = counts(zone, small, confidence, all_frames)
        jitter = sum(a != b for a, b in zip(series, series[1:], strict=False))
        sampled_series = counts(zone, small, confidence, sampled)
        errors = [s - r for s, r in zip(sampled_series, reference_counts, strict=True)]
        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        mean = sum(series) / len(series)
        print(f"{confidence:<5} {mean:6.2f} {jitter:7d} {mae:11.2f} {bias:12.2f}")

    chosen = config["detection"]["confidence_threshold"]
    series = counts(zone, small, chosen, all_frames)
    print(f"\n== Debounce (using the configured confidence {chosen}, limit {limit}) ==")
    print("frames_to_confirm  violations  frames_in_violation")
    for frames_to_confirm in DEBOUNCE_FRAMES:
        started, total = violation_summary(
            series, limit, frames_to_confirm, violation["cooldown_frames"]
        )
        print(f"{frames_to_confirm:<17} {started:11d} {total:20d}")
    print(
        f"(configured: confirm start {violation['frames_to_confirm_start']}, "
        f"end {violation['frames_to_confirm_end']}, "
        f"cooldown {violation['cooldown_frames']})"
    )


if __name__ == "__main__":
    main()
