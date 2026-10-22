import argparse
import sys

from src.calibration import calibrate_roi
from src.config_loader import load_config, save_config
from src.errors.base import PersonDetectorError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw the red zone on the first video frame and save it."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to the YAML config."
    )
    return parser.parse_args()


def ask_person_limit(current):
    while True:
        answer = input(f"People limit [{current}]: ").strip()
        if not answer:
            return current
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("Please enter a positive whole number.")


def main():
    args = parse_args()
    try:
        config = load_config(args.config)
        points = calibrate_roi(args.video)
        if points is None:
            print("Calibration cancelled. Config not changed.")
            return

        config["roi"] = [list(point) for point in points]
        config["person_limit"] = ask_person_limit(config["person_limit"])
        save_config(args.config, config)
        print(f"Saved {len(points)} points to {args.config}")
    except PersonDetectorError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()