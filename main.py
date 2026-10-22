import argparse
import sys

from src.config_loader import load_config
from src.errors.base import PersonDetectorError
from src.pipeline import run_pipeline
from src.reporting.run_dir import make_run_dir


def parse_args(video_required=True):
    parser = argparse.ArgumentParser(
        description="Detect people in a red zone and record violations."
    )
    parser.add_argument(
        "--video", required=video_required, help="Path to the input video."
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to the YAML config."
    )
    parser.add_argument(
        "--output", default="outputs", help="Base folder for generated files."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = load_config(args.config)
        run_dir = make_run_dir(args.output, args.video)
        run_pipeline(args.video, config, run_dir)
    except PersonDetectorError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
