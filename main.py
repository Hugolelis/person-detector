import argparse
from datetime import datetime
from pathlib import Path

from src.config_loader import load_config
from src.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect people in a red zone and record violations."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to the YAML config."
    )
    parser.add_argument(
        "--output", default="outputs", help="Base folder for generated files."
    )
    return parser.parse_args()


def make_run_dir(output_base, video_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(output_base) / f"{Path(video_path).stem}_{timestamp}"


def main():
    args = parse_args()
    config = load_config(args.config)
    run_dir = make_run_dir(args.output, args.video)
    run_pipeline(args.video, config, run_dir)


if __name__ == "__main__":
    main()