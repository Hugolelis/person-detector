from datetime import datetime
from pathlib import Path


def make_run_dir(output_base, video_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(output_base) / f"{Path(video_path).stem}_{timestamp}"
