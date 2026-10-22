from datetime import datetime
from pathlib import Path


def make_run_dir(output_base, source):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = "webcam" if isinstance(source, int) else Path(source).stem
    return Path(output_base) / f"{name}_{timestamp}"
