# Person Detector

> **Person Detector** is a computer vision pipeline that counts people inside a user-defined red zone of a recorded video, turns noisy per-frame counts into stable violation events, and saves every violation as a clip with the detections drawn on it.

<div align="left">

[![License](https://img.shields.io/badge/License-MIT-1a1a2e?style=for-the-badge&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-1a1a2e?style=for-the-badge&logo=python&logoColor=white)]()
[![Version](https://img.shields.io/badge/Version-0.1.0-1a1a2e?style=for-the-badge&logoColor=white)]()
[![Status](https://img.shields.io/badge/Status-In%20Development-1a1a2e?style=for-the-badge&logoColor=white)]()

</div>

<details>
<summary><strong>Table of Contents</strong></summary>

- [About](#about)
- [Demo](#demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Commands](#commands)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Development](#development)
- [Performance & Troubleshooting](#performance--troubleshooting)
- [License](#license)
- [Author](#author)

</details>

---

## About

Counting people in a video is easy; deciding when a zone is *actually* over its limit is not. A detector flickers: it misses someone for a frame or sees a ghost for one, so a naive "count above the limit" check fires dozens of false alerts on a single event. This project puts a debounce state machine between the detector and the alerts, counts a person only when their feet are inside the zone (so a zone drawn on the ground behaves like one), and keeps a rolling buffer so each clip shows what happened *before* the alert, not just after. Everything runs on the CPU, with no GPU required.

---

## Demo

```bash
poetry run python main.py --video videos/teste.mp4
```

```
frame 0: 4 people in zone
frame 50: 3 people in zone
frame 100: 4 people in zone
frame 150: 6 people in zone
frame 200: 6 people in zone
frame 250: 6 people in zone
frame 300: 4 people in zone
frame 350: 4 people in zone
```

```
outputs/teste_2026-10-22_08-04-33/
├── charts/occupancy.png
├── clips/violation_001.mp4
├── clips/violation_002.mp4
├── csv/events.csv
└── csv/series.csv
```

The same pipeline in the desktop window: live video with detections, live occupancy chart, and a panel to draw the zone and set the limit.

![Window](docs/window.png)

---

## Features

| Capability | Description |
|---|---|
| **Zone-based counting** | A person counts when the bottom-center of their bounding box (roughly the feet) is inside the zone polygon |
| **Debounce** | A violation starts after N consecutive frames above the limit, ends after M consecutive frames at or below it, with a cooldown between events |
| **Clips with context** | Each violation is saved with seconds of video from before the event (circular buffer) and after it, with the overlay drawn on |
| **Logs** | `series.csv` with the count of every frame and `events.csv` with the start, end, peak and clip of every violation |
| **Occupancy chart** | Bars with time on the horizontal axis and people on the vertical one, red above the limit, with the limit line |
| **Desktop window** | Live video, live chart, and a zone and limit editor in a single screen (PySide6) |
| **Speed readout** | The window shows the fps of the source video and the real processing speed (current and average, also as a multiple of real time); the CLI prints the average at the end |
| **Calibration** | Draw the zone by clicking on a frame, from the window or from the command line |
| **Configurable** | Model, confidence, zone, limit, debounce and clip length live in one YAML file |
| **CPU only** | Runs with the CPU build of PyTorch |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [Python](https://python.org) 3.12 | Core language |
| [Ultralytics YOLO](https://docs.ultralytics.com) | Person detection |
| [PyTorch](https://pytorch.org) (CPU build) | Inference backend |
| [OpenCV](https://opencv.org) | Video reading, drawing, clip writing |
| [PySide6](https://doc.qt.io/qtforpython-6/) | Desktop window |
| [Matplotlib](https://matplotlib.org) | Occupancy chart (PNG and live) |
| [NumPy](https://numpy.org) | Array operations |
| [PyYAML](https://pyyaml.org) | Configuration file |
| [Poetry](https://python-poetry.org) | Dependency and environment management |
| [pytest](https://pytest.org) and [Ruff](https://docs.astral.sh/ruff/) | Tests, linting and formatting |

---

## Architecture & Design Decisions

![Usage flow](diagram/PersonDetectorUsageFlow.png)

```
.
├── main.py                     CLI entry point
├── gui.py                      desktop window entry point
├── calibrate.py                command-line zone calibration
├── config/
│   └── config.yaml
├── src/
│   ├── pipeline.py             per-frame processing, shared by the CLI and the window
│   ├── person_detector.py      YOLO wrapper
│   ├── redzone.py              zone polygon and people counting
│   ├── violation.py            debounce state machine
│   ├── clip_recorder.py        circular buffer and clip writing
│   ├── metrics_logger.py       CSV logging
│   ├── overlay.py              drawing on frames
│   ├── chart.py                occupancy chart
│   ├── config_loader.py        YAML loading and saving
│   ├── run_dir.py              output folder naming
│   ├── errors/                 project exceptions
│   ├── enums/                  shared enums
│   └── gui/                    window, worker thread, live chart, video label
├── scripts/
│   └── tune_parameters.py      offline study of confidence and debounce
├── tests/
├── docs/                       README assets
└── diagram/                    usage flow
```

**Why this shape:** `Pipeline.process` handles exactly one frame and returns the result, so the CLI loop and the window's worker thread share the same logic instead of duplicating it. The debounce lives in its own class that only sees integer counts, which makes it testable with made-up sequences and no model. No class reads the YAML on its own: the entry points load the config and hand plain values down, so everything can be tested without files. Errors derive from one base exception, so the entry points catch every expected failure in a single `except` and let real bugs surface with a full traceback. The feet rule exists because the zone represents an area on the ground, and a box center would count someone standing in front of it.

**Known limitations:**
- YOLO on a CPU is the bottleneck: expect a few frames per second on 4K video, not real time.
- Recorded video only, with no live camera input.
- A single zone polygon per configuration.
- Clips are written with the `mp4v` codec, because that is what the pip build of OpenCV can encode; some viewers cannot play it.
- Only the top-level keys of the config are validated, so a missing nested key surfaces as a `KeyError`.
- The default parameters were checked against a single 15-second video.
- Developed and tested on Linux only.

---

## Getting Started

### Prerequisites

- Linux (developed on Ubuntu with GNOME and Wayland)
- Python 3.12
- [Poetry](https://python-poetry.org/)

No GPU is needed: the project installs the CPU build of PyTorch from the official PyTorch index.

### Installation

```bash
git clone https://github.com/Hugolelis/person_detector-ai.git
cd person_detector-ai
poetry env use python3.12
poetry install
```

The first run downloads the YOLO weights (`yolov8n.pt`, about 6 MB) into the project folder. They are git-ignored.

### Environment Variables

The project does not use environment variables. Everything is configured in `config/config.yaml`.

### Running locally

```bash
poetry run python gui.py --video videos/your_video.mp4
```

---

## Usage

### 1. Define the zone and the limit

Draw the zone on the first frame of your video, either in the window (**Edit zone**, then **Save zone and limit**) or from the terminal:

```bash
poetry run python calibrate.py --video videos/your_video.mp4
```

Left click adds a point, right click or `u` undoes the last one, `Enter` confirms and `Esc` cancels. The tool then asks for the person limit and writes everything to `config/config.yaml`.

### 2. Process a video from the command line

```bash
poetry run python main.py --video videos/your_video.mp4
```

### 3. Or process it in the window

```bash
poetry run python gui.py --video videos/your_video.mp4
```

Press **Start**. The video plays with the detections, the chart grows below it, and **Pause** freezes and resumes the processing. The side panel shows the fps of the source video and how fast it is actually being processed, in frames per second and as a multiple of real time. While **Edit zone** is active, left click on the video adds a point and right click removes the last one.

---

## Commands

| Command | Description |
|---|---|
| `python main.py --video PATH` | Process a video and write clips, CSVs and the chart. Optional: `--config` (default `config/config.yaml`), `--output` (default `outputs`) |
| `python gui.py --video PATH` | Open the window with live video, live chart and the zone editor. Same optional flags |
| `python calibrate.py --video PATH` | Draw the zone on the first frame and save it, asking for the limit. Optional: `--config` |
| `python -m scripts.tune_parameters --video PATH` | Evaluate confidence thresholds and debounce settings offline |

---

## Configuration

`config/config.yaml`:

| Key | Meaning |
|---|---|
| `detection.model` | YOLO weights (`yolov8n.pt` is the small, CPU-friendly one) |
| `detection.confidence_threshold` | Minimum confidence to keep a detection |
| `detection.threads` | PyTorch CPU threads. Use about the number of physical cores |
| `detection.stride` | *Optional.* Run YOLO every N frames and reuse the last boxes in between (default `1`). Counting and debounce still run on every frame |
| `roi` | Zone polygon, a list of `[x, y]` points in the pixels of the original frame |
| `person_limit` | A violation needs a count strictly **above** this value |
| `violation.frames_to_confirm_start` | Consecutive frames above the limit to start a violation |
| `violation.frames_to_confirm_end` | Consecutive frames at or below the limit to end it |
| `violation.cooldown_frames` | Frames ignored after a violation ends |
| `clip.seconds_before` / `clip.seconds_after` | Context kept around a violation |

---

## Output Structure

Each run writes to its own folder, so nothing is overwritten:

```
outputs/
└── your_video_2026-10-22_08-04-33/
    ├── clips/violation_001.mp4     one clip per violation, with the overlay drawn
    ├── csv/series.csv              frame, timestamp, count, in_violation
    ├── csv/events.csv              start and end (frame and time), peak count, clip path
    └── charts/occupancy.png        occupancy over time
```

In the chart, each bar is the median count over a time interval (0.5 s for short videos, growing with the duration so there are about 30 bars at most). Bars above the limit are red, and the dashed line marks the limit.

---

## Development

```bash
poetry run pytest          # tests
poetry run ruff check .    # lint
poetry run ruff format .   # format
```

The tests cover what does not need the model: zone counting, the debounce state machine, the chart bars and axes, the mapping of window clicks to frame pixels, and the pipeline using a fake detector.

`scripts/tune_parameters.py` runs the detector once, caches the result, and evaluates confidence thresholds and debounce settings offline. A larger model runs on a sample of frames and serves as the reference for the count error.

---

## Performance & Troubleshooting

**The video is slow.** YOLO on a CPU is the bottleneck, and a few settings matter a lot:
- Set the OS power mode to **Performance** (`powerprofilesctl set performance`). In power-saver mode the CPU can stay near a third of its maximum speed.
- Keep `detection.threads` close to the number of **physical** cores. Using every logical core while other programs run can make inference several times slower.
- Use `detection.stride: 3` (or similar) for a smoother window, at the cost of boxes that lag a couple of frames behind.
- Close heavy applications, such as browsers, while processing.

**The window crashes with `GLib-GIO-ERROR ... does not contain a key named 'antialiasing'`.** This happens when launching from the terminal of a **snap-installed VS Code**: the snap sets `GSETTINGS_SCHEMA_DIR` and `XDG_DATA_DIRS` to its own folders, and GTK then reads an outdated schema. Run from a regular terminal, or restore the original values for the command:

```bash
env -u GSETTINGS_SCHEMA_DIR XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG" \
    poetry run python gui.py --video videos/your_video.mp4
```

To fix it for the VS Code terminal permanently, add this to the VS Code user settings:

```json
"terminal.integrated.env.linux": {
    "GSETTINGS_SCHEMA_DIR": null,
    "XDG_DATA_DIRS": "/usr/local/share/:/usr/share/"
}
```

**Qt prints warnings about fonts or Wayland.** The lines mentioning `QFontDatabase`, `propagateSizeHints` or `textinput` are harmless.

**A clip does not play in the browser or in the file preview.** It uses the `mp4v` codec. Open it with VLC, or convert it: `ffmpeg -i clip.mp4 -c:v libx264 -pix_fmt yuv420p clip_h264.mp4`.

**`Could not open video file`.** The path is wrong, or the format is not supported by the FFmpeg backend of OpenCV.

---

## License

Distributed under the **MIT License**.

---

## Author

[Hugolelis](https://github.com/Hugolelis)
