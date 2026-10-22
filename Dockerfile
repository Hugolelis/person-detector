FROM python:3.12-slim

# System libraries: OpenCV (libgl1, libglib2.0-0) and Qt/PySide6 over X11.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libice6 \
        libegl1 \
        libdbus-1-3 \
        libfontconfig1 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN pip install poetry

WORKDIR /app

# Dependencies first: this layer is only rebuilt when the lock file changes.
COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

# YOLO weights are baked in so the container works offline.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Writable by any user, so the container can run as the host user
# and a named volume mounted here inherits these permissions.
RUN mkdir -p /cache/huggingface /tmp/cache && chmod -R 777 /cache /tmp/cache

COPY main.py gui.py calibrate.py ./
COPY src ./src
COPY scripts ./scripts
# Default config; the compose file mounts ./config over it so edits persist.
COPY config ./config

ENTRYPOINT ["python"]
CMD ["main.py", "--help"]
