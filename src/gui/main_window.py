from pathlib import Path
from time import perf_counter

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from src.config_loader import save_config
from src.errors.base import PersonDetectorError
from src.errors.video import VideoOpenError
from src.gui.fps_meter import FpsMeter, describe_speed
from src.gui.live_chart import LiveChart
from src.gui.video_label import VideoLabel
from src.gui.video_worker import VideoWorker, shrink_for_display
from src.overlay import draw_zone_preview
from src.run_dir import make_run_dir
from src.video_reader import VideoReader

VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm);;All files (*)"
DEFAULT_VIDEO_DIR = "videos"
WEBCAM_INDEX = 0
WEBCAM_WARMUP_FRAMES = 5


class MainWindow(QMainWindow):
    def __init__(self, source, config, config_path, output_base):
        """source is a file path, a camera index, or None to pick it in the window."""
        super().__init__()
        self.source = None
        self.config = config
        self.config_path = config_path
        self.output_base = output_base
        self.worker = None
        self.first_frame = None
        self.is_live = False
        self.calibrating = False
        self.calibration_points = []
        self.video_fps = None
        self.fps_meter = FpsMeter()

        self.setWindowTitle("Redzone Monitor")
        self.resize(1320, 900)
        self._build_ui()
        if source is not None:
            self.load_source(source)
        self._update_controls()

    def _build_ui(self):
        self.video_label = VideoLabel("Open a video to start.")
        self.video_label.setMinimumSize(640, 360)
        self.video_label.clicked.connect(self.on_video_clicked)

        self.open_button = QPushButton("Open video...")
        self.open_button.clicked.connect(self.choose_video)
        self.webcam_button = QPushButton("Use webcam")
        self.webcam_button.clicked.connect(self.use_webcam)
        self.video_name_label = QLabel("No video selected")
        self.video_name_label.setWordWrap(True)

        video_box = QGroupBox("Video")
        video_layout = QVBoxLayout(video_box)
        video_layout.addWidget(self.open_button)
        video_layout.addWidget(self.webcam_button)
        video_layout.addWidget(self.video_name_label)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.start_processing)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_processing)

        run_box = QGroupBox("Processing")
        run_layout = QVBoxLayout(run_box)
        run_layout.addWidget(self.start_button)
        run_layout.addWidget(self.pause_button)
        run_layout.addWidget(self.stop_button)

        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, 99)
        self.limit_spin.setValue(int(self.config["person_limit"]))
        self.edit_button = QPushButton("Edit zone")
        self.edit_button.setCheckable(True)
        self.undo_button = QPushButton("Undo last point")
        self.save_button = QPushButton("Save zone and limit")
        self.edit_button.toggled.connect(self.toggle_calibration)
        self.undo_button.clicked.connect(self.undo_point)
        self.save_button.clicked.connect(self.save_settings)

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Person limit"))
        limit_row.addWidget(self.limit_spin)

        zone_box = QGroupBox("Zone and limit")
        zone_layout = QVBoxLayout(zone_box)
        zone_layout.addLayout(limit_row)
        zone_layout.addWidget(self.edit_button)
        zone_layout.addWidget(self.undo_button)
        zone_layout.addWidget(self.save_button)

        self.fps_label = QLabel("")
        self.fps_label.setWordWrap(True)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        side_panel = QWidget()
        side_panel.setFixedWidth(260)
        side_layout = QVBoxLayout(side_panel)
        side_layout.addWidget(video_box)
        side_layout.addWidget(run_box)
        side_layout.addWidget(zone_box)
        side_layout.addWidget(self.fps_label)
        side_layout.addWidget(self.status_label, stretch=1)

        top_row = QHBoxLayout()
        top_row.addWidget(self.video_label, stretch=1)
        top_row.addWidget(side_panel)

        self.chart = LiveChart(int(self.config["person_limit"]))
        self.chart.setMinimumHeight(280)

        layout = QVBoxLayout()
        layout.addLayout(top_row, stretch=5)
        layout.addWidget(self.chart, stretch=3)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def choose_video(self):
        if self._is_running():
            return
        path = self._ask_video_path()
        if path:
            self.load_source(path)
            self._update_controls()

    def use_webcam(self):
        if self._is_running():
            return
        self.load_source(WEBCAM_INDEX)
        self._update_controls()

    def _ask_video_path(self):
        start_dir = DEFAULT_VIDEO_DIR
        if isinstance(self.source, str):
            start_dir = str(Path(self.source).parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video", start_dir, VIDEO_FILTER
        )
        return path

    def _read_preview_frame(self, reader):
        """The first frame of a file, or a settled frame of a camera."""
        frames = reader.frames()
        frame = None
        for _ in range(WEBCAM_WARMUP_FRAMES if reader.is_live else 1):
            frame = next(frames, None)
            if frame is None:
                break
        return frame

    def load_source(self, source):
        """Show a frame of source. On failure the current source is kept."""
        try:
            with VideoReader(source) as reader:
                live = reader.is_live
                fps = reader.fps
                first_frame = self._read_preview_frame(reader)
            if first_frame is None:
                raise VideoOpenError(f"No frames could be read from: {source}")
        except PersonDetectorError as e:
            self.status_label.setText(f"Error: {e}")
            if self.first_frame is None:
                self.video_label.setText(f"Error: {e}")
            return False

        if self.calibrating:
            self.edit_button.setChecked(False)
        self.source = source
        self.is_live = live
        self.video_fps = None if live else fps
        self.first_frame = first_frame
        self.video_name_label.setText(
            f"Webcam (camera {source})" if live else Path(source).name
        )
        self.status_label.setText("")
        self._update_fps_label()
        self.clear_view()
        return True

    def _update_fps_label(self, processing=None):
        lines = []
        if self.is_live:
            lines.append("Source: webcam (live)")
        elif self.video_fps:
            lines.append(f"Source video: {self.video_fps:.2f} fps")
        if processing:
            lines.append(f"Processing: {processing}")
        self.fps_label.setText("\n".join(lines))

    def _show_zone_preview(self, points):
        if self.first_frame is None:
            return
        canvas = self.first_frame.copy()
        draw_zone_preview(canvas, [tuple(point) for point in points])
        self.show_frame(shrink_for_display(canvas))

    def _is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def _update_controls(self):
        running = self._is_running()
        has_video = self.first_frame is not None
        self.open_button.setEnabled(not running)
        self.webcam_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.start_button.setEnabled(has_video and not running)
        self.pause_button.setEnabled(running)
        self.limit_spin.setEnabled(not running)
        self.edit_button.setEnabled(has_video and not running)
        self.undo_button.setEnabled(self.calibrating and not running)
        self.save_button.setEnabled(has_video and not running)

    def start_processing(self):
        if self._is_running():
            return
        if self.calibrating:
            self.edit_button.setChecked(False)
        run_dir = make_run_dir(self.output_base, self.source)
        self.chart.reset()
        self.chart.set_limit(int(self.config["person_limit"]))
        self.video_label.click_frame_size = None
        self.worker = VideoWorker(self.source, self.config, run_dir)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.on_finished)
        self.pause_button.setText("Pause")
        self.status_label.setText(f"Processing. Results in {run_dir}")
        self.fps_meter.reset()
        self._update_fps_label("measuring...")
        self.worker.start()
        self._update_controls()

    def stop_processing(self):
        if self._is_running():
            self.worker.stop()

    def toggle_pause(self):
        if not self._is_running():
            return
        if self.worker.is_paused:
            self.worker.resume()
            self.fps_meter.resume(perf_counter())
            self.pause_button.setText("Pause")
            self._update_fps_label("measuring...")
        else:
            self.worker.pause()
            self.fps_meter.pause(perf_counter())
            self.pause_button.setText("Resume")
            self._update_fps_label("paused")

    def on_frame(self, update):
        self.show_frame(update.frame)
        self.chart.add_point(update.time_s, update.count)
        self.fps_meter.tick(perf_counter())
        fps = self.fps_meter.current_fps()
        self._update_fps_label(
            describe_speed(fps, self.video_fps) if fps else "measuring..."
        )

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def show_error(self, message):
        self.status_label.setText(f"Error: {message}")

    def on_finished(self):
        self.pause_button.setText("Pause")
        average = self.fps_meter.average_fps()
        if average:
            self._update_fps_label(
                f"finished, {describe_speed(average, self.video_fps)}"
            )
        if not self.status_label.text().startswith("Error"):
            self.status_label.setText(f"Finished. Results in {self.worker.run_dir}")
        self.clear_view()
        self._update_controls()

    def clear_view(self):
        self.chart.reset()
        self._show_zone_preview(self.config["roi"])

    def toggle_calibration(self, checked):
        self.calibrating = checked
        if checked:
            self.calibration_points = []
            self.video_label.click_frame_size = (
                self.first_frame.shape[1],
                self.first_frame.shape[0],
            )
            self.status_label.setText("Press Save when the zone has at least 3 points.")
            self._render_calibration()
        else:
            self.video_label.click_frame_size = None
            self._show_zone_preview(self.config["roi"])
        self._update_controls()

    def on_video_clicked(self, x, y, button):
        if not self.calibrating or self._is_running():
            return
        if button == Qt.MouseButton.LeftButton:
            self.calibration_points.append((x, y))
        elif button == Qt.MouseButton.RightButton and self.calibration_points:
            self.calibration_points.pop()
        self._render_calibration()

    def undo_point(self):
        if self.calibrating and self.calibration_points:
            self.calibration_points.pop()
            self._render_calibration()

    def _render_calibration(self):
        self._show_zone_preview(self.calibration_points)

    def save_settings(self):
        if self.calibrating and len(self.calibration_points) < 3:
            self.status_label.setText("A zone needs at least 3 points.")
            return
        if self.calibrating:
            self.config["roi"] = [list(point) for point in self.calibration_points]
        self.config["person_limit"] = self.limit_spin.value()
        try:
            save_config(self.config_path, self.config)
        except PersonDetectorError as e:
            self.status_label.setText(f"Error: {e}")
            return
        self.chart.set_limit(self.config["person_limit"])
        self.status_label.setText(f"Saved to {self.config_path}")
        if self.calibrating:
            self.edit_button.setChecked(False)

    def closeEvent(self, event):
        if self._is_running():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)
