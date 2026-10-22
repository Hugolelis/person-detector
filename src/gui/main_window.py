from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from src.config_loader import save_config
from src.errors.base import PersonDetectorError
from src.gui.source_preview import (
    WEBCAM_INDEX,
    ask_video_path,
    describe_source,
    load_preview,
)
from src.gui.video_worker import VideoWorker, shrink_for_display
from src.gui.widgets.live_chart import LiveChart
from src.gui.widgets.panels import RunPanel, SourcePanel, ZonePanel
from src.gui.widgets.speed_readout import SpeedReadout
from src.gui.widgets.video_label import VideoLabel
from src.gui.zone_editor import ZoneEditor
from src.reporting.run_dir import make_run_dir
from src.video.overlay import draw_zone_preview

DEPTH_PLACEHOLDER = "The depth view starts with the processing."


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
        self.zone_editor = ZoneEditor()

        self.setWindowTitle("Redzone Monitor")
        self.resize(1320, 900)
        self._build_ui()
        self._connect_signals()
        if source is not None:
            self.load_source(source)
        self._update_controls()

    def _build_ui(self):
        self.video_label = VideoLabel("Open a video to start.")
        self.video_label.setMinimumSize(480, 270)
        self.depth_label = VideoLabel(DEPTH_PLACEHOLDER)
        self.depth_label.setMinimumSize(480, 270)

        self.source_panel = SourcePanel()
        self.run_panel = RunPanel(
            bool(self.config.get("depth", {}).get("enabled", False))
        )
        self.zone_panel = ZonePanel(int(self.config["person_limit"]))
        self.fps_label = SpeedReadout()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        side_panel = QWidget()
        side_panel.setFixedWidth(260)
        side_layout = QVBoxLayout(side_panel)
        side_layout.addWidget(self.source_panel)
        side_layout.addWidget(self.run_panel)
        side_layout.addWidget(self.zone_panel)
        side_layout.addWidget(self.fps_label)
        side_layout.addWidget(self.status_label, stretch=1)

        top_row = QHBoxLayout()
        top_row.addWidget(self.video_label, stretch=1)
        top_row.addWidget(self.depth_label, stretch=1)
        top_row.addWidget(side_panel)
        self.depth_label.setVisible(self.run_panel.depth_check.isChecked())

        self.chart = LiveChart(int(self.config["person_limit"]))
        self.chart.setMinimumHeight(280)

        layout = QVBoxLayout()
        layout.addLayout(top_row, stretch=5)
        layout.addWidget(self.chart, stretch=3)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _connect_signals(self):
        self.video_label.clicked.connect(self.on_video_clicked)
        self.source_panel.open_button.clicked.connect(self.choose_video)
        self.source_panel.webcam_button.clicked.connect(self.use_webcam)
        self.run_panel.depth_check.toggled.connect(self.depth_label.setVisible)
        self.run_panel.start_button.clicked.connect(self.start_processing)
        self.run_panel.pause_button.clicked.connect(self.toggle_pause)
        self.run_panel.stop_button.clicked.connect(self.stop_processing)
        self.zone_panel.edit_button.toggled.connect(self.toggle_calibration)
        self.zone_panel.undo_button.clicked.connect(self.undo_point)
        self.zone_panel.save_button.clicked.connect(self.save_settings)

    # ---- source ----

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
        return ask_video_path(self, self.source)

    def load_source(self, source):
        """Show a frame of source. On failure the current source is kept."""
        try:
            preview = load_preview(source)
        except PersonDetectorError as e:
            self.status_label.setText(f"Error: {e}")
            if self.first_frame is None:
                self.video_label.setText(f"Error: {e}")
            return False

        if self.zone_editor.active:
            self.zone_panel.edit_button.setChecked(False)
        self.source = source
        self.is_live = preview.live
        self.first_frame = preview.first_frame
        self.source_panel.name_label.setText(describe_source(source, preview.live))
        self.status_label.setText("")
        self.fps_label.set_source(preview.live, preview.fps)
        self.clear_view()
        return True

    # ---- view ----

    def _show_zone_preview(self, points):
        if self.first_frame is None:
            return
        canvas = self.first_frame.copy()
        draw_zone_preview(canvas, [tuple(point) for point in points])
        self.video_label.show_frame(shrink_for_display(canvas))

    def clear_view(self):
        self.chart.reset()
        self.depth_label.clear()
        self.depth_label.setText(DEPTH_PLACEHOLDER)
        self._show_zone_preview(self.config["roi"])

    def _is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def _update_controls(self):
        running = self._is_running()
        has_source = self.first_frame is not None
        self.source_panel.update_state(running)
        self.run_panel.update_state(running, has_source)
        self.zone_panel.update_state(running, has_source, self.zone_editor.active)

    # ---- processing ----

    def _run_config(self):
        """The saved config plus this run's depth switch, which is never written."""
        depth = {
            **self.config.get("depth", {}),
            "enabled": self.run_panel.depth_check.isChecked(),
        }
        return {**self.config, "depth": depth}

    def start_processing(self):
        if self._is_running():
            return
        if self.zone_editor.active:
            self.zone_panel.edit_button.setChecked(False)
        run_dir = make_run_dir(self.output_base, self.source)
        self.chart.reset()
        self.chart.set_limit(int(self.config["person_limit"]))
        self.video_label.click_frame_size = None
        self.worker = VideoWorker(self.source, self._run_config(), run_dir)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.on_finished)
        self.run_panel.pause_button.setText("Pause")
        self.status_label.setText(f"Processing. Results in {run_dir}")
        self.fps_label.begin()
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
            self.fps_label.resume()
            self.run_panel.pause_button.setText("Pause")
        else:
            self.worker.pause()
            self.fps_label.pause()
            self.run_panel.pause_button.setText("Resume")

    def on_frame(self, update):
        self.video_label.show_frame(update.frame)
        if update.depth_frame is not None:
            self.depth_label.show_frame(update.depth_frame)
        self.chart.add_point(update.time_s, update.count)
        self.fps_label.tick()

    def show_error(self, message):
        self.status_label.setText(f"Error: {message}")

    def on_finished(self):
        self.run_panel.pause_button.setText("Pause")
        self.fps_label.finish()
        if not self.status_label.text().startswith("Error"):
            self.status_label.setText(f"Finished. Results in {self.worker.run_dir}")
        self.clear_view()
        self._update_controls()

    # ---- zone editing ----

    def toggle_calibration(self, checked):
        if checked:
            self.zone_editor.begin()
            self.video_label.click_frame_size = (
                self.first_frame.shape[1],
                self.first_frame.shape[0],
            )
            self.status_label.setText("Press Save when the zone has at least 3 points.")
            self._render_calibration()
        else:
            self.zone_editor.end()
            self.video_label.click_frame_size = None
            self._show_zone_preview(self.config["roi"])
        self._update_controls()

    def on_video_clicked(self, x, y, button):
        if not self.zone_editor.active or self._is_running():
            return
        self.zone_editor.click(x, y, button)
        self._render_calibration()

    def undo_point(self):
        if self.zone_editor.active and self.zone_editor.undo():
            self._render_calibration()

    def _render_calibration(self):
        self._show_zone_preview(self.zone_editor.points)

    def save_settings(self):
        editing = self.zone_editor.active
        if editing and not self.zone_editor.can_save:
            self.status_label.setText("A zone needs at least 3 points.")
            return
        if editing:
            self.config["roi"] = self.zone_editor.roi()
        self.config["person_limit"] = self.zone_panel.limit_spin.value()
        try:
            save_config(self.config_path, self.config)
        except PersonDetectorError as e:
            self.status_label.setText(f"Error: {e}")
            return
        self.chart.set_limit(self.config["person_limit"])
        self.status_label.setText(f"Saved to {self.config_path}")
        if editing:
            self.zone_panel.edit_button.setChecked(False)

    def closeEvent(self, event):
        if self._is_running():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)
