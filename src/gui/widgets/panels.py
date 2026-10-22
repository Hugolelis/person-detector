from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class SourcePanel(QGroupBox):
    """Where the video comes from: a file or the webcam."""

    def __init__(self):
        super().__init__("Video")
        self.open_button = QPushButton("Open video...")
        self.webcam_button = QPushButton("Use webcam")
        self.name_label = QLabel("No video selected")
        self.name_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.open_button)
        layout.addWidget(self.webcam_button)
        layout.addWidget(self.name_label)

    def update_state(self, running):
        self.open_button.setEnabled(not running)
        self.webcam_button.setEnabled(not running)


class RunPanel(QGroupBox):
    """Starts, pauses and stops the processing."""

    def __init__(self, depth_enabled):
        super().__init__("Processing")
        self.depth_check = QCheckBox("Depth view (closest person)")
        self.depth_check.setChecked(depth_enabled)
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")

        layout = QVBoxLayout(self)
        layout.addWidget(self.depth_check)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)

    def update_state(self, running, has_source):
        self.depth_check.setEnabled(not running)
        self.start_button.setEnabled(has_source and not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)


class ZonePanel(QGroupBox):
    """The person limit and the zone editor."""

    def __init__(self, person_limit):
        super().__init__("Zone and limit")
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, 99)
        self.limit_spin.setValue(person_limit)
        self.edit_button = QPushButton("Edit zone")
        self.edit_button.setCheckable(True)
        self.undo_button = QPushButton("Undo last point")
        self.save_button = QPushButton("Save zone and limit")

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Person limit"))
        limit_row.addWidget(self.limit_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(limit_row)
        layout.addWidget(self.edit_button)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.save_button)

    def update_state(self, running, has_source, editing):
        self.limit_spin.setEnabled(not running)
        self.edit_button.setEnabled(has_source and not running)
        self.undo_button.setEnabled(editing and not running)
        self.save_button.setEnabled(has_source and not running)
