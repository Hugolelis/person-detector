import time

import cv2
import numpy as np
import pytest
import yaml
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from src.config_loader import save_config
from src.gui.live_chart import LiveChart
from src.gui.main_window import MainWindow

FRAMES = 24
FPS = 10


class FakeDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        return [(10, 10, 30, 80, 0.9), (60, 10, 80, 80, 0.8)]


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240)
    )
    for index in range(FRAMES):
        writer.write(np.full((240, 320, 3), index * 8, dtype=np.uint8))
    writer.release()
    return str(path)


@pytest.fixture
def window(app, video_path, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.PersonDetector", FakeDetector)
    config = {
        "roi": [[0, 0], [200, 0], [200, 200], [0, 200]],
        "person_limit": 1,
        "violation": {
            "frames_to_confirm_start": 2,
            "frames_to_confirm_end": 2,
            "cooldown_frames": 0,
        },
        "clip": {"seconds_before": 0.2, "seconds_after": 0.2},
        "detection": {"model": "fake.pt", "confidence_threshold": 0.4},
    }
    config_path = tmp_path / "config.yaml"
    save_config(config_path, config)
    win = MainWindow(video_path, config, str(config_path), str(tmp_path / "out"))
    win.show()
    yield win
    win.close()


def wait_until(app, condition, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def record_chart_points(window):
    points = []
    add_point = window.chart.add_point

    def recording_add_point(time_s, count):
        points.append((time_s, count))
        add_point(time_s, count)

    window.chart.add_point = recording_add_point
    return points


def widget_position(window, frame_x, frame_y):
    pixmap = window.video_label.pixmap()
    frame_width, frame_height = window.video_label.click_frame_size
    offset_x = (window.video_label.width() - pixmap.width()) / 2
    offset_y = (window.video_label.height() - pixmap.height()) / 2
    return QPoint(
        round(offset_x + frame_x * pixmap.width() / frame_width),
        round(offset_y + frame_y * pixmap.height() / frame_height),
    )


def test_window_shows_the_source_video_fps(window):
    assert f"Source video: {FPS:.2f} fps" in window.fps_label.text()
    assert window.start_button.isEnabled()
    assert not window.pause_button.isEnabled()


def test_zone_can_be_drawn_with_the_mouse_and_saved(window):
    window.edit_button.setChecked(True)
    targets = [(40, 30), (280, 40), (270, 200), (30, 210)]
    for frame_x, frame_y in targets:
        QTest.mouseClick(
            window.video_label,
            Qt.MouseButton.LeftButton,
            pos=widget_position(window, frame_x, frame_y),
        )
    QTest.mouseClick(
        window.video_label,
        Qt.MouseButton.LeftButton,
        pos=widget_position(window, 150, 120),
    )
    QTest.mouseClick(
        window.video_label,
        Qt.MouseButton.RightButton,
        pos=widget_position(window, 150, 120),
    )
    assert len(window.calibration_points) == 4

    window.limit_spin.setValue(3)
    window.save_button.click()

    saved = yaml.safe_load(open(window.config_path))
    assert saved["person_limit"] == 3
    assert len(saved["roi"]) == 4
    assert (
        max(
            abs(point[0] - target[0]) + abs(point[1] - target[1])
            for point, target in zip(saved["roi"], targets, strict=True)
        )
        <= 4
    )
    assert not window.edit_button.isChecked()


def test_a_zone_needs_at_least_three_points(window):
    window.edit_button.setChecked(True)
    window.save_button.click()
    assert "at least 3 points" in window.status_label.text()


def test_processing_reports_the_speed_and_saves_the_results(app, window):
    points = record_chart_points(window)
    window.start_button.click()
    assert not window.start_button.isEnabled()
    assert wait_until(app, lambda: "finished" in window.fps_label.text())
    assert not window._is_running()
    assert len(points) == FRAMES
    assert "Finished" in window.status_label.text()
    run_dirs = list((window.worker.run_dir.parent).iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "csv" / "series.csv").exists()
    assert (run_dirs[0] / "charts" / "occupancy.png").exists()


def test_view_is_cleared_when_the_video_ends(app, window):
    points = record_chart_points(window)
    window.start_button.click()
    assert wait_until(app, lambda: "finished" in window.fps_label.text())
    assert len(points) == FRAMES
    assert window.chart.timestamps == []
    assert window.chart.counts == []
    assert window.start_button.isEnabled()


def test_clear_view_resets_the_chart_and_shows_the_first_frame(app, window):
    window.chart.add_point(0.0, 3)
    window.video_label.clear()
    window.clear_view()
    assert window.chart.timestamps == []
    assert window.video_label.pixmap() is not None
    assert not window.video_label.pixmap().isNull()


def test_pause_stops_the_frames_and_resume_continues(app, window):
    points = record_chart_points(window)
    window.start_button.click()
    assert wait_until(app, lambda: len(points) >= 2)
    window.pause_button.click()
    assert window.pause_button.text() == "Resume"
    assert "paused" in window.fps_label.text()
    time.sleep(0.3)
    app.processEvents()
    frozen = len(points)
    time.sleep(0.5)
    app.processEvents()
    assert len(points) == frozen
    window.pause_button.click()
    assert wait_until(app, lambda: not window._is_running())
    assert len(points) == FRAMES


def test_closing_the_window_stops_the_worker(app, window):
    points = record_chart_points(window)
    window.start_button.click()
    assert wait_until(app, lambda: len(points) >= 1)
    window.close()
    assert not window._is_running()


def test_chart_redraws_once_per_video_second(app):
    chart = LiveChart(limit=2, redraw_interval=1.0)
    redraws = []
    chart.redraw = lambda: (
        redraws.append(len(chart.timestamps)),
        setattr(chart, "last_redraw_time", chart.timestamps[-1]),
    )
    for index in range(50):
        chart.add_point(index * 0.1, 1)
    assert redraws == [1, 11, 21, 31, 41]
    assert len(chart.timestamps) == 50
