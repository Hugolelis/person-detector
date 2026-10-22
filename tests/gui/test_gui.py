import sys
import time

import cv2
import numpy as np
import pytest
import yaml
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from src.config_loader import save_config
from src.gui.main_window import MainWindow
from src.gui.widgets.live_chart import LiveChart

FRAMES = 24
FPS = 10


class FakeDetector:
    delay = 0.0

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        time.sleep(FakeDetector.delay)
        return [(10, 10, 30, 80, 0.9), (60, 10, 80, 80, 0.8)]


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def write_video(path, fps=FPS, size=(320, 240), frames=FRAMES):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), index * 8, dtype=np.uint8))
    writer.release()
    return str(path)


@pytest.fixture
def video_path(tmp_path):
    return write_video(tmp_path / "clip.mp4")


def make_window(tmp_path, video_path, monkeypatch):
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
    return win


@pytest.fixture
def window(app, video_path, tmp_path, monkeypatch):
    win = make_window(tmp_path, video_path, monkeypatch)
    yield win
    win.close()


@pytest.fixture
def empty_window(app, tmp_path, monkeypatch):
    win = make_window(tmp_path, None, monkeypatch)
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
    assert window.run_panel.start_button.isEnabled()
    assert not window.run_panel.pause_button.isEnabled()


def test_zone_can_be_drawn_with_the_mouse_and_saved(window):
    window.zone_panel.edit_button.setChecked(True)
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
    assert len(window.zone_editor.points) == 4

    window.zone_panel.limit_spin.setValue(3)
    window.zone_panel.save_button.click()

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
    assert not window.zone_panel.edit_button.isChecked()


def test_a_zone_needs_at_least_three_points(window):
    window.zone_panel.edit_button.setChecked(True)
    window.zone_panel.save_button.click()
    assert "at least 3 points" in window.status_label.text()


def test_processing_reports_the_speed_and_saves_the_results(app, window):
    points = record_chart_points(window)
    window.run_panel.start_button.click()
    assert not window.run_panel.start_button.isEnabled()
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
    window.run_panel.start_button.click()
    assert wait_until(app, lambda: "finished" in window.fps_label.text())
    assert len(points) == FRAMES
    assert window.chart.timestamps == []
    assert window.chart.counts == []
    assert window.run_panel.start_button.isEnabled()


def test_clear_view_resets_the_chart_and_shows_the_first_frame(app, window):
    window.chart.add_point(0.0, 3)
    window.video_label.clear()
    window.clear_view()
    assert window.chart.timestamps == []
    assert window.video_label.pixmap() is not None
    assert not window.video_label.pixmap().isNull()


def test_pause_stops_the_frames_and_resume_continues(app, window, monkeypatch):
    monkeypatch.setattr(FakeDetector, "delay", 0.05)
    points = record_chart_points(window)
    window.run_panel.start_button.click()
    assert wait_until(app, lambda: len(points) >= 2)
    window.run_panel.pause_button.click()
    assert window.run_panel.pause_button.text() == "Resume"
    assert "paused" in window.fps_label.text()
    time.sleep(0.3)
    app.processEvents()
    frozen = len(points)
    time.sleep(0.5)
    app.processEvents()
    assert len(points) == frozen
    window.run_panel.pause_button.click()
    assert wait_until(app, lambda: not window._is_running())
    assert len(points) == FRAMES


def test_closing_the_window_stops_the_worker(app, window):
    points = record_chart_points(window)
    window.run_panel.start_button.click()
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


def test_window_can_start_without_a_video(empty_window):
    assert not empty_window.run_panel.start_button.isEnabled()
    assert not empty_window.zone_panel.edit_button.isEnabled()
    assert empty_window.source_panel.open_button.isEnabled()
    assert "Open a video" in empty_window.video_label.text()


def test_choosing_a_video_loads_it_and_enables_processing(empty_window, tmp_path):
    other = write_video(tmp_path / "other.mp4", fps=25)
    empty_window._ask_video_path = lambda: other
    empty_window.source_panel.open_button.click()
    assert empty_window.source == other
    assert empty_window.source_panel.name_label.text() == "other.mp4"
    assert "Source video: 25.00 fps" in empty_window.fps_label.text()
    assert empty_window.run_panel.start_button.isEnabled()


def test_cancelling_the_dialog_keeps_the_current_video(window):
    current = window.source
    window._ask_video_path = lambda: ""
    window.source_panel.open_button.click()
    assert window.source == current


def test_an_invalid_file_keeps_the_current_video(window, tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video")
    current = window.source
    window._ask_video_path = lambda: str(broken)
    window.source_panel.open_button.click()
    assert window.source == current
    assert "Error" in window.status_label.text()
    assert window.run_panel.start_button.isEnabled()


def test_changing_video_clears_the_chart_and_leaves_zone_editing(window, tmp_path):
    other = write_video(tmp_path / "other.mp4")
    window.zone_panel.edit_button.setChecked(True)
    window.chart.add_point(0.0, 3)
    window._ask_video_path = lambda: other
    window.source_panel.open_button.click()
    assert window.chart.timestamps == []
    assert not window.zone_panel.edit_button.isChecked()


def test_the_video_cannot_be_changed_while_processing(app, window, tmp_path):
    window.run_panel.start_button.click()
    assert not window.source_panel.open_button.isEnabled()
    other = write_video(tmp_path / "other.mp4")
    window._ask_video_path = lambda: other
    window.choose_video()
    assert window.source != other
    assert wait_until(app, lambda: "finished" in window.fps_label.text())
    assert window.source_panel.open_button.isEnabled()


def test_results_folder_is_named_after_the_chosen_video(app, empty_window, tmp_path):
    other = write_video(tmp_path / "street.mp4")
    empty_window._ask_video_path = lambda: other
    empty_window.source_panel.open_button.click()
    empty_window.run_panel.start_button.click()
    assert wait_until(app, lambda: "finished" in empty_window.fps_label.text())
    assert empty_window.worker.run_dir.name.startswith("street_")


def test_webcam_button_loads_a_live_source(empty_window, fake_camera):
    empty_window.source_panel.webcam_button.click()
    assert empty_window.source == 0
    assert empty_window.is_live
    assert "webcam" in empty_window.source_panel.name_label.text().lower()
    assert "webcam" in empty_window.fps_label.text().lower()
    assert empty_window.run_panel.start_button.isEnabled()


def test_unavailable_webcam_shows_an_error_and_keeps_start_disabled(
    empty_window, fake_camera
):
    fake_camera.broken_indexes = {0}
    empty_window.source_panel.webcam_button.click()
    assert "Could not open camera 0" in empty_window.status_label.text()
    assert empty_window.source is None
    assert not empty_window.run_panel.start_button.isEnabled()


def test_webcam_runs_until_stop_and_uses_real_time(app, empty_window, fake_camera):
    fake_camera.delay = 0.06
    empty_window.source_panel.webcam_button.click()
    points = record_chart_points(empty_window)
    empty_window.run_panel.start_button.click()
    assert empty_window.run_panel.stop_button.isEnabled()
    assert not empty_window.source_panel.webcam_button.isEnabled()
    assert wait_until(app, lambda: len(points) >= 12)
    assert empty_window._is_running()
    empty_window.run_panel.stop_button.click()
    assert wait_until(app, lambda: "finished" in empty_window.fps_label.text())
    assert not empty_window.run_panel.stop_button.isEnabled()
    assert empty_window.chart.timestamps == []

    run_dir = empty_window.worker.run_dir
    assert run_dir.name.startswith("webcam_")
    times = [time_s for time_s, _ in points]
    assert times == sorted(times)
    # Frames arrive every ~60 ms, while frame index / 30 fps would say ~33 ms.
    assert times[-1] > 1.5 * (len(times) - 1) / 30
    assert (run_dir / "csv" / "series.csv").exists()
    assert (run_dir / "charts" / "occupancy.png").exists()


def test_stop_ends_a_video_file_run_early(app, window, monkeypatch):
    monkeypatch.setattr(FakeDetector, "delay", 0.05)
    points = record_chart_points(window)
    window.run_panel.start_button.click()
    assert wait_until(app, lambda: len(points) >= 2)
    window.run_panel.stop_button.click()
    assert wait_until(app, lambda: not window._is_running())
    assert len(points) < FRAMES


def test_depth_panel_follows_the_checkbox_and_is_off_by_default(window):
    assert not window.run_panel.depth_check.isChecked()
    assert not window.depth_label.isVisible()
    window.run_panel.depth_check.setChecked(True)
    assert window.depth_label.isVisible()


def test_the_checkbox_is_locked_while_processing(app, window):
    window.run_panel.start_button.click()
    assert not window.run_panel.depth_check.isEnabled()
    assert wait_until(app, lambda: "finished" in window.fps_label.text())
    assert window.run_panel.depth_check.isEnabled()


def test_depth_choice_goes_to_the_run_but_not_to_the_saved_config(window):
    window.run_panel.depth_check.setChecked(True)
    assert window._run_config()["depth"]["enabled"] is True
    assert "depth" not in window.config


def test_processing_with_depth_fills_the_depth_panel(app, window, monkeypatch):
    class FakeEstimator:
        def __init__(self, *args):
            pass

        def estimate(self, frame):
            return np.full(frame.shape[:2], 0.5, dtype=np.float32)

    monkeypatch.setattr("src.pipeline.DepthEstimator", FakeEstimator)
    monkeypatch.setattr(FakeDetector, "delay", 0.03)
    window.config["depth"] = {"stride": 1}
    window.run_panel.depth_check.setChecked(True)
    window.run_panel.start_button.click()
    assert wait_until(
        app,
        lambda: (
            window.depth_label.pixmap() is not None
            and not window.depth_label.pixmap().isNull()
        ),
    )
    assert wait_until(app, lambda: not window._is_running())
    assert "Error" not in window.status_label.text()
    assert window.depth_label.text() == "The depth view starts with the processing."


def test_a_missing_depth_package_is_reported_in_the_window(app, window, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    window.run_panel.depth_check.setChecked(True)
    window.run_panel.start_button.click()
    assert wait_until(
        app, lambda: "poetry add transformers" in window.status_label.text()
    )
    assert wait_until(app, lambda: not window._is_running())
    assert "poetry add transformers" in window.status_label.text()
    assert wait_until(app, lambda: "stopped" in window.fps_label.text())
    assert "measuring" not in window.fps_label.text()


def fits_the_label(label):
    pixmap = label.pixmap()
    return (
        pixmap.width() <= label.width()
        and pixmap.height() <= label.height()
        and (pixmap.width() == label.width() or pixmap.height() == label.height())
    )


def test_the_image_is_refitted_when_the_depth_panel_opens(app, window):
    app.processEvents()
    assert fits_the_label(window.video_label)
    before = window.video_label.width()
    window.run_panel.depth_check.setChecked(True)
    app.processEvents()
    assert window.video_label.width() < before
    assert fits_the_label(window.video_label)


def test_the_image_is_refitted_when_the_window_is_resized(app, window):
    window.resize(1000, 700)
    app.processEvents()
    assert fits_the_label(window.video_label)
    window.resize(1500, 950)
    app.processEvents()
    assert fits_the_label(window.video_label)


def test_the_zone_keeps_its_place_on_the_image_after_a_resize(app, window):
    window.zone_panel.edit_button.setChecked(True)
    window.resize(1100, 750)
    app.processEvents()
    QTest.mouseClick(
        window.video_label,
        Qt.MouseButton.LeftButton,
        pos=widget_position(window, 80, 60),
    )
    assert window.zone_editor.points[0] == pytest.approx((80, 60), abs=4)
    assert fits_the_label(window.video_label)


def test_showing_a_message_discards_the_old_image(app, empty_window):
    empty_window.depth_label.show_image(QPixmap(40, 30))
    empty_window.depth_label.setText("hello")
    empty_window.depth_label.resize(300, 200)
    app.processEvents()
    assert empty_window.depth_label.text() == "hello"
