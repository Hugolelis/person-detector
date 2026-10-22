from PySide6.QtCore import Qt
from src.gui.zone_editor import ZoneEditor

LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton


def test_a_new_editor_is_idle_and_empty():
    editor = ZoneEditor()
    assert not editor.active
    assert editor.points == []


def test_begin_clears_the_points_and_activates_editing():
    editor = ZoneEditor()
    editor.points = [(1, 1)]
    editor.begin()
    assert editor.active
    assert editor.points == []


def test_left_click_adds_and_right_click_removes_the_last_point():
    editor = ZoneEditor()
    editor.begin()
    editor.click(10, 20, LEFT)
    editor.click(30, 40, LEFT)
    editor.click(0, 0, RIGHT)
    assert editor.points == [(10, 20)]


def test_right_click_without_points_does_nothing():
    editor = ZoneEditor()
    editor.begin()
    editor.click(0, 0, RIGHT)
    assert editor.points == []


def test_undo_reports_whether_it_removed_something():
    editor = ZoneEditor()
    editor.begin()
    assert not editor.undo()
    editor.click(5, 5, LEFT)
    assert editor.undo()
    assert editor.points == []


def test_a_zone_needs_three_points_to_be_saved():
    editor = ZoneEditor()
    editor.begin()
    for point in [(0, 0), (10, 0)]:
        editor.click(*point, LEFT)
    assert not editor.can_save
    editor.click(10, 10, LEFT)
    assert editor.can_save


def test_roi_is_a_list_of_lists_ready_for_the_yaml():
    editor = ZoneEditor()
    editor.begin()
    editor.click(1, 2, LEFT)
    assert editor.roi() == [[1, 2]]
