import pytest
import yaml
from src.config_loader import load_config
from src.errors.config import ConfigError

BASE = {
    "roi": [[0, 0], [10, 0], [10, 10]],
    "person_limit": 1,
    "violation": {},
    "clip": {},
    "detection": {},
}


def write_config(tmp_path, **extra):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**BASE, **extra}))
    return path


def test_the_counting_section_is_optional(tmp_path):
    assert load_config(write_config(tmp_path))["person_limit"] == 1


@pytest.mark.parametrize("value", [0, 30, 99.5, 100])
def test_valid_overlap_percentages_are_accepted(tmp_path, value):
    path = write_config(tmp_path, counting={"min_overlap_percent": value})
    assert load_config(path)["counting"]["min_overlap_percent"] == value


@pytest.mark.parametrize("value", [-1, 101, "half", [30]])
def test_invalid_overlap_percentages_are_rejected(tmp_path, value):
    path = write_config(tmp_path, counting={"min_overlap_percent": value})
    with pytest.raises(ConfigError, match="min_overlap_percent"):
        load_config(path)
