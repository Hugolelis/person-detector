from pathlib import Path

import yaml
from src.errors.config import ConfigError

REQUIRED_KEYS = ("roi", "person_limit", "violation", "clip", "detection")


class _ConfigDumper(yaml.SafeDumper):
    pass


def _represent_list(dumper, data):
    is_flat = all(isinstance(item, (int, float)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=is_flat)


_ConfigDumper.add_representer(list, _represent_list)


def load_config(path):
    path = Path(path)
    try:
        with open(path) as file:
            config = yaml.safe_load(file)
    except OSError as e:
        raise ConfigError(f"Could not read config file: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {path}") from e

    if not isinstance(config, dict):
        raise ConfigError(f"Config file is empty or malformed: {path}")

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ConfigError(f"Missing keys in config: {', '.join(missing)}")

    overlap = config.get("counting", {}).get("min_overlap_percent")
    if overlap is not None and not (
        isinstance(overlap, (int, float)) and 0 <= overlap <= 100
    ):
        raise ConfigError(
            f"counting.min_overlap_percent must be a number from 0 to 100: {overlap}"
        )

    return config


def save_config(path, config):
    path = Path(path)
    try:
        with open(path, "w") as file:
            yaml.dump(config, file, Dumper=_ConfigDumper, sort_keys=False)
    except OSError as e:
        raise ConfigError(f"Could not write config file: {path}") from e
