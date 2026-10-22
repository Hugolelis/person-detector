from pathlib import Path

import yaml
from src.errors.config import ConfigError

REQUIRED_KEYS = ("roi", "person_limit", "violation", "clip", "detection")


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

    return config