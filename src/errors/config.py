from .base import PersonDetectorError


class ConfigError(PersonDetectorError):
    """Raised when the config file is missing or invalid."""
