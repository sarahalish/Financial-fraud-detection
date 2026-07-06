"""Configuration management.

Loads config/config.yaml once and resolves all paths relative to the
project root, so the pipeline works no matter where it is launched from.
"""

from pathlib import Path

import yaml

# Project root = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the YAML configuration and resolve paths to absolute paths."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve every entry in `paths` relative to the project root
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str((PROJECT_ROOT / value).resolve())

    return cfg
