import yaml
import os
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class Config:
    """Type-safe configuration container."""

    data: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        if name in self.data:
            val = self.data[name]
            if isinstance(val, dict):
                return Config(val)
            return val
        raise AttributeError(f"'Config' object has no attribute '{name}'")


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """Loads and parses the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)
