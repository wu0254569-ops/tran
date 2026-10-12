"""Configuration loading.

All tunable parameters live in configs/*.yaml; code never hard-codes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = "configs/default.yaml"


class Config:
    """Thin wrapper over a nested dict with dotted-path access."""

    def __init__(self, data: dict[str, Any], base_dir: Path) -> None:
        self._data = data
        self.base_dir = base_dir

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the project root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"missing path config: {dotted}")
        return (self.base_dir / value).resolve()

    def as_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path = _DEFAULT_PATH) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config(data, base_dir=config_path.parent.parent)
