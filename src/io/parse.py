"""Tolerant parsing of small-model outputs into structured fields."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in `text`, or None."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def find_action(text: str, actions: list[str]) -> str | None:
    """Return the first action word mentioned in `text`."""
    lowered = text.lower()
    best: tuple[int, str] | None = None
    for action in actions:
        match = re.search(rf"\b{re.escape(action.lower())}\b", lowered)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), action)
    return best[1] if best else None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)
