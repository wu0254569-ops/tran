"""Short-term (model cache) and long-term memory, backed by JSON files."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.loop.schema import LongTermOp


class MemoryStore:
    def __init__(self, short_path: str | Path, long_path: str | Path) -> None:
        self.short_path = Path(short_path)
        self.long_path = Path(long_path)
        self.short_path.parent.mkdir(parents=True, exist_ok=True)

    # -- short-term memory (model cache, fully overwritten each round) -----
    def read_short(self) -> str:
        if not self.short_path.exists():
            return ""
        data = json.loads(self.short_path.read_text(encoding="utf-8"))
        return str(data.get("content", ""))

    def write_short(self, content: Any) -> None:
        if isinstance(content, (list, tuple)):
            content = "\n".join(str(item) for item in content)
        self.short_path.write_text(
            json.dumps({"content": content}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -- long-term memory --------------------------------------------------
    def _load_long(self) -> dict[str, Any]:
        if not self.long_path.exists():
            return {}
        return json.loads(self.long_path.read_text(encoding="utf-8"))

    def _save_long(self, data: dict[str, Any]) -> None:
        self.long_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def long_index(self) -> list[str]:
        return sorted(self._load_long().keys())

    def apply_ops(self, ops: list[LongTermOp]) -> None:
        data = self._load_long()
        for op in ops:
            if op.op == "write" or op.op == "update":
                data[op.key] = op.value
            elif op.op == "delete":
                data.pop(op.key, None)
        self._save_long(data)

    def read_long(self, keys: list[str], fuzzy: bool = True) -> dict[str, Any]:
        data = self._load_long()
        result: dict[str, Any] = {}
        for key in keys:
            if key in data:
                result[key] = data[key]
                continue
            if not fuzzy:
                continue
            best_key, best_score = None, 0.0
            for candidate in data:
                score = SequenceMatcher(None, key.lower(), candidate.lower()).ratio()
                if score > best_score:
                    best_key, best_score = candidate, score
            if best_key is not None and best_score >= 0.6:
                result[best_key] = data[best_key]
        return result
