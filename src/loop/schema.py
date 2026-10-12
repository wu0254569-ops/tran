"""Core data contracts for the one-round interface.

These types are intentionally stable: swapping the model, the game or the whole
loop framework must not change them (see 系统提示词/系统提示词.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """A snapshot of the environment.

    In text mode only `text` is populated; in image mode `image` is set. Both
    may be present. `board` carries the structured grid for logging / scoring.
    """

    image: Any = None  # PIL.Image.Image | None
    text: str = ""
    score: int = 0
    over: bool = False
    board: Any = None  # list[list[int]] | None


@dataclass
class RoundInput:
    """Input at the start of a round."""

    raw_image: Observation
    short_term_memory: str = ""
    long_term_index: list[str] = field(default_factory=list)


@dataclass
class DecisionOutput:
    """Model output of the decision stage."""

    self_thought: str
    action: str
    long_term_read: list[str] = field(default_factory=list)


@dataclass
class EnvironmentStep:
    """The environment's reaction to one action."""

    action: str
    result_image: Observation


@dataclass
class LongTermOp:
    """One instruction the model asks the system to execute on long-term memory."""

    op: str  # write | update | delete
    key: str
    value: Any = None


@dataclass
class ReflectionOutput:
    """Model output of the reflection + memory stage."""

    result_evaluation: str
    self_reflection: str
    wrong_action: str
    right_action: str
    finetune: bool
    finetune_loss: float | None = None
    long_term_ops: list[LongTermOp] = field(default_factory=list)
    short_term_memory: str = ""
