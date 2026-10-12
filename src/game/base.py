"""Game environment abstraction.

Keeping this interface minimal lets us swap 2048 for Pokemon later without
touching the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.loop.schema import Observation


class GameEnv(ABC):
    """A turn-based game driven by single-word actions."""

    @abstractmethod
    def reset(self) -> Observation:
        """Start a new episode and return the first observation."""

    @abstractmethod
    def step(self, action: str) -> tuple[Observation, bool, dict]:
        """Apply one action; return (observation, done, info)."""

    @abstractmethod
    def observe(self) -> Observation:
        """Return the current observation without acting."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def actions(self) -> list[str]:
        """The action vocabulary (single words)."""
