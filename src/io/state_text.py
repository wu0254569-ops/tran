"""Serialize a game state to plain text (the current perception mode)."""

from __future__ import annotations

EMPTY = "."


def format_board(grid: list[list[int]]) -> str:
    """Render the 4x4 grid as text, one row per line, empty cells as '.'."""
    return "\n".join(
        " ".join(str(value) if value else EMPTY for value in row) for row in grid
    )


def format_state(grid: list[list[int]], score: int, over: bool) -> str:
    """Human-readable state block handed to the model."""
    return (
        "2048 board (rows top to bottom, '.' means empty):\n"
        f"{format_board(grid)}\n"
        f"Score: {score}\n"
        f"Game over: {over}"
    )
