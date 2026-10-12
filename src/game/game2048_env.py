"""2048 environment driven through a headless browser (fully offline).

Design notes:
- The model may see either text (default) or a screenshot; the DOM is read by the
  harness for scoring and state serialization.
- The page exposes window.game2048 for deterministic control.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, Playwright, sync_playwright

from src.game.base import GameEnv
from src.io.state_text import format_state
from src.loop.schema import Observation

_KEY_MAP = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
}


class Game2048WebEnv(GameEnv):
    """A 2048 episode running in a headless Chromium page."""

    def __init__(
        self,
        page_path: Path,
        actions: list[str],
        viewport: dict,
        seed: int = 42,
        capture_image: bool = False,
        full_page: bool = True,
    ) -> None:
        self._page_path = Path(page_path).resolve()
        self._actions = list(actions)
        self._viewport = {"width": int(viewport["width"]), "height": int(viewport["height"])}
        self._seed = int(seed)
        self._capture_image = bool(capture_image)
        self._full_page = bool(full_page)

        self._playwright: Playwright | None = None
        self._page: Page | None = None

    # -- lifecycle ---------------------------------------------------------
    def _ensure_page(self) -> Page:
        if self._page is not None:
            return self._page
        self._playwright = sync_playwright().start()
        browser = self._playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport=self._viewport)
        self._page = context.new_page()
        return self._page

    def _url(self) -> str:
        return f"file://{self._page_path}?seed={self._seed}"

    def reset(self) -> Observation:
        page = self._ensure_page()
        page.goto(self._url())
        page.wait_for_function("() => !!window.game2048")
        return self.observe()

    # -- core API ----------------------------------------------------------
    def step(self, action: str) -> tuple[Observation, bool, dict]:
        if action not in self._actions:
            raise ValueError(f"unknown action: {action!r}; expected one of {self._actions}")
        page = self._page
        assert page is not None, "call reset() before step()"
        page.keyboard.press(_KEY_MAP[action])
        page.wait_for_timeout(40)
        observation = self.observe()
        info = {"score": observation.score, "over": observation.over}
        return observation, observation.over, info

    def observe(self) -> Observation:
        page = self._page
        assert page is not None
        state = page.evaluate("() => window.game2048.getState()")
        board = state["grid"]
        observation = Observation(
            image=self._screenshot(page) if self._capture_image else None,
            text=format_state(board, int(state["score"]), bool(state["over"])),
            score=int(state["score"]),
            over=bool(state["over"]),
            board=board,
        )
        return observation

    def close(self) -> None:
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
            self._page = None

    @property
    def actions(self) -> list[str]:
        return list(self._actions)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _screenshot(page: Page) -> Image.Image:
        png = page.screenshot(full_page=True)
        return Image.open(BytesIO(png)).convert("RGB")
