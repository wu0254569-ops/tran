"""2048 environment driven through a headless browser (fully offline).

Design notes:
- The model only ever sees screenshots. Scores are read from the DOM by the
  harness for logging / evaluation and are never fed to the model.
- The page exposes window.game2048 for deterministic control.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, Playwright, sync_playwright

from src.game.base import GameEnv
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
        full_page: bool = True,
    ) -> None:
        self._page_path = Path(page_path).resolve()
        self._actions = list(actions)
        self._viewport = {"width": int(viewport["width"]), "height": int(viewport["height"])}
        self._seed = int(seed)
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
        return self._screenshot()

    # -- core API ----------------------------------------------------------
    def step(self, action: str) -> tuple[Observation, bool, dict]:
        if action not in self._actions:
            raise ValueError(f"unknown action: {action!r}; expected one of {self._actions}")
        page = self._page
        assert page is not None, "call reset() before step()"
        page.keyboard.press(_KEY_MAP[action])
        page.wait_for_timeout(50)
        state = page.evaluate("() => window.game2048.getState()")
        observation = self._screenshot()
        info = {"score": state["score"], "over": state["over"]}
        return observation, bool(state["over"]), info

    def observe(self) -> Observation:
        return self._screenshot()

    def close(self) -> None:
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
            self._page = None

    @property
    def actions(self) -> list[str]:
        return list(self._actions)

    # -- helpers -----------------------------------------------------------
    def _screenshot(self) -> Observation:
        page = self._page
        assert page is not None
        png = page.screenshot(full_page=self._full_page)
        image = Image.open(BytesIO(png)).convert("RGB")
        state = page.evaluate("() => window.game2048.getState()")
        return Observation(image=image, score=int(state["score"]), over=bool(state["over"]))
