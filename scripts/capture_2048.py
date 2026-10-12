"""Capture a 2048 screenshot to a PNG file (also used to build a smoke image)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.game.game2048_env import Game2048WebEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="web/2048/smoke.png")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--moves", default="", help="comma-separated actions before capture")
    args = parser.parse_args()

    config = load_config()
    env = Game2048WebEnv(
        page_path=config.path("paths.web_2048"),
        actions=config.get("game.action_words"),
        viewport=config.get("game.viewport"),
        seed=args.seed if args.seed is not None else config.get("game.seed", 42),
        full_page=config.get("game.screenshot_full_page", True),
    )
    try:
        observation = env.reset()
        for action in [a for a in args.moves.split(",") if a]:
            observation, _, _ = env.step(action)
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        observation.image.save(out_path)
        print(f"saved {out_path} (score={observation.score}, over={observation.over})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
