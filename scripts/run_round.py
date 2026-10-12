"""Run a few rounds of the core loop on 2048 (text perception mode)."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.game.game2048_env import Game2048WebEnv
from src.loop.round import RoundRunner
from src.memory.store import MemoryStore
from src.model.vlm import VlmModel


def build_model(config) -> VlmModel:
    model = VlmModel(
        model_dir=config.path("paths.model_dir"),
        dtype=config.get("model.dtype", "bfloat16"),
        device=config.get("model.device", "cpu"),
        image_token_index=config.get("model.image_token_index", -200),
        lora=config.get("model.lora"),
        optimizer=config.get("model.optimizer"),
        loss_cfg=config.get("loss"),
    )
    print("[model] loading ...")
    model.load()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    rounds = args.rounds if args.rounds is not None else config.get("round.max_rounds", 5)
    capture_image = config.get("perception.mode", "text") == "image"

    model = build_model(config)
    memory = MemoryStore(
        config.path("paths.short_term_memory"), config.path("paths.long_term_memory")
    )
    actions = config.get("game.action_words")
    env = Game2048WebEnv(
        page_path=config.path("paths.web_2048"),
        actions=actions,
        viewport=config.get("game.viewport"),
        seed=args.seed if args.seed is not None else config.get("game.seed", 42),
        capture_image=capture_image,
    )
    runner = RoundRunner(
        model, memory, actions, max_new_tokens=config.get("round.max_new_tokens", 96)
    )

    try:
        observation = env.reset()
        for index in range(rounds):
            report, observation = runner.run(env, observation, index)
            print("-" * 70)
            print(f"round {index}: action={report.action} score={report.score} "
                  f"over={report.over}")
            print(f"  thought        : {report.self_thought[:120]}")
            print(f"  evaluation     : {report.evaluation[:120]}")
            print(f"  wrong/right    : {report.wrong_action} -> {report.right_action}")
            print(f"  finetune       : {report.finetune} metrics={report.train_metrics}")
            print(f"  short-memory   : {report.short_term_memory[:120]}")
            if report.notes:
                print(f"  notes          : {report.notes}")
            if report.over:
                print("[game over]")
                break
        model.save_adapter(config.path("paths.lora_output_dir"))
        print(f"[done] adapter saved to {config.path('paths.lora_output_dir')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
