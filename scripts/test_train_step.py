"""Validate the custom similarity loss and the immediate LoRA update step."""

from __future__ import annotations

from src.config import load_config
from src.model.vlm import VlmModel

BOARD = "2048 board (rows top to bottom, '.' means empty):\n2 . 4 4\n. . 2 .\n. . . .\n. . . .\nScore: 8"


def main() -> None:
    config = load_config()
    model = VlmModel(
        model_dir=config.path("paths.model_dir"),
        dtype=config.get("model.dtype", "bfloat16"),
        lora=config.get("model.lora"),
        optimizer=config.get("model.optimizer"),
        loss_cfg=config.get("loss"),
    ).load()

    messages = [
        {"role": "user", "content": BOARD + "\nChoose one move: up, down, left, right."}
    ]
    input_ids, attention_mask = model.encode_messages(messages)

    for actions in (["down"], ["left"]):
        right = actions[0]
        token_id = model.action_token_id(right)
        before = model.action_similarity(input_ids, attention_mask, token_id)
        print(f"[{right}] cosine before = {before:.4f}")
        for step in range(5):
            metrics = model.train_step(input_ids, attention_mask, token_id)
            print(f"  step {step}: loss={metrics['loss']:.4f} cos={metrics['cosine']:.4f}")
        after = model.action_similarity(input_ids, attention_mask, token_id)
        print(f"[{right}] cosine after  = {after:.4f}  (delta {after - before:+.4f})")


if __name__ == "__main__":
    main()
