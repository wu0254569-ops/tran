"""Check action tokenization and text-only generation for the 0.5B model."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("model").resolve()
ACTIONS = ["up", "down", "left", "right"]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    for action in ACTIONS:
        plain = tokenizer.encode(action, add_special_tokens=False)
        spaced = tokenizer.encode(" " + action, add_special_tokens=False)
        print(f"{action:6s} plain={plain} spaced={spaced} "
              f"decoded={[tokenizer.decode([t]) for t in spaced]}")

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    gen = model.generation_config
    gen.bos_token_id = model.config.bos_token_id
    gen.eos_token_id = model.config.eos_token_id
    gen.pad_token_id = model.config.eos_token_id
    gen.do_sample = False

    prompt = (
        "You are playing 2048. Board (rows top to bottom):\n"
        "2 . 4 4\n. . 2 .\n. . . .\n. . . .\n"
        "Choose one move: up, down, left, right. Answer with one word."
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        out = model.generate(inputs=input_ids, max_new_tokens=8)
    print("text-only generation ->", repr(tokenizer.decode(out[0], skip_special_tokens=True)))


if __name__ == "__main__":
    main()
