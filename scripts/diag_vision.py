"""Vision diagnostic: can the 0.5B model read a 2048 board under different framings?

Loads the model once and runs several (image, prompt) conditions.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.smoke_model import build_input_ids, resolve_dtype

IMAGE_TOKEN_INDEX = -200
MODEL_DIR = Path("model").resolve()


def crop_board(image: Image.Image) -> Image.Image:
    # Board area inside the 720x900 page.
    return image.crop((24, 105, 696, 780))


def load_model(dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=resolve_dtype(dtype_name),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    gen = model.generation_config
    gen.bos_token_id = model.config.bos_token_id
    gen.eos_token_id = model.config.eos_token_id
    gen.pad_token_id = model.config.eos_token_id
    gen.do_sample = False
    return tokenizer, model


def run(model, tokenizer, image: Image.Image, prompt: str, max_new_tokens: int = 48) -> str:
    input_ids = build_input_ids(tokenizer, prompt)
    attention_mask = torch.ones_like(input_ids)
    pixel_values = model.get_vision_tower().image_processor(
        images=image, return_tensors="pt"
    )["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()


def main() -> None:
    full = Image.open("web/2048/smoke.png").convert("RGB")
    board = crop_board(full)
    board_big = board.resize((1024, 1024), Image.LANCZOS)
    board_big.save("logs/diag_board_big.png")

    tokenizer, model = load_model("bfloat16")

    prompt_grid = "Read the 4x4 numbers of this 2048 board, row by row."
    prompt_score = "What is the score number shown at the top of this 2048 game?"

    conditions = [
        ("full-page / grid prompt", full, prompt_grid),
        ("cropped board / grid prompt", board, prompt_grid),
        ("cropped+1024 / grid prompt", board_big, prompt_grid),
        ("cropped+1024 / score prompt", board_big, prompt_score),
    ]
    for name, image, prompt in conditions:
        print("=" * 60)
        print(f"[{name}] prompt={prompt!r} image={image.size}")
        print("->", run(model, tokenizer, image, prompt))


if __name__ == "__main__":
    main()
