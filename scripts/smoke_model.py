"""Smoke test: load FastVLM-0.5B and run a single image -> text generation.

Goal: verify the model loads with its remote code and can read a screenshot.
Run:
    conda run -n fastvlm python scripts/smoke_model.py --image web/2048/smoke.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

IMAGE_TOKEN_INDEX = -200


def build_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    """Splice the image placeholder token into the chat template, per model README."""
    messages = [{"role": "user", "content": "<image>\n" + prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    pre, post = rendered.split("<image>", 1)
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    return torch.cat([pre_ids, img_tok, post_ids], dim=1)


def resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    model_path = str(Path(args.model_dir).resolve())
    dtype = resolve_dtype(args.dtype)

    print(f"[load] model={model_path} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    # This checkpoint ships a minimal generation_config.json without special tokens,
    # which breaks the inputs_embeds generation path; restore them from the model config.
    generation_config = model.generation_config
    generation_config.bos_token_id = model.config.bos_token_id
    generation_config.eos_token_id = model.config.eos_token_id
    generation_config.pad_token_id = model.config.eos_token_id
    generation_config.do_sample = False

    input_ids = build_input_ids(tokenizer, args.prompt)
    attention_mask = torch.ones_like(input_ids)

    image = Image.open(args.image).convert("RGB")
    pixel_values = model.get_vision_tower().image_processor(
        images=image, return_tensors="pt"
    )["pixel_values"].to(model.dtype)
    print(f"[input] image={args.image} size={image.size} tokens={input_ids.shape[1]}")

    with torch.no_grad():
        out = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=args.max_new_tokens,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("[output]")
    print(text)


if __name__ == "__main__":
    main()
