"""Diagnostic: how far can the 0.5B model go with a maximally legible board?

Renders a synthetic 4x4 board (no page chrome, huge digits, labeled rows/cols)
and asks the model to read it. This is a diagnostic only; the experiment still
feeds the real game screenshot.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from scripts.diag_vision import load_model, run

GRID = [[2, 0, 4, 4], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
CELL = 220
PAD = 40
LABEL = 48


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(grid: list[list[int]]) -> Image.Image:
    width = PAD * 2 + LABEL + CELL * 4
    height = PAD * 2 + LABEL + CELL * 4
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    big = load_font(120)
    small = load_font(40)

    origin_x = PAD + LABEL
    origin_y = PAD + LABEL
    for r in range(4):
        draw.text((PAD, origin_y + r * CELL + CELL // 2 - 20), str(r + 1), fill="black", font=small)
    for c in range(4):
        draw.text((origin_x + c * CELL + CELL // 2 - 16, PAD + 8), "ABCD"[c], fill="black", font=small)

    for r in range(4):
        for c in range(4):
            x0 = origin_x + c * CELL
            y0 = origin_y + r * CELL
            draw.rectangle([x0, y0, x0 + CELL, y0 + CELL], outline="black", width=6)
            value = grid[r][c]
            text = str(value)
            color = "black" if value else "lightgray"
            box = draw.textbbox((0, 0), text, font=big)
            tw, th = box[2] - box[0], box[3] - box[1]
            draw.text((x0 + (CELL - tw) // 2, y0 + (CELL - th) // 2 - box[1]), text, fill=color, font=big)
    return image


def main() -> None:
    image = render(GRID)
    image.save("logs/diag_board_synth.png")
    tokenizer, model = load_model("bfloat16")
    prompts = [
        "What number is in row 1 column C?",
        "What number is in row 2 column C?",
        "Read row 1 from left to right as four numbers.",
        "Read row 3 from left to right as four numbers.",
    ]
    for prompt in prompts:
        print("=" * 60)
        print(f"prompt={prompt!r}")
        print("->", run(model, tokenizer, image, prompt, max_new_tokens=40))


if __name__ == "__main__":
    main()
