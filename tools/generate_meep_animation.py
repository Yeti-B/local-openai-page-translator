from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageSequence


CANVAS_SIZE = 192
MASK_SCALE = 4
FRAME_DURATION_MS = 50
PAUSE_DURATION_MS = 500
SWING_CYCLES = 3
CYCLE_FRAMES = 8
BODY_SWING_PX = 11


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    bbox = image.getchannel("A").getbbox()
    if not bbox:
        raise ValueError("source image has no visible pixels")
    return bbox


def fit_source(source_path: Path) -> Image.Image:
    image = Image.open(source_path).convert("RGBA")
    bird = image.crop(alpha_bbox(image))

    margin = 4
    scale = min(
        (CANVAS_SIZE - margin * 2) / bird.width,
        (CANVAS_SIZE - margin * 2) / bird.height,
    )
    size = (round(bird.width * scale), round(bird.height * scale))
    bird = bird.resize(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    x = (CANVAS_SIZE - size[0]) // 2
    y = round((CANVAS_SIZE - size[1]) * 0.54)
    canvas.alpha_composite(bird, (x, y))
    return canvas


def scaled_points(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(round(x * MASK_SCALE), round(y * MASK_SCALE)) for x, y in points]


def draw_scaled_ellipse(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: int,
) -> None:
    draw.ellipse(tuple(round(value * MASK_SCALE) for value in box), fill=fill)


def mask_from_shapes(kind: str) -> Image.Image:
    size = CANVAS_SIZE * MASK_SCALE
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    if kind in {"head", "forbidden"}:
        # Fixed no-body zone: head, full bill, bill root, mouth gap, and a
        # broad front-chest cover. Head and forbidden use the same footprint
        # so erased body pixels cannot leave an uncovered transparent seam.
        draw_scaled_ellipse(draw, (114, 35, 164, 93), 255)
        draw.polygon(scaled_points([(137, 56), (191, 79), (192, 111), (134, 91)]), fill=255)
        draw.polygon(scaled_points([(126, 70), (176, 86), (176, 123), (120, 115)]), fill=255)
        draw.polygon(scaled_points([(108, 52), (154, 47), (169, 139), (102, 145), (96, 87)]), fill=255)
        draw.polygon(scaled_points([(102, 94), (160, 91), (160, 153), (104, 154)]), fill=255)
    elif kind == "body":
        # Body and legs only, with generous overlap under the fixed chest
        # cover. The upper face/bill source pixels are intentionally excluded.
        draw.rectangle((0, 0, round(118 * MASK_SCALE), size), fill=255)
        draw.polygon(scaled_points([(108, 72), (156, 96), (160, 154), (96, 156)]), fill=255)
        draw.rectangle((98 * MASK_SCALE, 120 * MASK_SCALE, 166 * MASK_SCALE, size), fill=255)
    else:
        raise ValueError(f"unknown mask kind: {kind}")

    return mask.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)


def harden_mask(mask: Image.Image) -> Image.Image:
    return mask.point(lambda value: 255 if value else 0)


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    alpha = ImageChops.multiply(image.getchannel("A"), mask)
    layer = image.copy()
    layer.putalpha(alpha)
    return layer


def shift_layer(image: Image.Image, dx: float, dy: float) -> Image.Image:
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.Resampling.BICUBIC,
    )


def erase_forbidden_pixels(body: Image.Image, forbidden_mask: Image.Image) -> Image.Image:
    hard_forbidden = harden_mask(forbidden_mask)
    allowed = ImageChops.invert(hard_forbidden)
    cleaned = body.copy()
    cleaned.putalpha(ImageChops.multiply(cleaned.getchannel("A"), allowed))
    return cleaned


def make_offsets() -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for cycle in range(SWING_CYCLES):
        for frame in range(CYCLE_FRAMES):
            phase = math.tau * frame / CYCLE_FRAMES
            dx = BODY_SWING_PX * math.sin(phase)
            dy = 0.8 * math.sin(phase + math.pi / 2)
            offsets.append((dx, dy))
    offsets.append((0.0, 0.0))
    return offsets


def make_durations(frame_count: int) -> list[int]:
    if frame_count < 2:
        return [PAUSE_DURATION_MS]
    return [FRAME_DURATION_MS] * (frame_count - 1) + [PAUSE_DURATION_MS]


def build_frames(base: Image.Image) -> tuple[list[Image.Image], Image.Image, Image.Image, Image.Image]:
    head_mask = mask_from_shapes("head")
    body_mask = mask_from_shapes("body")
    forbidden_mask = mask_from_shapes("forbidden")
    head_layer = apply_mask(base, harden_mask(head_mask))
    body_layer = apply_mask(base, body_mask)

    frames: list[Image.Image] = []
    guard_violations = Image.new("L", base.size, 0)
    post_clean_violations = Image.new("L", base.size, 0)

    for dx, dy in make_offsets():
        shifted_body = shift_layer(body_layer, dx, dy)
        shifted_alpha = shifted_body.getchannel("A")
        violation = ImageChops.multiply(shifted_alpha, forbidden_mask)
        guard_violations = ImageChops.lighter(guard_violations, violation)
        cleaned_body = erase_forbidden_pixels(shifted_body, forbidden_mask)
        post_clean_violation = ImageChops.multiply(cleaned_body.getchannel("A"), forbidden_mask)
        post_clean_violations = ImageChops.lighter(post_clean_violations, post_clean_violation)

        frame = Image.new("RGBA", base.size, (0, 0, 0, 0))
        frame.alpha_composite(cleaned_body)
        frame.alpha_composite(head_layer)
        frames.append(frame)

    return frames, forbidden_mask, guard_violations, post_clean_violations


def save_preview(frames: list[Image.Image], preview_path: Path) -> None:
    frame_w, frame_h = frames[0].size
    sheet = Image.new("RGBA", (frame_w * len(frames), frame_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    tile = 12
    for y in range(0, sheet.height, tile):
        for x in range(0, sheet.width, tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(238, 242, 247, 255))
    for index, frame in enumerate(frames):
        sheet.alpha_composite(frame, (index * frame_w, 0))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_path)


def nonzero_pixel_count(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return sum(histogram[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("extension/icons/woodcock-master.png"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("extension/icons/woodcock-meep.webp"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("extension/icons/woodcock-meep-preview.png"),
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"missing animation source: {args.source}")

    base = fit_source(args.source)
    frames, forbidden_mask, guard_violations, post_clean_violations = build_frames(base)
    durations = make_durations(len(frames))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        lossless=True,
        method=6,
        disposal=2,
    )
    save_preview(frames, args.preview)

    with Image.open(args.out) as animated:
        frame_count = sum(1 for _ in ImageSequence.Iterator(animated))

    forbidden_pixels = nonzero_pixel_count(forbidden_mask)
    violation_pixels = nonzero_pixel_count(guard_violations)
    post_clean_violation_pixels = nonzero_pixel_count(post_clean_violations)
    if post_clean_violation_pixels:
        raise RuntimeError(
            "shifted body still has visible pixels inside forbidden mask "
            f"after cleanup: {post_clean_violation_pixels}"
        )

    print(f"wrote {args.out} ({frame_count} frames)")
    print(f"wrote {args.preview}")
    print(f"timing: {frame_count - 1} motion frames @ {FRAME_DURATION_MS}ms + 1 pause @ {PAUSE_DURATION_MS}ms")
    print(f"forbidden pixels checked: {forbidden_pixels}")
    print(f"shifted body pixels erased from forbidden zone: {violation_pixels}")
    print("post-clean shifted body pixels in forbidden zone: 0")


if __name__ == "__main__":
    main()
