from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageSequence


CANVAS_SIZE = 256
FRAME_DURATION_MS = 80


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        raise ValueError("source image has no visible pixels")
    return bbox


def fit_source(source_path: Path) -> Image.Image:
    image = Image.open(source_path).convert("RGBA")
    bird = image.crop(alpha_bbox(image))

    margin = 5
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


def make_head_mask(base: Image.Image) -> Image.Image:
    alpha = base.getchannel("A")
    shape = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(shape)

    # Keep the small bead eye, low head, and long bill nearly motionless.
    draw.ellipse((161, 47, 209, 110), fill=255)
    draw.polygon([(187, 78), (254, 113), (251, 135), (185, 98)], fill=255)
    draw.polygon([(151, 68), (198, 63), (203, 121), (151, 112)], fill=255)

    return ImageChops.multiply(alpha, shape)


def make_body_mask(base: Image.Image) -> Image.Image:
    alpha = base.getchannel("A")
    shape = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(shape)

    # The body layer overlaps the lower head/neck area. The fixed head layer
    # covers that overlap, preventing cutout gaps when the body rocks.
    draw.rectangle((0, 0, 176, CANVAS_SIZE), fill=255)
    draw.polygon([(158, 82), (205, 112), (193, 178), (150, 174)], fill=255)
    draw.rectangle((150, 118, 221, CANVAS_SIZE), fill=255)

    return ImageChops.multiply(alpha, shape)


def isolate_layers(base: Image.Image) -> tuple[Image.Image, Image.Image]:
    head_mask = make_head_mask(base)
    body_mask = make_body_mask(base)

    head = Image.new("RGBA", base.size, (0, 0, 0, 0))
    head.alpha_composite(base)
    head.putalpha(head_mask)

    body = Image.new("RGBA", base.size, (0, 0, 0, 0))
    body.alpha_composite(base)
    body.putalpha(body_mask)
    return body, head


def paste_layer(canvas: Image.Image, layer: Image.Image, dx: int, dy: int) -> None:
    bbox = alpha_bbox(layer)
    crop = layer.crop(bbox)
    canvas.alpha_composite(crop, (bbox[0] + dx, bbox[1] + dy))


def build_frames(base: Image.Image) -> list[Image.Image]:
    body, head = isolate_layers(base)
    frames: list[Image.Image] = []

    # Reference motion: the head/bill stays steady while the plump body
    # rocks forward and back with a deliberately larger "meep" shuffle.
    body_offsets = [
        (-13, 2),
        (-6, -1),
        (5, -2),
        (14, 1),
        (8, 4),
        (-3, 3),
        (-12, 1),
        (-7, -1),
    ]

    for dx, dy in body_offsets:
        frame = Image.new("RGBA", base.size, (0, 0, 0, 0))
        paste_layer(frame, body, dx, dy)
        frame.alpha_composite(head)
        frames.append(frame)
    return frames


def save_preview(frames: list[Image.Image], preview_path: Path) -> None:
    frame_w, frame_h = frames[0].size
    sheet = Image.new("RGBA", (frame_w * len(frames), frame_h), (255, 255, 255, 0))
    for index, frame in enumerate(frames):
        sheet.alpha_composite(frame, (index * frame_w, 0))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("extension/icons/woodcock-transparent.png"),
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

    source = args.source
    if not source.exists():
        source = Path("extension/icons/woodcock-128.png")

    base = fit_source(source)
    frames = build_frames(base)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        lossless=True,
        method=6,
        disposal=2,
    )
    save_preview(frames, args.preview)

    with Image.open(args.out) as animated:
        frame_count = sum(1 for _ in ImageSequence.Iterator(animated))
    print(f"wrote {args.out} ({frame_count} frames)")
    print(f"wrote {args.preview}")


if __name__ == "__main__":
    main()
