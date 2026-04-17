from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_stem_map(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def draw_label_bar(width: int, label: str, font: ImageFont.ImageFont, bar_h: int = 28) -> Image.Image:
    bar = Image.new("RGB", (width, bar_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (bar_h - th) // 2 - bbox[1]
    draw.text((x, y), label, fill=(20, 20, 20), font=font)
    return bar


def labeled_panel(img_path: Path, label: str, panel_size: int, font: ImageFont.ImageFont) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    bar = draw_label_bar(panel_size, label, font=font, bar_h=28)
    out = Image.new("RGB", (panel_size, panel_size + bar.height), color=(255, 255, 255))
    out.paste(bar, (0, 0))
    out.paste(img, (0, bar.height))
    return out


def ensure_paths_exist(paths: Iterable[Path]):
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")


def main():
    parser = argparse.ArgumentParser("Make labeled horizontal strips for socalfire samples")
    parser.add_argument(
        "--pre_dir",
        type=str,
        default="/home/sjq/concept_sliders/datasets/remote/socalfire/pre",
    )
    parser.add_argument(
        "--post_dir",
        type=str,
        default="/home/sjq/concept_sliders/datasets/remote/socalfire/post",
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default="/home/sjq/concept_sliders/outputs/refined/socalfire-train/eval",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sjq/concept_sliders/outputs/eval/socalfire-train",
    )
    parser.add_argument("--panel_size", type=int, default=256)
    args = parser.parse_args()

    pre_dir = Path(args.pre_dir).expanduser().resolve()
    post_dir = Path(args.post_dir).expanduser().resolve()
    eval_root = Path(args.eval_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    scale_dirs = {
        "scale0": eval_root / "scale0",
        "scale0.25": eval_root / "scale0.25",
        "scale0.5": eval_root / "scale0.5",
        "scale0.75": eval_root / "scale0.75",
        "scale1.0": eval_root / "scale1",
    }

    ensure_paths_exist([pre_dir, post_dir, *scale_dirs.values()])
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_map = list_stem_map(pre_dir)
    post_map = list_stem_map(post_dir)
    scale_maps = {k: list_stem_map(v) for k, v in scale_dirs.items()}

    common = set(pre_map) & set(post_map)
    for m in scale_maps.values():
        common &= set(m)
    common_stems = sorted(common, key=lambda x: (len(x), x))

    if not common_stems:
        raise ValueError("No common stems across pre/post and all scale folders.")

    font = ImageFont.load_default()
    print(f"Found {len(common_stems)} matched stems.")
    print(f"Saving strips to: {out_dir}")

    for i, stem in enumerate(common_stems, start=1):
        panels = [
            labeled_panel(pre_map[stem], "pre", args.panel_size, font),
            labeled_panel(scale_maps["scale0"][stem], "scale=0", args.panel_size, font),
            labeled_panel(scale_maps["scale0.25"][stem], "scale=0.25", args.panel_size, font),
            labeled_panel(scale_maps["scale0.5"][stem], "scale=0.5", args.panel_size, font),
            labeled_panel(scale_maps["scale0.75"][stem], "scale=0.75", args.panel_size, font),
            labeled_panel(scale_maps["scale1.0"][stem], "scale=1.0", args.panel_size, font),
            labeled_panel(post_map[stem], "post", args.panel_size, font),
        ]

        w = sum(im.width for im in panels)
        h = panels[0].height
        strip = Image.new("RGB", (w, h), color=(255, 255, 255))
        x = 0
        for p in panels:
            strip.paste(p, (x, 0))
            x += p.width

        strip.save(out_dir / f"{stem}.png")
        if i % 100 == 0 or i == len(common_stems):
            print(f"[{i}/{len(common_stems)}] done")

    print("All strips generated.")


if __name__ == "__main__":
    main()
