from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_scales(scale_str: str) -> list[float]:
    values = [x.strip() for x in scale_str.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one scale is required.")
    return [float(x) for x in values]


def scale_dir_name(scale: float) -> str:
    return f"scale{scale:g}"


def pair_key(s1: float, s2: float) -> str:
    return f"s{s1:g}_to_s{s2:g}"


def find_image_names(folder: Path) -> set[str]:
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")
    return {
        p.name for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }


def build_transform(image_size: int):
    tfms = []
    if image_size > 0:
        tfms.append(transforms.Resize((image_size, image_size)))
    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(tfms)


def load_image_tensor(path: Path, transform) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def compute_lpips_value(loss_fn, img1: torch.Tensor, img2: torch.Tensor, device: torch.device) -> float:
    return float(loss_fn(img1.to(device), img2.to(device)).item())


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def compute_cv(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if math.isclose(mean, 0.0):
        return float("nan")
    return float(arr.std() / mean)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_method_specs(method_specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in method_specs:
        if "=" not in spec:
            raise ValueError(
                f'Invalid --method "{spec}". Expected format: name=path/to/generated/root'
            )
        name, path_str = spec.split("=", 1)
        name = name.strip()
        path_str = path_str.strip()
        if not name or not path_str:
            raise ValueError(
                f'Invalid --method "{spec}". Both method name and path are required.'
            )
        parsed.append((name, Path(path_str).expanduser().resolve()))
    if not parsed:
        raise ValueError("At least one --method must be provided.")
    return parsed


def make_plot(
    consecutive_keys: list[str],
    transition_labels: list[str],
    method_stats: dict[str, dict[str, dict[str, float] | float]],
    out_path: Path,
) -> None:
    method_names = list(method_stats.keys())
    x = np.arange(len(consecutive_keys))
    width = 0.8 / max(len(method_names), 1)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    colors = ["steelblue", "lightcoral", "goldenrod", "mediumseagreen", "slateblue"]
    offsets = (np.arange(len(method_names)) - (len(method_names) - 1) / 2.0) * width
    for idx, method_name in enumerate(method_names):
        stats = method_stats[method_name]
        means = stats["means"]
        stds = stats["stds"]
        ax.bar(
            x + offsets[idx],
            [means[k] for k in consecutive_keys],
            width,
            yerr=[stds[k] for k in consecutive_keys],
            label=method_name,
            color=colors[idx % len(colors)],
            capsize=4,
        )
    ax.set_xlabel("Severity Transition")
    ax.set_ylabel("LPIPS Distance")
    ax.set_title("Perceptual Distance Between Consecutive Severity Steps")
    ax.set_xticks(x)
    ax.set_xticklabels(transition_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser("Consecutive-step LPIPS linearity analysis")
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help=(
            'Method spec in the format "name=generated_root". '
            'Example: --method ours=outputs/refine/socalfire/test'
        ),
    )
    parser.add_argument(
        "--pre_dir",
        type=str,
        default="datasets/remote/socalfire/test/pre",
        help="Ground-truth pre-disaster image directory.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0,0.25,0.5,0.75,1",
        help="Ordered severity scales used for consecutive-step analysis.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval/linearity/socalfire_method_comparison_test",
        help="Directory for raw results, summaries, and plots.",
    )
    args = parser.parse_args()

    pre_dir = Path(args.pre_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.method is None:
        method_specs = [
            "ours=outputs/refine/socalfire/test",
            "w/o pseudo=outputs/ablation/socalfire_no_pseudo/test",
            "w/o scale=outputs/ablation/socalfire_no_scale/test",
            "w/o refine=outputs/ablation/socalfire_no_refine/test",
        ]
    else:
        method_specs = args.method
    methods = parse_method_specs(method_specs)

    scales = parse_scales(args.scales)
    if len(scales) < 2:
        raise ValueError("Need at least two scales for linearity analysis.")

    consecutive_pairs = list(zip(scales[:-1], scales[1:]))
    consecutive_keys = [pair_key(s1, s2) for s1, s2 in consecutive_pairs]
    transition_labels = [f"{s1:g}->{s2:g}" for s1, s2 in consecutive_pairs]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = build_transform(args.image_size)
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    method_scale_dirs: dict[str, dict[float, Path]] = {}
    valid_names = find_image_names(pre_dir)
    for method_name, gen_root in methods:
        scale_dirs = {s: gen_root / scale_dir_name(s) for s in scales}
        for s, folder in scale_dirs.items():
            if not folder.exists():
                raise FileNotFoundError(
                    f'Missing generated folder for method "{method_name}", scale {s:g}: {folder}'
                )
        method_scale_dirs[method_name] = scale_dirs
        for folder in scale_dirs.values():
            valid_names &= find_image_names(folder)
    image_names = sorted(valid_names)

    if not image_names:
        raise RuntimeError("No matched images found across pre/generated directories.")

    method_results: dict[str, dict[str, dict[str, float]]] = {
        method_name: {} for method_name, _ in methods
    }

    for idx, name in enumerate(image_names, start=1):
        if idx % 50 == 0 or idx == len(image_names):
            print(f"Processed {idx}/{len(image_names)} images")

        for method_name, _ in methods:
            method_results[method_name][name] = {}
            scale_tensors = {
                s: load_image_tensor(method_scale_dirs[method_name][s] / name, transform)
                for s in scales
            }
            for s1, s2 in consecutive_pairs:
                method_results[method_name][name][pair_key(s1, s2)] = compute_lpips_value(
                    loss_fn,
                    scale_tensors[s1],
                    scale_tensors[s2],
                    device,
                )

    summary_methods: dict[str, dict[str, dict[str, float] | float]] = {}
    for method_name, _ in methods:
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for key in consecutive_keys:
            values = [method_results[method_name][name][key] for name in image_names]
            means[key], stds[key] = compute_mean_std(values)
        summary_methods[method_name] = {
            "means": means,
            "stds": stds,
            "cv": compute_cv([means[k] for k in consecutive_keys]),
        }

    summary = {
        "num_images": len(image_names),
        "scales": scales,
        "transitions": consecutive_keys,
        "methods": summary_methods,
    }

    for method_name, _ in methods:
        safe_name = method_name.lower().replace("/", "_").replace(" ", "_")
        save_json(out_dir / f"linearity_results_{safe_name}.json", method_results[method_name])
    save_json(out_dir / "linearity_summary.json", summary)
    make_plot(
        consecutive_keys=consecutive_keys,
        transition_labels=transition_labels,
        method_stats=summary_methods,
        out_path=out_dir / "linearity_analysis.pdf",
    )
    make_plot(
        consecutive_keys=consecutive_keys,
        transition_labels=transition_labels,
        method_stats=summary_methods,
        out_path=out_dir / "linearity_analysis.png",
    )

    for method_name, _ in methods:
        stats = summary_methods[method_name]
        means = stats["means"]
        stds = stats["stds"]
        print(f"\n{method_name} consecutive-step LPIPS:")
        for key in consecutive_keys:
            print(f"{key}: {means[key]:.4f} ± {stds[key]:.4f}")
        print(f"{method_name} CV: {stats['cv']:.4f}")

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
