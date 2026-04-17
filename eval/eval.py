import argparse
import csv
import json
import math
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import models, transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_pil(path):
    return Image.open(path).convert("RGB")


def pil_to_np_uint8(img):
    return np.array(img).astype(np.uint8)


def pil_to_tensor_lpips(img):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img).unsqueeze(0)


def build_classifier(arch: str):
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, 2)
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, 2)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model


def load_classifier(ckpt_path, arch, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_classifier(arch)

    # 兼容几种常见 checkpoint 格式
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def classifier_transform(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_post_prob(model, img, tf, device):
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs[0, 1].item()   # class 1 = post


def paired_image_names(folder: Path):
    return sorted([
        p.name for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def gen_subdir_for_scale(gen_root: Path, s: float, layout: str) -> Path:
    if layout == "gen_s":
        return gen_root / f"gen_s_{s:.2f}"
    if layout == "scale":
        return gen_root / f"scale{s:g}"
    raise ValueError(f"Unknown gen_layout: {layout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_post_dir", type=str, required=True)
    parser.add_argument("--gen_root", type=str, required=True)
    parser.add_argument(
        "--gen_layout",
        type=str,
        choices=["gen_s", "scale"],
        default="gen_s",
        help='Subfolder naming: "gen_s" -> gen_s_0.25; "scale" -> scale0.25 (uses scale{s:g}).',
    )
    parser.add_argument("--s_list", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--classifier_arch", type=str, default="mobilenet_v3_small")
    parser.add_argument("--classifier_img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="If set, write metrics.json and metrics.csv here.",
    )
    parser.add_argument(
        "--metric_size",
        type=int,
        default=256,
        help="Resize both GT and generated images to this square size before metrics (0 = no resize).",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt_post_dir = Path(args.gt_post_dir)
    gen_root = Path(args.gen_root)
    s_values = [float(x) for x in args.s_list.split(",")]

    # LPIPS
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    # classifier
    clf = load_classifier(args.classifier_ckpt, args.classifier_arch, device)
    clf_tf = classifier_transform(args.classifier_img_size)

    names = paired_image_names(gt_post_dir)

    print("| s | LPIPS↓ | SSIM↑ | PSNR↑ | post prob↑ | n |")
    print("|---:|---:|---:|---:|---:|---:|")

    rows_out = []

    for s in s_values:
        folder = gen_subdir_for_scale(gen_root, s, args.gen_layout)
        lpips_vals = []
        ssim_vals = []
        psnr_vals = []
        post_probs = []

        for name in names:
            gt_path = gt_post_dir / name
            gen_path = folder / name
            if not gen_path.exists():
                continue

            gt_img = load_pil(gt_path)
            gen_img = load_pil(gen_path)
            if args.metric_size and args.metric_size > 0:
                sz = (args.metric_size, args.metric_size)
                if gt_img.size != sz:
                    gt_img = gt_img.resize(sz, Image.Resampling.LANCZOS)
                if gen_img.size != sz:
                    gen_img = gen_img.resize(sz, Image.Resampling.LANCZOS)
            elif gen_img.size != gt_img.size:
                gen_img = gen_img.resize(gt_img.size, Image.Resampling.LANCZOS)

            # LPIPS
            gt_t = pil_to_tensor_lpips(gt_img).to(device)
            gen_t = pil_to_tensor_lpips(gen_img).to(device)
            lp = lpips_fn(gen_t, gt_t).item()
            lpips_vals.append(lp)

            # SSIM / PSNR
            gt_np = pil_to_np_uint8(gt_img)
            gen_np = pil_to_np_uint8(gen_img)

            ssim = structural_similarity(
                gt_np, gen_np,
                channel_axis=2,
                data_range=255
            )
            psnr = peak_signal_noise_ratio(gt_np, gen_np, data_range=255)
            ssim_vals.append(ssim)
            psnr_vals.append(psnr)

            # post probability
            prob_post = predict_post_prob(clf, gen_img, clf_tf, device)
            post_probs.append(prob_post)

        mean_lpips = np.mean(lpips_vals) if lpips_vals else math.nan
        mean_ssim = np.mean(ssim_vals) if ssim_vals else math.nan
        mean_psnr = np.mean(psnr_vals) if psnr_vals else math.nan
        mean_post = np.mean(post_probs) if post_probs else math.nan
        n_matched = len(lpips_vals)

        print(
            f"| {s:.2f} | {mean_lpips:.4f} | {mean_ssim:.4f} | {mean_psnr:.4f} | "
            f"{mean_post:.4f} | {n_matched} |"
        )
        rows_out.append(
            {
                "s": s,
                "gen_subdir": str(folder),
                "mean_lpips": mean_lpips,
                "mean_ssim": mean_ssim,
                "mean_psnr": mean_psnr,
                "mean_post_prob": mean_post,
                "n_matched": n_matched,
                "n_gt": len(names),
            }
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "gt_post_dir": str(gt_post_dir.resolve()),
            "gen_root": str(gen_root.resolve()),
            "gen_layout": args.gen_layout,
            "metric_size": args.metric_size,
            "s_list": s_values,
            "classifier_ckpt": str(Path(args.classifier_ckpt).resolve()),
            "classifier_arch": args.classifier_arch,
            "rows": rows_out,
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "s",
                    "mean_lpips",
                    "mean_ssim",
                    "mean_psnr",
                    "mean_post_prob",
                    "n_matched",
                    "n_gt",
                    "gen_subdir",
                ]
            )
            for r in rows_out:
                w.writerow(
                    [
                        r["s"],
                        r["mean_lpips"],
                        r["mean_ssim"],
                        r["mean_psnr"],
                        r["mean_post_prob"],
                        r["n_matched"],
                        r["n_gt"],
                        r["gen_subdir"],
                    ]
                )
        print(f"Wrote {out_dir / 'metrics.json'} and {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()