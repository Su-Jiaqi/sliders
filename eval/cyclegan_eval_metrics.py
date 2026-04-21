from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lpips
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import models, transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def list_image_map(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def load_pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_pil(img: Image.Image, size: int) -> Image.Image:
    if size <= 0:
        return img
    return img.resize((size, size), Image.Resampling.LANCZOS)


def pil_to_uint8_np(img: Image.Image, size: int) -> np.ndarray:
    return np.array(resize_pil(img, size), dtype=np.uint8)


def pil_to_lpips_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tf(img).unsqueeze(0)


def collect_cyclegan_triplets(
    pred_root: Path, data_root: Path, split: str
) -> List[Tuple[str, Path, Path, Path]]:
    pred_dir = pred_root / f"{split}_latest" / "images"
    pre_dir = data_root / split / "pre"
    post_dir = data_root / split / "post"

    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing CycleGAN image dir: {pred_dir}")
    if not pre_dir.exists():
        raise FileNotFoundError(f"Missing pre dir: {pre_dir}")
    if not post_dir.exists():
        raise FileNotFoundError(f"Missing post dir: {post_dir}")

    pre_map = list_image_map(pre_dir)
    post_map = list_image_map(post_dir)

    triplets: List[Tuple[str, Path, Path, Path]] = []
    for p in sorted(pred_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if not p.name.endswith("_fake_B.png"):
            continue
        stem = p.name[: -len("_fake_B.png")]
        if stem in pre_map and stem in post_map:
            triplets.append((stem, p, post_map[stem], pre_map[stem]))
    return triplets


def compute_lpips_ssim_psnr(
    triplets: List[Tuple[str, Path, Path, Path]], metric_size: int, device: torch.device
) -> Dict[str, float]:
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    lp_vals: List[float] = []
    ssim_vals: List[float] = []
    psnr_vals: List[float] = []

    with torch.no_grad():
        for _, pred_path, gt_path, _ in triplets:
            pred_img = load_pil_rgb(pred_path)
            gt_img = load_pil_rgb(gt_path)

            pred_t = pil_to_lpips_tensor(pred_img, metric_size).to(device)
            gt_t = pil_to_lpips_tensor(gt_img, metric_size).to(device)
            lp_vals.append(float(lpips_model(pred_t, gt_t).item()))

            pred_np = pil_to_uint8_np(pred_img, metric_size)
            gt_np = pil_to_uint8_np(gt_img, metric_size)
            ssim_vals.append(float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255)))
            psnr_vals.append(float(peak_signal_noise_ratio(gt_np, pred_np, data_range=255)))

    return {
        "lpips": float(np.mean(lp_vals)) if lp_vals else math.nan,
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else math.nan,
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else math.nan,
    }


def get_inception_features(
    image_paths: List[Path], image_size: int, batch_size: int, device: torch.device
) -> np.ndarray:
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception.fc = nn.Identity()
    inception.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            x = torch.stack([tf(load_pil_rgb(p)) for p in batch_paths], dim=0).to(device)
            feats.append(inception(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid_from_features(feat_real: np.ndarray, feat_fake: np.ndarray) -> float:
    mu1 = np.mean(feat_real, axis=0)
    mu2 = np.mean(feat_fake, axis=0)
    sigma1 = np.cov(feat_real, rowvar=False)
    sigma2 = np.cov(feat_fake, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_clip_i(
    triplets: List[Tuple[str, Path, Path, Path]], image_size: int, batch_size: int, device: torch.device
) -> Dict[str, float]:
    from transformers import CLIPImageProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    sims: List[float] = []
    with torch.no_grad():
        for i in range(0, len(triplets), batch_size):
            chunk = triplets[i : i + batch_size]
            pred_imgs = [resize_pil(load_pil_rgb(p), image_size) for _, p, _, _ in chunk]
            gt_imgs = [resize_pil(load_pil_rgb(g), image_size) for _, _, g, _ in chunk]

            pred_inputs = processor(images=pred_imgs, return_tensors="pt").to(device)
            gt_inputs = processor(images=gt_imgs, return_tensors="pt").to(device)

            pred_feat = model.get_image_features(**pred_inputs)
            gt_feat = model.get_image_features(**gt_inputs)
            pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
            gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
            sims.extend((pred_feat * gt_feat).sum(dim=-1).detach().cpu().tolist())
    return {"clip_i": float(np.mean(sims)) if sims else math.nan}


def compute_dino_i(
    triplets: List[Tuple[str, Path, Path, Path]],
    image_size: int,
    batch_size: int,
    device: torch.device,
    dino_model_name: str,
) -> Dict[str, float]:
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(dino_model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(dino_model_name)

    sims: List[float] = []
    with torch.no_grad():
        for i in range(0, len(triplets), batch_size):
            chunk = triplets[i : i + batch_size]
            pred_imgs = [resize_pil(load_pil_rgb(p), image_size) for _, p, _, _ in chunk]
            gt_imgs = [resize_pil(load_pil_rgb(g), image_size) for _, _, g, _ in chunk]

            pred_inputs = processor(images=pred_imgs, return_tensors="pt").to(device)
            gt_inputs = processor(images=gt_imgs, return_tensors="pt").to(device)

            pred_out = model(**pred_inputs).last_hidden_state[:, 0, :]
            gt_out = model(**gt_inputs).last_hidden_state[:, 0, :]

            pred_out = pred_out / pred_out.norm(dim=-1, keepdim=True)
            gt_out = gt_out / gt_out.norm(dim=-1, keepdim=True)
            sims.extend((pred_out * gt_out).sum(dim=-1).detach().cpu().tolist())
    return {"dino_i": float(np.mean(sims)) if sims else math.nan}


def build_classifier(arch: str, num_classes: int = 2) -> nn.Module:
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, num_classes)
    else:
        raise ValueError(f"Unsupported classifier arch: {arch}")
    return model


def load_classifier(checkpoint_path: str, arch: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "model_state", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    model = build_classifier(arch, 2)
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model


def compute_cas(
    triplets: List[Tuple[str, Path, Path, Path]],
    classifier_ckpt: str,
    classifier_arch: str,
    classifier_img_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model = load_classifier(classifier_ckpt, classifier_arch, device)
    tf = transforms.Compose(
        [
            transforms.Resize(int(classifier_img_size * 1.15)),
            transforms.CenterCrop(classifier_img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for _, pred_path, _gt, pre_path in triplets:
        xs.append(tf(load_pil_rgb(pre_path)))
        ys.append(0)
        xs.append(tf(load_pil_rgb(pred_path)))
        ys.append(1)

    x_all = torch.stack(xs, dim=0)
    y_all = torch.tensor(ys, dtype=torch.long)

    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, x_all.size(0), batch_size):
            xb = x_all[i : i + batch_size].to(device)
            yb = y_all[i : i + batch_size].to(device)
            pred = model(xb).argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return {"cas_accuracy": correct / max(total, 1)}


def evaluate_split(
    split: str,
    pred_root: Path,
    data_root: Path,
    metric_size: int,
    feat_size: int,
    batch_size: int,
    device: torch.device,
    run_clip: bool,
    run_fid: bool,
    run_dino: bool,
    dino_model_name: str,
    classifier_ckpt: Optional[str],
    classifier_arch: str,
    classifier_img_size: int,
) -> Dict:
    triplets = collect_cyclegan_triplets(pred_root, data_root, split)
    if not triplets:
        raise ValueError(f"No matched fake_B pairs for split={split}")

    metrics: Dict[str, object] = {"split": split, "n_matched": len(triplets)}
    metrics.update(compute_lpips_ssim_psnr(triplets, metric_size, device))

    if run_fid:
        real_feats = get_inception_features([g for _, _, g, _ in triplets], feat_size, batch_size, device)
        fake_feats = get_inception_features([p for _, p, _, _ in triplets], feat_size, batch_size, device)
        metrics["fid"] = compute_fid_from_features(real_feats, fake_feats)
    else:
        metrics["fid"] = None

    if run_clip:
        try:
            metrics.update(compute_clip_i(triplets, feat_size, batch_size, device))
        except Exception as e:  # noqa: BLE001
            metrics["clip_i"] = None
            metrics["clip_error"] = str(e)
    else:
        metrics["clip_i"] = None

    if run_dino:
        try:
            metrics.update(compute_dino_i(triplets, feat_size, batch_size, device, dino_model_name))
        except Exception as e:  # noqa: BLE001
            metrics["dino_i"] = None
            metrics["dino_error"] = str(e)
    else:
        metrics["dino_i"] = None

    if classifier_ckpt:
        try:
            metrics.update(
                compute_cas(
                    triplets=triplets,
                    classifier_ckpt=classifier_ckpt,
                    classifier_arch=classifier_arch,
                    classifier_img_size=classifier_img_size,
                    batch_size=batch_size,
                    device=device,
                )
            )
        except Exception as e:  # noqa: BLE001
            metrics["cas_accuracy"] = None
            metrics["cas_error"] = str(e)
    else:
        metrics["cas_accuracy"] = None

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser("CycleGAN generation metric evaluator")
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--metric_size", type=int, default=256)
    parser.add_argument("--feat_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_clip", action="store_true")
    parser.add_argument("--run_fid", action="store_true")
    parser.add_argument("--run_dino", action="store_true")
    parser.add_argument("--dino_model_name", type=str, default="facebook/dino-vitb16")
    parser.add_argument("--classifier_ckpt", type=str, default=None)
    parser.add_argument("--classifier_arch", type=str, default="mobilenet_v3_small")
    parser.add_argument("--classifier_img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="outputs/eval/cyclegan")
    args = parser.parse_args()

    pred_root = resolve_path(args.pred_root)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    rows: List[Dict] = []
    for split in splits:
        rows.append(
            evaluate_split(
                split=split,
                pred_root=pred_root,
                data_root=data_root,
                metric_size=args.metric_size,
                feat_size=args.feat_size,
                batch_size=args.batch_size,
                device=device,
                run_clip=args.run_clip,
                run_fid=args.run_fid,
                run_dino=args.run_dino,
                dino_model_name=args.dino_model_name,
                classifier_ckpt=args.classifier_ckpt,
                classifier_arch=args.classifier_arch,
                classifier_img_size=args.classifier_img_size,
            )
        )

    payload = {
        "pred_root": str(pred_root),
        "data_root": str(data_root),
        "splits": splits,
        "metric_size": args.metric_size,
        "feat_size": args.feat_size,
        "rows": rows,
    }
    with open(output_dir / "cyclegan_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_fields = [
        "split",
        "n_matched",
        "lpips",
        "ssim",
        "psnr",
        "fid",
        "clip_i",
        "dino_i",
        "cas_accuracy",
    ]
    with open(output_dir / "cyclegan_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, None) for k in csv_fields})

    print("| split | n | LPIPS↓ | SSIM↑ | PSNR↑ | FID↓ | CLIP-I↑ | DINO-I↑ | CAS↑ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['split']} | {row['n_matched']} | "
            f"{row.get('lpips', math.nan):.4f} | {row.get('ssim', math.nan):.4f} | "
            f"{row.get('psnr', math.nan):.4f} | "
            f"{(row.get('fid') if row.get('fid') is not None else math.nan):.4f} | "
            f"{(row.get('clip_i') if row.get('clip_i') is not None else math.nan):.4f} | "
            f"{(row.get('dino_i') if row.get('dino_i') is not None else math.nan):.4f} | "
            f"{(row.get('cas_accuracy') if row.get('cas_accuracy') is not None else math.nan):.4f} |"
        )
    print(f"\nSaved: {output_dir / 'cyclegan_metrics.json'}")
    print(f"Saved: {output_dir / 'cyclegan_metrics.csv'}")


if __name__ == "__main__":
    main()
