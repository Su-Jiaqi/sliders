import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================
# 默认参数（不传参数也能跑）
# =========================
DEFAULT_GPU = 2
DEFAULT_BATCH_SIZE = 32

# DEFAULT_CKPT = "/home/sjq/concept_sliders/output-models/classifier/socalfire/socalfire_cls_xbd/best.pt"
DEFAULT_DATA_DIR = "/home/sjq/concept_sliders/outputs/refined/socalfire-test/prepared-classifier-2"
# DEFAULT_SAVE_CSV = "/home/sjq/concept_sliders/outputs/classifier/xbd-generated_test-2.csv"

DEFAULT_CKPT = "/home/sjq/concept_sliders/output-models/classifier/socalfire/socalfire_cls_generated-2/best.pt"
# DEFAULT_DATA_DIR = "/home/sjq/concept_sliders/datasets/remote/socalfire/test"
# DEFAULT_SAVE_CSV = "/home/sjq/concept_sliders/outputs/classifier/generated-xbd_test-2.csv"

DEFAULT_SAVE_CSV = "/home/sjq/concept_sliders/outputs/classifier/generated-test-2.csv"
# DEFAULT_SAVE_CSV = "/home/sjq/concept_sliders/outputs/classifier/xbd=test.csv"

def build_model(arch: str, num_classes: int):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, num_classes)

    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return model


def make_fixed_class_dataset(root_dir: str, transform):
    ds = datasets.ImageFolder(root_dir, transform=transform)

    classes = set(ds.class_to_idx.keys())
    if classes != {"pre", "post"}:
        raise ValueError(f"Expected subfolders {{'pre','post'}}, got {ds.class_to_idx}")

    ds.class_to_idx = {"pre": 0, "post": 1}

    new_samples = []
    for path, _ in ds.samples:
        cls_name = os.path.basename(os.path.dirname(path))
        new_label = ds.class_to_idx[cls_name]
        new_samples.append((path, new_label))

    ds.samples = new_samples
    ds.targets = [y for _, y in new_samples]
    return ds


@torch.no_grad()
def evaluate(model, loader, device, class_to_idx, save_csv=None):
    model.eval()

    num_classes = len(class_to_idx)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    correct = 0
    total = 0

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    results = []

    sample_index = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        for i in range(len(y)):
            true_label = y[i].item()
            pred_label = pred[i].item()
            cm[true_label, pred_label] += 1

            if save_csv is not None:
                filepath = loader.dataset.samples[sample_index][0]
                results.append({
                    "filepath": filepath,
                    "true_label": idx_to_class[true_label],
                    "pred_label": idx_to_class[pred_label],
                    "prob_pre": probs[i][0].item(),
                    "prob_post": probs[i][1].item(),
                })
                sample_index += 1

    acc = correct / total

    metrics = {}
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    if save_csv is not None and len(results) > 0:
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    return acc, cm, metrics


def main():

    device = torch.device(f"cuda:{DEFAULT_GPU}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(DEFAULT_CKPT, map_location=device)

    arch = checkpoint["arch"]
    img_size = checkpoint["img_size"]
    class_to_idx = checkpoint["class_to_idx"]

    print("Loaded model:", arch)
    print("Image size:", img_size)
    print("Class mapping:", class_to_idx)

    model = build_model(arch, num_classes=2)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    test_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = make_fixed_class_dataset(DEFAULT_DATA_DIR, transform=test_tf)
    loader = DataLoader(dataset,
                        batch_size=DEFAULT_BATCH_SIZE,
                        shuffle=False)

    acc, cm, metrics = evaluate(
        model, loader, device,
        class_to_idx,
        save_csv=DEFAULT_SAVE_CSV
    )

    print("\n=== Evaluation Result ===")
    print("Accuracy:", acc)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm.numpy())

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    for c in metrics:
        name = idx_to_class[c]
        m = metrics[c]
        print(f"\nClass {name}")
        print("  Precision:", m["precision"])
        print("  Recall:", m["recall"])
        print("  F1:", m["f1"])

    print("\nSaved detailed predictions to:", DEFAULT_SAVE_CSV)


if __name__ == "__main__":
    main()