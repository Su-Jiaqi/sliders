#!/usr/bin/env bash
# Run the full severity-validation pipeline (align ids -> S_real labels -> Tier-1
# probe -> Experiment A -> Experiment B, x2 probe backbones each) for every
# disaster category that has both raw xBD labels extracted and RiskSlider
# generated test-set outputs already on disk.
#
# Designed to run unattended in the background (nohup + disown at the call
# site) so it survives the calling shell/session going away.
set -uo pipefail

cd /home/xjtucxy/sjq/sliders
RAW_ROOT="datasets/DisasterDataset_extracted/DisasterDataset"
CKPT="output-models/classifier/socalfire_cls_real_fresh/best.pt"

# name : disaster_key : local_folder : kfold
CATEGORIES=(
  "socalfire:socal-fire:socalfire:5"
  "santarosa:santa-rosa-wildfire:santarosa:5"
  "hurricane-florence:hurricane-florence:hurricane-florence:5"
  "volcano:guatemala-volcano:volcano:3"
)

for entry in "${CATEGORIES[@]}"; do
  IFS=':' read -r NAME KEY LOCAL_FOLDER KFOLD <<< "$entry"

  echo "################################################################"
  echo "### $NAME  (key=$KEY, local=$LOCAL_FOLDER, kfold=$KFOLD)"
  echo "################################################################"

  DATA_DIR="claudecode/data/$NAME"
  RESULT_DIR="claudecode/result/$NAME"
  mkdir -p "$DATA_DIR" "$RESULT_DIR/tier1" "$RESULT_DIR/experiment_a" "$RESULT_DIR/experiment_b"

  LOCAL_ROOT="datasets/remote/$LOCAL_FOLDER"
  GEN_UNREF="outputs/infer/$LOCAL_FOLDER/test"
  GEN_REF="outputs/refine-2/$LOCAL_FOLDER/test"

  if [ ! -d "$GEN_UNREF" ] || [ ! -d "$GEN_REF" ]; then
    echo "[SKIP] $NAME: missing generated outputs ($GEN_UNREF or $GEN_REF)"
    continue
  fi

  echo "--- [1/5] align ids ---"
  python3 claudecode/code/data_prep/align_flood_ids.py \
    --raw_root "$RAW_ROOT" \
    --disaster_key "$KEY" \
    --local_root "$LOCAL_ROOT" \
    --out_csv "$DATA_DIR/id_mapping.csv" \
    --n_sample 25 --try_offsets 1,0,-1,2,-2
  if [ $? -ne 0 ]; then echo "[FAIL] $NAME: align_flood_ids"; continue; fi

  echo "--- [2/5] compute S_real ---"
  python3 claudecode/code/data_prep/compute_flood_severity_labels.py \
    --mapping_csv "$DATA_DIR/id_mapping.csv" \
    --out_csv "$DATA_DIR/severity_labels.csv"
  if [ $? -ne 0 ]; then echo "[FAIL] $NAME: compute_severity_labels"; continue; fi

  for PROBE_TAG_ARGS in "imagenet:--imagenet_only" "wildfire:--ckpt $CKPT"; do
    PROBE_TAG="${PROBE_TAG_ARGS%%:*}"
    PROBE_ARGS="${PROBE_TAG_ARGS#*:}"

    echo "--- [3/5] tier1 probe ($PROBE_TAG) ---"
    python3 claudecode/code/common/severity_probe.py \
      --severity_csv "$DATA_DIR/severity_labels.csv" \
      --images_root "$LOCAL_ROOT" \
      --kfold "$KFOLD" \
      $PROBE_ARGS \
      > "$RESULT_DIR/tier1/tier1_${PROBE_TAG}.log" 2>&1

    echo "--- [4/5] experiment A ($PROBE_TAG) ---"
    python3 claudecode/code/experiment_a/multiscale_correlation.py \
      --severity_csv "$DATA_DIR/severity_labels.csv" \
      --real_images_root "$LOCAL_ROOT" \
      --gen_unrefined_root "$GEN_UNREF" \
      --gen_refined_root "$GEN_REF" \
      --kfold "$KFOLD" \
      $PROBE_ARGS \
      --out_csv "$RESULT_DIR/experiment_a/experiment_a_results_${PROBE_TAG}_probe.csv" \
      > "$RESULT_DIR/experiment_a/experiment_a_${PROBE_TAG}.log" 2>&1

    echo "--- [5/5] experiment B ($PROBE_TAG) ---"
    python3 claudecode/code/experiment_b/group_ceiling_and_concordance.py \
      --severity_csv "$DATA_DIR/severity_labels.csv" \
      --real_images_root "$LOCAL_ROOT" \
      --gen_unrefined_root "$GEN_UNREF" \
      --gen_refined_root "$GEN_REF" \
      --kfold "$KFOLD" \
      $PROBE_ARGS \
      --out_dir "$RESULT_DIR/experiment_b" \
      > "$RESULT_DIR/experiment_b/experiment_b_${PROBE_TAG}.log" 2>&1
    mv "$RESULT_DIR/experiment_b/group_ceiling_curves.csv" "$RESULT_DIR/experiment_b/group_ceiling_curves_${PROBE_TAG}_probe.csv" 2>/dev/null
    mv "$RESULT_DIR/experiment_b/pairwise_concordance.csv" "$RESULT_DIR/experiment_b/pairwise_concordance_${PROBE_TAG}_probe.csv" 2>/dev/null
  done

  echo "--- plotting $NAME ---"
  python3 - "$NAME" <<'PYEOF'
import sys, importlib.util
from pathlib import Path

name = sys.argv[1]
result_dir = Path(f"claudecode/result/{name}")

def load_module(path, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Experiment A plot
mod_a = load_module("claudecode/code/experiment_a/plot_experiment_a.py", "plot_a")
data_imagenet = mod_a.load(result_dir / "experiment_a" / "experiment_a_results_imagenet_probe.csv")
data_wildfire = mod_a.load(result_dir / "experiment_a" / "experiment_a_results_wildfire_probe.csv")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
mod_a.plot_one(axes[0], data_imagenet, "Probe: ImageNet-only features")
mod_a.plot_one(axes[1], data_wildfire, "Probe: wildfire-finetuned ψ features")
fig.suptitle(f"Experiment A: generated severity vs real S_real — {name} test split", fontsize=13, y=1.06)
plt.tight_layout()
plt.savefig(result_dir / "experiment_a" / "experiment_a_correlation_vs_scale.png", dpi=300, bbox_inches="tight")
plt.savefig(result_dir / "experiment_a" / "experiment_a_correlation_vs_scale.pdf", bbox_inches="tight")
plt.close(fig)

# Experiment B plot
mod_b = load_module("claudecode/code/experiment_b/plot_experiment_b.py", "plot_b")
data_imagenet_b = mod_b.load(result_dir / "experiment_b" / "pairwise_concordance_imagenet_probe.csv")
data_wildfire_b = mod_b.load(result_dir / "experiment_b" / "pairwise_concordance_wildfire_probe.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
mod_b.plot_one(axes[0], data_imagenet_b, "Probe: ImageNet-only features")
mod_b.plot_one(axes[1], data_wildfire_b, "Probe: wildfire-finetuned ψ features")
fig.suptitle(f"Experiment B: pairwise rank concordance — {name} test split", fontsize=13, y=1.08)
plt.tight_layout()
plt.savefig(result_dir / "experiment_b" / "experiment_b_concordance_vs_scale.png", dpi=300, bbox_inches="tight")
plt.savefig(result_dir / "experiment_b" / "experiment_b_concordance_vs_scale.pdf", bbox_inches="tight")
plt.close(fig)

print(f"plots saved for {name}")
PYEOF

  echo "=== $NAME done ==="
done

echo "ALL CATEGORIES DONE"
