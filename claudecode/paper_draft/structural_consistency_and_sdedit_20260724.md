# Structural consistency (building footprint IoU) and SDEdit baseline — done, 2026-07-24

## Structural consistency: does the generated image preserve the correct building layout?

Pixel/semantic metrics (LPIPS/CLIP-I/CAS) can't tell whether a method hallucinates a
plausible-looking but spatially wrong building arrangement. Tests this directly using
xBD's own building-footprint polygon labels (found under
`datasets/DisasterDataset_extracted/`, pixel-space `xy` WKT polygons per image,
matched to the anonymized `datasets/remote/socalfire` test set via md5 hashing of the
underlying images — all 246 test scenes matched exactly).

**Setup**: no pretrained building-segmentation model is available on this machine (no
segmentation_models_pytorch/detectron2/mmseg; torchvision's COCO models have no
"building" class), so trained a lightweight U-Net from scratch
(`train_building_segmenter.py`, 769 xBD train pre-disaster images + rasterized
footprint masks, 654/115 train/val split, 40 epochs, best val IoU 0.568). Applied the
trained segmenter to each method's generated post-disaster test image and computed
IoU against the ground-truth footprint mask (rasterized from xBD's own post-disaster
labels — the footprint persists even for destroyed buildings, only the damage
subtype changes). Evaluated on the fixed subset of 108/246 test scenes where the
ground truth actually contains at least one building (the same 108 for every method,
so n is not method-dependent -- an earlier version of this script filtered
differently per method and gave a confounded comparison; this is the corrected one).
Script: `claudecode/code/classifier_fix/structural_consistency/eval_structural_iou.py`.
Data: `claudecode/result/classifier_fix/structural_consistency_iou.csv`.

**Caught and fixed a real bug before trusting the mask data**: the first version of
the polygon rasterizer scaled coordinates down to the target resolution *before*
drawing them onto a canvas still sized at the original resolution, squeezing all
building shapes into a small corner of the image (visually confirmed by overlaying
masks on the source photos — the bug was obvious once rendered). Fixed by drawing at
native resolution and resizing the whole canvas afterward; re-verified visually
against the source imagery before proceeding.

### Results (n=108 for every method)

| Method | mean IoU | median IoU |
|---|---|---|
| Real post-disaster (ceiling reference) | 0.2822 | 0.2832 |
| ControlNet | 0.1022 | 0.0824 |
| Pix2Pix | 0.2218 | 0.2315 |
| **CycleGAN** | **0.3644** | **0.3981** |
| Palette | 0.0134 | 0.0007 |
| SDEdit (training-free) | 0.0175 | 0.0108 |
| **RiskSlider (Ours)** | **0.2892** | **0.3190** |

**Honest interpretation, mostly favorable**: RiskSlider (0.289) is essentially at
parity with genuine real post-disaster photos scored by the same segmenter (0.282)
-- generated buildings land in the correct footprint locations about as reliably as
real photos do. RiskSlider is clearly better than ControlNet, Pix2Pix, Palette, and
SDEdit. **CycleGAN is again the one exception**, scoring meaningfully higher than
both RiskSlider and even the real-image ceiling (0.364 vs 0.289 vs 0.282) --
consistent with, and now extending to a structural-fidelity axis, the
already-established pattern that CycleGAN specifically preserves pixel-level
structure/layout better than RiskSlider (also seen in CLIP-I/DINO-I and the FID/KID
tie). Plausible explanation: CycleGAN's cycle-consistency objective directly
optimizes for structure preservation between input and output, which is exactly what
this metric rewards. **Recommended framing**: another data point for the
already-recommended honest framing that CycleGAN is RiskSlider's one genuine,
specific competitor on structure/pixel-fidelity-adjacent metrics, while RiskSlider
still wins decisively on LPIPS/SSIM/PSNR and most semantic metrics.

## SDEdit baseline (training-free)

Tests whether RiskSlider's LoRA fine-tuning + explicit severity conditioning is
actually necessary, or whether the pretrained SD v1.4 backbone's own img2img/SDEdit
editing capability gets most of the way there with zero extra training. Partially
noised each real pre-disaster test image's latent (strength=0.65, i.e. 33/50
denoising steps), then denoised with the **plain, untouched pretrained UNet** (no
LoRA, no extra conditioning channels) using the same disaster-description text
prompt via standard classifier-free guidance (scale=7.5). One endpoint-style
comparison (SDEdit has no natural continuous severity handle), evaluated with the
same Table 1 metrics using the confirmed leak-free classifier for CAS. Script:
`claudecode/code/classifier_fix/sdedit_baseline.py`. Data:
`outputs/eval/sdedit/controlnet_metrics.csv`.

| Metric | SDEdit | Palette (current Table 1 worst) |
|---|---|---|
| LPIPS↓ | 0.6803 | 0.6016 |
| SSIM↑ | 0.1289 | 0.3213 |
| PSNR↑ | 11.91 | 16.27 |
| FID↓ | 651.55 | 434.47 |
| CLIP-I↑ | 0.6653 | 0.7581 |
| DINO-I↑ | 0.1007 | 0.3983 |
| CAS↑ (clean clf) | 0.9350 | 0.9919 |
| Structural IoU (n=108) | 0.0175 | 0.0134 |

**Result: SDEdit is dramatically worse than every trained baseline in Table 1**,
including the current worst method (Palette) by a wide margin on every metric except
CAS (where SDEdit is only slightly below Palette, both near-saturated) and structural
IoU (where both are near-zero, i.e. equally bad). This is a clean, unambiguous
result, not a close call. **Recommended framing**: this directly answers the "maybe
you don't even need to fine-tune, just use the pretrained backbone's own editing
capability" critique -- on this evidence, the pretrained backbone alone (no LoRA
fine-tuning, no learned severity conditioning) produces markedly worse post-disaster
imagery across structural fidelity, distributional realism, and semantic similarity.
RiskSlider's fine-tuning step is doing real, necessary work.

## Not done: same-backbone-no-progression-conditioning control

This remains the one item from the reviewer checklist requiring a full new LoRA
training run (same SD v1.4 backbone, same data, same rank/alpha/steps, but no
severity conditioning -- i.e. an endpoint-only fine-tune). Flagging for a separate
resource-budget decision; everything else in this batch (structural IoU, SDEdit) is
done and required no new training beyond the lightweight segmenter above.
