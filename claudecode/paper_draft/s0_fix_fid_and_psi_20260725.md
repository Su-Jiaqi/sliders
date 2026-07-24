# s=0 fix: single test-split FID, and ψ's soft probability at s=0 — 2026-07-25

## FID(s=0), test split — one number, applies everywhere

Confirmed the user's reasoning before running anything further: since s=0 is now
the same 246 real x_pre images for every test-split variant/table, FID(s=0) (a
distributional distance between the "generated" set and the real-post set) is
necessarily identical across `test_direct`, `test_no_scale`, and any other
test-split variant — already empirically confirmed for CLIP-I/DINO-I
(`train_direct`==`train_refined`: 0.8806/0.7317 both), same logic applies to FID.
Computed once:

**FID(s=0, test split, n=246) = 220.856**

Applies to `tab:app_refined_scale` and `tab:app_continuous_generation_scale`'s Test
row s=0 cell (the only two tables with an FID column that include s=0 —
`tab:app_ablation_multiscale` doesn't report FID). Train split's FID(s=0) was
already computed earlier: **125.410** (`train_refined`'s run, applies to
`train_direct` too by the same argument).

## ψ (semantic teacher, Eq. 6) soft probability at s=0 — confirmed it needs updating

ψ = `output-models/classifier/socalfire_cls_real_fresh/best.pt`, confirmed via
`logs/refine-2/run_socalfire_scale1stronger.sh:14`'s `CLASSIFIER_CKPT`, passed
directly as `--classifier_checkpoint` to `refine/scale1_only_refiner_stronger.py
train` — this is the actual, load-bearing checkpoint used as the frozen semantic
teacher in Eq. 6, distinct from the CAS/PCR classifier (`socalfire_cls_clean_split`)
used everywhere else this session.

Computed mean softmax $P_\psi(\text{post})$ over the 246 test images (standard
transform: `Resize(224*1.15)+CenterCrop(224)`, same as every other classifier
evaluation this session), on `cuda:6`:

| | $P_\psi(\text{post} \mid \tilde x(0))$ |
|---|---|
| OLD (model-generated s=0) | **0.4299** — matches the paper's cited 0.430 to 3 decimals, confirms correct checkpoint/methodology |
| **NEW (x_pre passthrough)** | **0.0173** |

The paper's `tab:app_psi_prob` s=0 cell (currently 0.430) needs to change to
**0.0173**. This is expected and mechanistically sound: ψ is a well-trained
pre/post classifier, and a genuine unprocessed pre-disaster photo should register
as confidently "not post-disaster" to it — unlike the old model-generated s=0
sample, which carried enough of the pipeline's own processing signature to fool ψ
into a much less confident (0.430, near-chance) reading. This is a real,
substantial change (~25x reduction), not a rounding-level correction.

## Recommendation

Both of these are small, mechanical table-cell updates (not new methodological
concerns like the SSIM continuity finding) — safe to fold in alongside the rest of
the s=0 fix's already-decided changes.
