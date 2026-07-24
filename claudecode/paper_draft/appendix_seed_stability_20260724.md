# appendices/02_experimental_details.tex "Inference-time stability" — tab:app_seed_stability CAS is the 4th distinct leaked-classifier instance found this session

## What was flagged

User noticed `tab:app_seed_stability`'s three seeds (42/43/44) all show CAS
(0.9980/0.9959/0.9959) about 0.008-0.010 *above* the paper's own cross-referenced
clean baseline (0.9878, confirmed consistent across Table 1, `tab:app_refined_scale`,
and Table 2), while LPIPS/PSNR for the same three seeds are all slightly *worse* than
that baseline (0.2197-0.2204 vs 0.2166; 23.69-23.71 vs 23.8062). A single pipeline
scored by one consistent classifier shouldn't show CAS parked uniformly above the
reference while every other metric sits uniformly below it — that's the signature of
a different classifier, not seed noise, which is exactly the pattern already found
three times this session (Table 1, Table 2, Cross-Category).

## Investigation

**1) Source**: `outputs/stability/socalfire_seed{42,43,44}/eval/socalfire-infered_metrics.csv`
(and sibling `.json`), generated 2026-05-04 via the standard
`eval/socalfire_infered_eval_metrics.py` (same script used everywhere else in the
project — no custom eval code for this table). `infer_root` for each is
`outputs/stability/socalfire_seed{N}/refined`. No wrapper `.sh` survives and the
script doesn't echo its own argv, so the classifier checkpoint wasn't directly
recorded — had to be identified empirically (see next point).

**2) Classifier — confirmed leaked, a 4th distinct checkpoint from the three already
found this session**. Re-scored all three seeds' actual saved images
(`outputs/stability/socalfire_seed{N}/refined/test/scale1`, still on disk, n=246 each)
against four candidate checkpoints on `cuda:1`. Exact matches to the published table,
digit for digit, for all three seeds:

| Seed | published CAS | `socalfire_cls_real_fresh` | `socalfire_cls_clean_split` | `socalfire_cls_20260421_201243` | `_shared_real_classifier` |
|---|---|---|---|---|---|
| 42 | 0.9980 | **0.9979674796747967** (match) | 0.9878048780487805 | 0.9939024390243902 | 0.991869918699187 |
| 43 | 0.9959 | **0.9959349593495935** (match) | 0.9857723577235772 | 0.9939024390243902 | 0.9898373983739838 |
| 44 | 0.9959 | **0.9959349593495935** (match) | 0.9878048780487805 | — | — |

The table was scored with `output-models/classifier/socalfire_cls_real_fresh/best.pt`
— not any of the three checkpoints already implicated in Table 1/2/Cross-Category.
This one is confirmed leaked by the repo's own bug-fix code, not just inferred: both
`claudecode/code/classifier_fix/train_classifier_clean_split.py`'s docstring ("for the
production checkpoint `output-models/classifier/socalfire_cls_real_fresh/best.pt`,
`--test_dir` was `datasets/remote/socalfire/test`, i.e. the EXACT test split reused
throughout the paper... checkpoint selection therefore directly optimized against the
same 246 pairs the paper later reports metrics on") and
`claudecode/code/classifier_fix/recompute_table1_cas.py` (labels it explicitly:
`"leaked (original, test_acc-selected)"`) identify it as leaked via the same
`classifier/train.py:326` bug as every other instance found this session. It is also,
separately, the semantic teacher ψ used to train the production refiner itself
(`unified_refiner_scale1stronger`) — already flagged as a distinct, lower-urgency,
not-yet-applied item in `server_audit_20260724.md`, unrelated to this table's fix.

**3) Generation/refiner pipeline — confirmed current production (`refine-2`), not the
old `refine-1` pipeline**. This matters because the Cross-Category finding
(`appendix_cross_category_tables_20260724.md`) showed the *same symptom* can also come
from a stale generation run, not just a stale classifier — needed to rule that out
here specifically. Ran a direct LPIPS comparison between seed 42's actual saved images
and both candidate source pipelines (60 test images, `cuda:1`): seed 42 vs current
production `outputs/refine-2/socalfire/test/scale1` → mean LPIPS **0.0100**; seed 42 vs
old `outputs/refine-1/socalfire/test/scale1` → mean LPIPS **0.0459** (4.6x larger).
Seed-to-seed noise on a shared pipeline should be small; 0.0459 is categorically
larger, consistent with `refine-1` being visually a different generator/refiner
checkpoint altogether (as already established for the Cross-Category finding). The
seed-stability run is confirmed same-pipeline-family as production, just genuine
per-seed generation noise — **this table's LPIPS/SSIM/PSNR numbers do not need to
change, and don't need explaining away; that variation is the real, intended content
of a seed-stability table.**

**4) Clean-classifier rescore — done, no new generation needed** (already saved to
`claudecode/result/classifier_fix/seed_stability_clean/seed_stability_clean_cas.csv`):

| Seed | LPIPS | SSIM | PSNR | CAS Acc. (clean) |
|---|---|---|---|---|
| 42 | 0.2197 | 0.5934 | 23.71 | 0.9878 |
| 43 | 0.2202 | 0.5933 | 23.69 | 0.9858 |
| 44 | 0.2204 | 0.5931 | 23.69 | 0.9878 |
| **mean ± std** | 0.2201 ± 0.0004 | 0.5933 ± 0.0002 | 23.69 ± 0.01 | **0.9871 ± 0.0012** |

With the clean classifier, CAS now sits *below* (not above) the cross-referenced
baseline of 0.9878 on average, consistent with 42 and 44 landing exactly on 0.9878 and
43 dipping slightly to 0.9858 — the same direction and rough magnitude as the other
metrics' seed-to-seed spread, i.e. now behaves like genuine noise around a single
number instead of being uniformly offset from it. This directly resolves the
"CAS moves opposite to LPIPS/PSNR" red flag that triggered the investigation.

**5) Not applicable** — this is a real instance of the classifier-leak bug, not pure
seed variance. Point 5's fallback (keep original numbers) does not apply.

## Recommended fix

Replace `tab:app_seed_stability`'s CAS column with the clean-classifier values above
(LPIPS/SSIM/PSNR columns are unaffected and correct as published). No new
generation/inference needed — only the classifier-scoring step was rerun, on already-
existing images, on `cuda:1` (the training job on `cuda:0` was left untouched
throughout).
