# claudecode

RiskSlider 论文"连续烈度"核心主张的独立验证实验(见对话中的 research plan)。所有脚本从仓库根目录运行。

## 目录结构

```
claudecode/
├── code/
│   ├── data_prep/        # 原始 xBD 标签对齐 + 严重度计算
│   ├── common/           # 跨实验共享的探针基础设施(特征提取、CV、bootstrap CI)
│   ├── experiment_a/     # 实验A: 生成图像终点/多尺度严重度 与 真实 S_real 的相关性
│   └── experiment_b/     # 实验B(进行中)
├── data/                 # 中间数据产物(id映射、S_real 标签)
└── result/                # 每个实验的输出(csv/png/pdf)
    ├── tier1/
    ├── experiment_a/
    └── experiment_b/
```

## 已跑通的流程(midwest-flooding pilot)

```bash
# 1. 校验 local_id = raw_tile_index + 1 假设,产出 id 映射
python claudecode/code/data_prep/align_flood_ids.py \
  --raw_root datasets/DisasterDataset_extracted/DisasterDataset \
  --disaster_key midwest-flooding \
  --local_root datasets/remote/midwest-flooding \
  --out_csv claudecode/data/flood_id_mapping.csv

# 2. 计算真实连续严重度 S_real (复用 dataselect/generate_image_level_labels.py)
python claudecode/code/data_prep/compute_flood_severity_labels.py \
  --mapping_csv claudecode/data/flood_id_mapping.csv \
  --out_csv claudecode/data/flood_severity_labels.csv

# 3. Tier-1 线性探针(严格 train-only CV,test 只碰一次)
python claudecode/code/common/severity_probe.py \
  --severity_csv claudecode/data/flood_severity_labels.csv \
  --images_root datasets/remote/midwest-flooding \
  --ckpt output-models/classifier/socalfire_cls_real_fresh/best.pt   # 或 --imagenet_only

# 4. 实验A: 对生成图像(refine前/后,所有 s)做相关性分析
python claudecode/code/experiment_a/multiscale_correlation.py \
  --severity_csv claudecode/data/flood_severity_labels.csv \
  --gen_unrefined_root outputs/infer/midwest-flooding/test \
  --gen_refined_root outputs/refine-2/midwest-flooding/test \
  --imagenet_only --out_csv claudecode/result/experiment_a/experiment_a_results_imagenet_probe.csv

python claudecode/code/experiment_a/plot_experiment_a.py
```

## 核心结论(实验A)

未 refine 的生成模块输出,在所有 s 上和真实洪水损毁严重度 `S_real` 都不显著相关(ρ≈0.17~0.23, p>0.05)。经过 semantic-aware residual refinement 后,相关性随 s 单调上升,在 s=1 时强显著(ImageNet 探针 ρ=0.511, p=3.8e-6;wildfire-微调探针 ρ=0.479, p=1.8e-5),两个独立特征骨干结果一致。详见 `result/experiment_a/`。

副产品发现:纯 ImageNet 特征比 wildfire-微调过的语义教师 ψ 在跨灾害类型(flooding)上更能读出严重度,提示 ψ 的微调可能牺牲了跨类型泛化性。
