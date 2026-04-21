# Unified Refiner 数据准备与完整流程

下面这套流程对应 4 个脚本：

- `scale0_only_refiner.py`
- `scale1_only_refiner.py`
- `build_pseudo_targets.py`
- `unified_scale_refiner.py`

目标是最终得到 **一个统一模型**，以后所有 `scale0 / scale0.25 / scale0.5 / scale0.75 / scale1` 都走同一个 refine 方法。

---

## 1. 你最终在训练什么

统一模型学的是：

\[
f(gen_s, pre, s) \rightarrow refined_s
\]

其中：

- `gen_s`：slider 在某个 scale 下生成的图
- `pre`：灾前图
- `s`：当前 scale 值
- `refined_s`：这张图经过 refiner 后的结果

这就是为什么统一模型 **必须输入 scale**。否则模型不知道这张图应该被修到多接近 pre 或 post。

---

## 2. 目录结构准备

建议目录整理成下面这样：

```text
YOUR_PROJECT/
├── data/
│   ├── pre/
│   │   ├── xxx.png
│   │   ├── yyy.png
│   │   └── ...
│   ├── post/
│   │   ├── xxx.png
│   │   ├── yyy.png
│   │   └── ...
│   └── scales/
│       ├── scale0/
│       │   ├── xxx.png
│       │   └── ...
│       ├── scale0.25/
│       │   ├── xxx.png
│       │   └── ...
│       ├── scale0.5/
│       │   ├── xxx.png
│       │   └── ...
│       ├── scale0.75/
│       │   ├── xxx.png
│       │   └── ...
│       └── scale1/
│           ├── xxx.png
│           └── ...
├── pseudo_targets/
│   ├── scale0/
│   ├── scale0.25/
│   ├── scale0.5/
│   ├── scale0.75/
│   └── scale1/
├── ckpts/
│   ├── scale0_teacher/
│   ├── scale1_teacher/
│   └── unified_refiner/
└── outputs/
```

### 文件名要求
所有对应图片必须 **同 stem**，例如：

- `pre/abc.png`
- `post/abc.png`
- `scales/scale0/abc.png`
- `scales/scale0.25/abc.png`
- `scales/scale0.5/abc.png`
- `scales/scale0.75/abc.png`
- `scales/scale1/abc.png`

必须一一对应，否则脚本会自动跳过不匹配的 stem。

---

## 3. 第一步：训练 scale1 teacher


输入：

- `scale1_gen`
- `pre`

目标：

- `post`

示例：

```bash
python scale1_only_refiner.py train \
  --pre_dir /path/to/data/pre \
  --post_dir /path/to/data/post \
  --scale1_dir /path/to/data/scales/scale1 \
  --save_dir /path/to/ckpts/scale1_teacher
```

训练完成后，重点 checkpoint：

```text
/path/to/ckpts/scale1_teacher/best.pt
```

---

## 4. 第二步：训练 scale0 teacher

新增脚本 `scale0_only_refiner.py` 是对称版本。

输入：

- `scale0_gen`
- `pre`

目标：

- `pre`

示例：

```bash
python scale0_only_refiner.py train \
  --pre_dir /path/to/data/pre \
  --scale0_dir /path/to/data/scales/scale0 \
  --save_dir /path/to/ckpts/scale0_teacher
```

训练完成后，重点 checkpoint：

```text
/path/to/ckpts/scale0_teacher/best.pt
```

---

## 5. 第三步：为所有 scale 生成伪标签

这一步是关键。

你没有真实的 `scale0.25 / scale0.5 / scale0.75` GT，所以先用两个 teacher 造 pseudo target。

### 脚本输入

- `pre_dir`
- `scale_root`：包含所有 `scale*` 子目录
- `scale0_checkpoint`
- `scale1_checkpoint`
- `output_root`

### 脚本干了什么
对于任意一张 `gen_s`：

1. 用 `scale0_teacher` 推出一个“往 pre 修”的 residual
2. 用 `scale1_teacher` 推出一个“往 post 修”的 residual
3. 按 scale 融合：

\[
r_s = w_{pre} r_{pre} + w_{post} r_{post}
\]

默认线性权重：

\[
w_{pre}=1-s, \quad w_{post}=s
\]

如果你设 `--gamma > 1`，靠近端点时会更强调端点效果。

最后：

\[
pseudo_s = clamp(gen_s + \alpha r_s)
\]

其中：

- `alpha`：总修正强度，默认 1.0
- `gamma`：端点强调系数，默认 1.0

### 推荐第一次先这样跑

```bash
python build_pseudo_targets.py \
  --pre_dir /path/to/data/pre \
  --scale_root /path/to/data/scales \
  --scale0_checkpoint /path/to/ckpts/scale0_teacher/best.pt \
  --scale1_checkpoint /path/to/ckpts/scale1_teacher/best.pt \
  --output_root /path/to/pseudo_targets \
  --include_endpoints \
  --alpha 1.0 \
  --gamma 1.0
```

生成后你会得到：

```text
pseudo_targets/
├── scale0/
├── scale0.25/
├── scale0.5/
├── scale0.75/
└── scale1/
```

其中中间 scale 是伪标签，端点如果加了 `--include_endpoints`，也会把 teacher refine 的结果一起存进去，方便你统一查看。

### 如果中间图修得太猛
把 `alpha` 降到 0.8 或 0.7。

### 如果想让靠近 0 和 1 的图更明显偏向端点
把 `gamma` 设成 1.5 或 2.0。

---

## 6. 第四步：训练统一 unified refiner

统一模型输入：

- `gen_s`（任意 scale）
- `pre`
- `scale_map(s)`（全图 broadcast 的 1 通道）

目标：

- `s=0` 时：`pre`
- `s=1` 时：`post`
- 中间 scale：`pseudo_targets/scaleX/*.png`

### 训练命令

```bash
python unified_scale_refiner.py train \
  --pre_dir /path/to/data/pre \
  --post_dir /path/to/data/post \
  --scale_root /path/to/data/scales \
  --pseudo_root /path/to/pseudo_targets \
  --save_dir /path/to/ckpts/unified_refiner
```

训练完成后，重点 checkpoint：

```text
/path/to/ckpts/unified_refiner/best.pt
```

---

## 7. 第五步：推理怎么做

这是你最关心的部分。

统一模型训练完以后，**以后不再需要两个 teacher**。

对任意一个 scale 目录，直接这样 refine：

```bash
python unified_scale_refiner.py refine \
  --checkpoint /path/to/ckpts/unified_refiner/best.pt \
  --pre_dir /path/to/data/pre \
  --input_dir /path/to/data/scales/scale0.5 \
  --output_dir /path/to/outputs/refined_scale0.5
```

如果 `input_dir` 名字本身是 `scale0.5`，脚本会自动解析出 `s=0.5`。

如果你的目录名不是标准格式，就手动给：

```bash
python unified_scale_refiner.py refine \
  --checkpoint /path/to/ckpts/unified_refiner/best.pt \
  --pre_dir /path/to/data/pre \
  --input_dir /path/to/some/custom_dir \
  --output_dir /path/to/outputs/custom \
  --scale_value 0.5
```

### 端点也是同样推

#### refine scale0

```bash
python unified_scale_refiner.py refine \
  --checkpoint /path/to/ckpts/unified_refiner/best.pt \
  --pre_dir /path/to/data/pre \
  --input_dir /path/to/data/scales/scale0 \
  --output_dir /path/to/outputs/refined_scale0
```

#### refine scale1

```bash
python unified_scale_refiner.py refine \
  --checkpoint /path/to/ckpts/unified_refiner/best.pt \
  --pre_dir /path/to/data/pre \
  --input_dir /path/to/data/scales/scale1 \
  --output_dir /path/to/outputs/refined_scale1
```

所以 **推理阶段永远是同一个模型、同一种方式、只换 scale 值**。

---

## 8. 你在训练什么，一句话版

### teacher 训练
- `scale0_teacher` 学：`scale0 -> pre`
- `scale1_teacher` 学：`scale1 -> post`

### pseudo label 生成
- 用两端 teacher 给中间 scale 合成监督目标

### unified model 训练
- 学：`(gen_s, pre, s) -> refined_s`

### unified model 推理
- 任意 scale 都走同一个模型

---

## 9. 推荐的完整执行顺序

```bash
# 1) 训练 scale1 teacher
python scale1_only_refiner.py train \
  --pre_dir /path/to/data/pre \
  --post_dir /path/to/data/post \
  --scale1_dir /path/to/data/scales/scale1 \
  --save_dir /path/to/ckpts/scale1_teacher

# 2) 训练 scale0 teacher
python scale0_only_refiner.py train \
  --pre_dir /path/to/data/pre \
  --scale0_dir /path/to/data/scales/scale0 \
  --save_dir /path/to/ckpts/scale0_teacher

# 3) 生成所有 scale 的 pseudo targets
python build_pseudo_targets.py \
  --pre_dir /path/to/data/pre \
  --scale_root /path/to/data/scales \
  --scale0_checkpoint /path/to/ckpts/scale0_teacher/best.pt \
  --scale1_checkpoint /path/to/ckpts/scale1_teacher/best.pt \
  --output_root /path/to/pseudo_targets \
  --include_endpoints \
  --alpha 1.0 \
  --gamma 1.0

# 4) 训练统一模型
python unified_scale_refiner.py train \
  --pre_dir /path/to/data/pre \
  --post_dir /path/to/data/post \
  --scale_root /path/to/data/scales \
  --pseudo_root /path/to/pseudo_targets \
  --save_dir /path/to/ckpts/unified_refiner

# 5) 用统一模型 refine 某个 scale
python unified_scale_refiner.py refine \
  --checkpoint /path/to/ckpts/unified_refiner/best.pt \
  --pre_dir /path/to/data/pre \
  --input_dir /path/to/data/scales/scale0.5 \
  --output_dir /path/to/outputs/refined_scale0.5
```

---

## 10. 实验建议

### 第一轮最稳配置
- `alpha = 1.0`
- `gamma = 1.0`
- unified `residual_scale = 0.50`

### 如果中间 scale 太像 post
- 降低 pseudo 生成时的 `alpha`
- 或 unified 训练时降低 `residual_scale`
- 或提高 `scale0_teacher` 的质量

### 如果中间 scale 变化太弱
- 提高 `alpha`
- 或把 `gamma` 提大到 1.5
- 或提高 unified `lambda_lpips_mid`

### 如果端点很好但中间不自然
- 优先检查 pseudo targets 本身
- 先把 `pseudo_targets/scale0.25, scale0.5, scale0.75` 可视化看看
- 不要先怀疑 unified model

---

## 11. 最关键的理解

你之前一直在问：

- 训练时到底在训什么？
- 推理时到底怎么推？

现在可以记成：

### 训练时
统一模型在学习：

> 给我 `gen_s + pre + s`，我输出这个 scale 下应该有的 refined 图。

### 推理时
统一模型在执行：

> 读入一张当前 scale 的生成图，读入 pre，告诉模型当前 scale 是多少，然后直接输出 refine 结果。

这就是整件事的核心。
