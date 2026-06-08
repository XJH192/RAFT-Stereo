# RAFT-Stereo + LoFTR 前端增强版

这是一个基于 RAFT-Stereo 的双目视差推理项目。我在原始项目上加入了一个可选的 LoFTR 前端，用来在 RAFT 细化之前提供更好的初始匹配；同时新增了 baseline / LoFTR 对比脚本，方便你直接看效果有没有变好。

## 这个版本做了什么

- 给 RAFT-Stereo 接入了可选的 dense frontend
- 支持 LoFTR 作为前端初始化
- 大图推理时会对 LoFTR 输入自动缩放，减少 OOM 风险
- 新增 baseline vs LoFTR 的对比可视化脚本
- 统一了 demo / evaluate / train 的 checkpoint 加载方式
- 环境文件里补上了 `kornia` 依赖

## 环境

推荐直接创建我这版环境：

```bash
conda env create -f environment_cuda11.yaml
conda activate raftstereo
```

如果你已经有环境，至少确认下面这个包在：

```bash
pip install kornia==0.6.12
```

## 数据和权重

默认约定如下：

- Middlebury 数据集：`datasets/Middlebury/MiddEval3/...`
- 预训练权重：`models/*.pth`

如果你要跑 Middlebury 的示例，确保目录里有对应的 `im0.png`、`im1.png` 和 `disp0GT.pfm`。

## 原版推理：不带 LoFTR

不加任何前端参数，就是原版 RAFT-Stereo 推理。

```bash
cd /home/xjh66/PycharmProjects/RAFT-Stereo

python demo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --corr_implementation alt \
  --mixed_precision \
  -l datasets/Middlebury/MiddEval3/testF/*/im0.png \
  -r datasets/Middlebury/MiddEval3/testF/*/im1.png
```

如果你想跑验证集评估：

```bash
python evaluate_stereo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --dataset middlebury_H \
  --corr_implementation alt \
  --mixed_precision \
  --valid_iters 12
```

## LoFTR 前端推理

启用 LoFTR 前端时，加上下面这两个参数：

- `--use_dense_frontend`
- `--dense_frontend_type loftr`

示例：

```bash
cd /home/xjh66/PycharmProjects/RAFT-Stereo

python demo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --corr_implementation alt \
  --mixed_precision \
  --use_dense_frontend \
  --dense_frontend_type loftr \
  -l datasets/Middlebury/MiddEval3/testF/*/im0.png \
  -r datasets/Middlebury/MiddEval3/testF/*/im1.png
```

如果要看验证指标：

```bash
python evaluate_stereo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --dataset middlebury_H \
  --corr_implementation alt \
  --mixed_precision \
  --valid_iters 12 \
  --use_dense_frontend \
  --dense_frontend_type loftr
```

## baseline vs LoFTR 对比可视化

我额外加了一个对比脚本，可以一次跑两份结果：

- 原版 RAFT-Stereo
- LoFTR 前端版

并输出：

- baseline disparity 图
- LoFTR disparity 图
- 两者差异图
- `summary.csv`
- `manifest.json`

示例：

```bash
cd /home/xjh66/PycharmProjects/RAFT-Stereo

python compare_loftr.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --left_imgs datasets/Middlebury/MiddEval3/trainingH/Teddy/im0.png \
  --right_imgs datasets/Middlebury/MiddEval3/trainingH/Teddy/im1.png \
  --gt_disparities datasets/Middlebury/MiddEval3/trainingH/Teddy/disp0GT.pfm \
  --mixed_precision \
  --valid_iters 12
```

输出目录默认是 `compare_output/`，里面会有：

- `baseline/`
- `loftr/`
- `compare/`
- `summary.csv`

## 怎么判断效果有没有变好

重点看这几类东西：

- **误差指标**：`MAE / Bad1 / Bad3` 越小越好
- **可视化图**：边界、薄结构、弱纹理区域有没有更稳定
- **差异图**：LoFTR 和 baseline 的差值大不大
- **速度**：LoFTR 会增加推理时间

如果没有 GT，也可以先看对比图和差异图；如果有 GT，就直接看 `summary.csv`。

## 项目结构说明

- `core/raft_stereo.py`：RAFT-Stereo 主前向
- `core/dense_matcher.py`：LoFTR / dense frontend
- `demo.py`：单张/成对图片推理
- `evaluate_stereo.py`：验证集评估
- `compare_loftr.py`：baseline vs LoFTR 可视化对比

## 备注

- 不加 `--use_dense_frontend` 就是原版行为
- LoFTR 适合弱纹理、重复纹理、边缘不清楚的场景，但会更慢
- 大图会先缩放再做 LoFTR 匹配，避免显存一下子爆掉
- 如果你换了新环境，记得确认 `kornia` 可用
