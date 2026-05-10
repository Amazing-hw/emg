# EMG Transfer Learning

基于 Meta 开源 EMG 项目的迁移学习研究：将大规模手部姿态估计预训练模型迁移到小样本手势分类任务。

## 项目概述

本仓库包含三个项目，均基于同一款 **16 通道 EMG 腕带**（2000Hz 采样率）：

| 项目 | 来源 | 任务 | 数据规模 |
|------|------|------|----------|
| **emg_nature** | Meta Nature 2024 | 手势分类 / 手写识别 / 腕部角度 | 小样本 |
| **emg2pose1** | Meta NeurIPS 2024 | 20 DOF 手部关节角度回归 | 25K 文件, 193 人, 370 小时 |
| **emg_transfer** | 本项目 | 迁移学习：emg2pose → 手势分类 | — |

## 核心思路

```
emg2pose (大规据预训练)              emg_nature (小样本微调)
┌─────────────────────┐            ┌─────────────────────┐
│ TDS Backbone        │ ──迁移──→ │ TDS Backbone (微调)  │
│ 16→256→256→256→64  │            │ 16→256→256→256→64  │
│ 输出: 20 关节角度    │            │ + 1层LSTM(64→128)  │
│ 数据: 193人 370小时  │            │ + Linear(128→9)    │
└─────────────────────┘            │ 输出: 9类手势        │
                                   │ 数据: 100人         │
                                   └─────────────────────┘
```

### 为什么可行

- 两个项目使用**同一款 16 通道腕带**，EMG 信号物理特性一致
- TDS (Time-Depth Separable) 卷积 backbone 在 193 人、370 小时数据上预训练，学到了通用的 EMG 时空特征
- 将预训练 backbone 迁移到手势分类，只需少量标注数据微调即可

## 实验结果

| 指标 | 数值 |
|------|------|
| 最佳 val_accuracy | **52.8%** (epoch 24) |
| Test CLER | **14.1%** |
| 预训练权重 | tracking_vemg2pose.ckpt |
| 训练时间 | ~12.5 小时 (RTX 2060 6GB) |

### 训练曲线

| 阶段 | Epoch | Train Loss | Val Accuracy |
|------|-------|-----------|-------------|
| 冻结 backbone | 0 | 0.671 | 0.134 |
| | 4 | 0.015 | 0.394 |
| 解冻微调 | 5 | 0.014 | 0.457 |
| | 10 | 0.015 | 0.490 |
| | 16 | 0.011 | 0.501 |
| | 24 | 0.008 | **0.528** |

### 关键发现

1. **预训练特征可直接迁移**：仅冻结 backbone 训练 4 个 epoch，accuracy 即达 0.394
2. **解冻带来显著增益**：解冻后 accuracy 从 0.394 跃升至 0.457（+16%）
3. **收敛速度快**：24 epoch 即达到最佳，远超原始模型（需 100+ epoch）
4. **快速 plateau**：24 epoch 后不再提升，需优化学习率策略和 batch size

## 仓库结构

```
emg/
├── README.md                    ← 本文件
├── .gitignore                   ← 排除数据文件(.hdf5)、checkpoint(.ckpt)、日志
│
├── emg_nature/                  ← Meta neuromotor interface 项目
│   └── generic-neuromotor-interface-main/.../
│       ├── generic_neuromotor_interface/   ← 源码 (data, networks, lightning, train)
│       ├── config/                        ← Hydra 配置
│       └── notebooks/                     ← 数据探索和评估 notebook
│
├── emg2pose1/                   ← Meta emg2pose 项目
│   └── emg2pose-main/.../
│       ├── emg2pose/                      ← 源码 (TDS networks, pose_modules, train)
│       ├── config/                        ← Hydra 配置 (5 个实验)
│       └── emg2pose/UmeTrack/             ← 手部前向运动学引擎
│
└── emg_transfer/                ← ★ 迁移学习项目 (本工作)
    ├── PROJECT.md               ← 项目详细文档
    ├── RESULTS.md               ← 训练结果分析
    ├── run_pipeline.py          ← 自动化流水线
    ├── config/
    │   ├── discrete_gestures_transfer.yaml          ← 顶层配置
    │   ├── data_module/discrete_gestures_data_module.yaml
    │   ├── data_module/data_split/discrete_gestures_split.yaml
    │   └── lightning_module/discrete_gestures_transfer_module.yaml
    └── emg_transfer/
        ├── networks.py          ← TDS blocks + Emg2PoseTdsGestureArchitecture + Baseline
        ├── lightning.py         ← DiscreteGesturesModule (动态metric + 冻结 + 差分LR)
        ├── train.py             ← 训练入口 (Hydra + 预训练权重加载)
        ├── transforms.py        ← 数据变换 (无 Reinhard)
        ├── data.py / data_module.py  ← HDF5 数据加载
        ├── cler.py              ← CLER 评估指标 (Needleman-Wunsch 对齐)
        ├── constants.py         ← 手势类别定义
        ├── augmentation.py      ← RotationAugmentation
        └── utils.py             ← 工具函数
```

## 快速开始

### 1. 环境准备

```bash
conda create -n emg python=3.10
conda activate emg
pip install torch pytorch-lightning hydra-core h5py numba numpy pandas tqdm
```

### 2. 数据准备

```bash
# 下载手势数据集 (31GB)
cd emg_nature/generic-neuromotor-interface-main/generic-neuromotor-interface-main
python -m generic_neuromotor_interface.scripts.download_data \
    --task discrete_gestures \
    --output-dir D:/emg/emg_nature/emg_data

# 下载 emg2pose 预训练 checkpoint
curl -L "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz" \
    -o emg2pose_checkpoints.tar.gz
tar -xzf emg2pose_checkpoints.tar.gz
```

### 3. 训练

```bash
cd emg_transfer

# 加载预训练权重微调 (Phase 2)
python -m emg_transfer.train

# 随机初始化对照组 (Phase 1 消融)
python -m emg_transfer.train pretrained_encoder_ckpt=null

# 自定义超参
python -m emg_transfer.train \
    lightning_module.freeze_backbone_epochs=10 \
    lightning_module.learning_rate=1e-4 \
    lightning_module.backbone_lr_ratio=0.01
```

### 4. 查看结果

```bash
# 训练日志
cat training_output.log

# TensorBoard (如已安装)
tensorboard --logdir logs/
```

## 9 类手势

| 编号 | 类别 | 说明 |
|------|------|------|
| 0 | index_press | 食指按下 |
| 1 | index_release | 食指抬起 |
| 2 | middle_press | 中指按下 |
| 3 | middle_release | 中指抬起 |
| 4 | thumb_click | 拇指点击 |
| 5 | thumb_down | 拇指向下 |
| 6 | thumb_in | 拇指向内 |
| 7 | thumb_out | 拇指向外 |
| 8 | thumb_up | 拇指向上 |

## 评估指标

- **CLER (Classification Error Rate)**：基于 Needleman-Wunsch 序列对齐的分类错误率，同时评估分类正确性和时序精度，越低越好
- **val_accuracy**：基于事件窗口内 argmax 的粗粒度准确率

## 技术要点

### 关键设计决策

| 决策 | 说明 |
|------|------|
| **去掉 Reinhard 压缩** | 保持与预训练输入分布一致（原始 EMG） |
| **完整 TDS backbone** | 不截断，利用全部 4 层预训练知识 |
| **50Hz 输出** (stride=40) | 40ms 事件脉冲 ≈ 2 步，足够检测 |
| **1 层 LSTM (64→128)** | 轻量分类头，防小样本过拟合 |
| **动态 metric 窗口** | 根据 output_freq 自动缩放 |
| **差分学习率** | backbone 1e-5, head 5e-4 (50:1 比例) |

### 标签时序

每个手势事件产生 40ms 宽度的 binary pulse（从事件时间 +80ms 到 +120ms），pulse_window 由 `transforms.py` 中的 `DiscreteGesturesTransform` 控制。

## 许可证

- emg_nature 和 emg2pose1 的代码版权归 Meta Platforms, Inc. 所有，使用其原始 LICENSE
- emg_transfer 为本工作的原创代码

## 参考

- [A generic non-invasive neuromotor interface for human-computer interaction](https://www.nature.com/articles/s41586-024-08058-x) — emg_nature 论文
- [emg2pose: A Large and Diverse Benchmark for Surface Electromyographic Hand Pose Estimation](https://arxiv.org/abs/2410.11825) — emg2pose 论文 (NeurIPS 2024)
- [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://arxiv.org/abs/1904.02619) — TDS 架构论文 (Hannun et al. 2019)
