# emg_transfer 项目文档

## 一、项目概述

本项目实现从 emg2pose（手部姿态重建）到 emg_nature（手势分类）的迁移学习。

**核心思路**：将 emg2pose 在大规模数据上预训练的 TDS 卷积 backbone 迁移到 emg_nature 的 9 类手势分类任务上，通过端到端微调提升小数据集上的分类性能。

两个上游项目共用同一款 16 通道 EMG 腕带（2000Hz 采样率），具备直接迁移的数据基础。

## 二、项目结构

```
emg_transfer/
├── config/
│   ├── discrete_gestures_transfer.yaml          # 顶层实验配置 (入口)
│   ├── data_module/
│   │   ├── discrete_gestures_data_module.yaml   # 数据加载配置
│   │   └── data_split/
│   │       └── discrete_gestures_split.yaml     # train/val/test 划分
│   └── lightning_module/
│       └── discrete_gestures_transfer_module.yaml  # 模型和训练超参
├── emg_transfer/
│   ├── __init__.py
│   ├── constants.py          # 手势类别定义、采样率等常量
│   ├── utils.py              # 工具函数 (路径拼接等)
│   ├── augmentation.py       # RotationAugmentation 数据增强
│   ├── transforms.py         # 数据变换 (EMG→Tensor, 手势脉冲生成)
│   ├── data.py               # HDF5 数据读取、滑窗数据集
│   ├── data_module.py        # PyTorch Lightning DataModule
│   ├── networks.py           # ★ 核心 — TDS blocks + Emg2PoseTdsGestureArchitecture + Baseline
│   ├── lightning.py          # ★ 核心 — DiscreteGesturesModule (动态metric + 冻结 + 差分LR)
│   ├── train.py              # ★ 核心 — 训练入口 (Hydra + 预训练权重加载)
│   └── cler.py               # CLER 评估指标 (Needleman-Wunsch 序列对齐)
└── run_pipeline.py           # 自动化流水线脚本 (下载→解压→训练)
```

## 三、方法详解

### 3.1 预训练 Backbone

来源：emg2pose 项目的 `TdsNetwork`（Hannun et al. 2019 的 TDS 架构）。

```
TDS 编码器结构:
  Conv1dBlock(16→256, k=11, s=5)      → stride=5
  Conv1dBlock(256→256, k=5, s=2)      → stride=10
  TdsStage1: Conv1d(k=17, s=4)        → stride=40
           + 2× TDSConv2dBlock(k=9)
  TdsStage2: Conv1d(k=9, s=2)         → stride=80
           + 2× TDSConv2dBlock(k=5)
           → Linear(256→64)

  输入: (B, 16, T) 原始 EMG @2000Hz
  输出: (B, 64, T/80) @25Hz
  left_context = 1790 samples (~0.9s)
```

预训练于 emg2pose 的 tracking_vemg2pose 任务：193 人、370 小时 EMG 数据，预测 20 DOF 手部关节角度。

### 3.2 迁移架构

完整 TDS backbone 输出 25Hz 特征后，插值到 50Hz，接轻量分类头：

```
完整 Pipeline:
  原始 EMG (B, 16, T) @2000Hz
    → TDS Encoder → (B, 64, T_feat) @25Hz
    → F.interpolate → (B, 64, T_target) @50Hz
    → 1-Layer LSTM(64→128)
    → Linear(128→9)
    → (B, 9, T_target) @50Hz
```

### 3.3 关键设计决策

| 决策 | 说明 |
|------|------|
| **去掉 Reinhard 压缩** | TDS backbone 在原始 EMG 上预训练，保持输入分布一致 |
| **完整 TDS** | 不截断，利用全部 4 层预训练知识 |
| **stride=40, left_context=1790** | 真实值，不伪装；target 切片按 `[:,:,1790::40]` |
| **50Hz 输出** | 原始模型 200Hz → 迁移模型 50Hz，40ms 事件脉冲 ≈ 2 步 |
| **1 层 LSTM** | 轻量设计 (64→128)，防止在小数据集上过拟合 |
| **动态 Metric 窗口** | validation accuracy 和 release mask 的 rpad 按 output_freq 自动缩放 |

### 3.4 训练策略

```
Phase 1 (epoch 0-4):  冻结 TDS backbone
                      只训练 LSTM + classifier
                      head_lr = 5e-4

Phase 2 (epoch 5-250): 解冻 backbone
                      全量端到端微调
                      backbone_lr = 1e-5 (head_lr × 0.02)
                      head_lr = 5e-4

学习率调度: Linear warmup (0→1×, 5 epochs) → MultiStepLR (milestones=[25], gamma=0.5)
损失函数: BCEWithLogitsLoss + 手指状态 mask
梯度裁剪: 0.5
```

### 3.5 9 类手势

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

其中 index 和 middle 的 press/release 是成对事件，训练时用 `FingerStateMaskGenerator` 掩码：release 事件只在手指处于 pressed 状态时才参与 loss 计算。

## 四、各脚本注意点

### 4.1 `networks.py`

**TDS 构建块**：`Conv1dBlock`, `TDSConv2dBlock`, `TDSFullyConnectedBlock`, `TDSConvEncoder`, `TdsStage`, `TdsNetwork`, `build_tds_network()`

这些是从 emg2pose 复制的自包含副本，不依赖 emg2pose 项目。注意：
- `Conv1dBlock` 默认 `norm_type="layer"`，输出为 BCT 格式
- `TdsStage` 在 `out_channels` 不为 None 时会在线性投影前进行 swapaxes 操作
- `TdsNetwork.left_context` 通过 `_get_left_context()` 精确计算（1790 samples）

**Emg2PoseTdsGestureArchitecture**：
- `left_context` 和 `stride` 属性被 `DiscreteGesturesModule._step()` 用于 target 切片
- `forward()` 中的 `target_len = (T - left_context - 1) // stride + 1` 保证了与 sliced targets 的精确对齐
- 加载预训练权重时：`model.network.layers.X.*` → `encoder.layers.X.*`

**DiscreteGesturesArchitecture**：原始 emg_nature baseline，保留了 Reinhard 压缩。

### 4.2 `lightning.py`

**DiscreteGesturesModule** 的关键修改：

1. **动态 Metric 窗口**：
   ```python
   output_freq = 2000 // self.network.stride  # 50Hz for transfer, 200Hz for baseline
   w_start = round(0.050 * output_freq)        # 50ms 前的窗口
   w_end = round(0.150 * output_freq)          # 150ms 后的窗口
   rpad = round(0.040 * output_freq)           # release mask 右填充
   ```

2. **Backbone 冻结/解冻**：
   ```python
   def on_train_epoch_start(self):
       if self.current_epoch < self.freeze_backbone_epochs:
           self.network.encoder.requires_grad_(False)  # 冻结
       elif self.current_epoch == self.freeze_backbone_epochs:
           self.network.encoder.requires_grad_(True)   # 解冻
   ```

3. **差分学习率**：
   ```python
   backbone_lr = learning_rate * backbone_lr_ratio  # 1e-5 = 5e-4 × 0.02
   head_lr = learning_rate                           # 5e-4
   ```

### 4.3 `train.py`

- 使用 Hydra 管理配置，入口 `@hydra.main(config_path="../config", config_name="discrete_gestures_transfer")`
- 预训练权重加载逻辑：从 `config.get("pretrained_encoder_ckpt")` 读取路径，调用 `module.network.load_encoder_from_ckpt()`
- 训练完成后自动加载 best checkpoint 做 validate + test

### 4.4 `transforms.py`

- `DiscreteGesturesTransform`：提取原始 EMG（无 Reinhard），将手势事件时间转换为 binary pulse matrix
- `pulse_window: [0.08, 0.12]`：每个事件产生 40ms 的脉冲（从事件时间 +80ms 到 +120ms）
- `HandwritingTransform` 中的 `charset` 是懒加载的，只有实例化时才尝试从 emg_nature 导入

### 4.5 `data.py` / `data_module.py`

- 数据格式：HDF5 文件，`/data/emg` 和 `/data/time`，`/prompts` 包含事件标注
- 训练时：`window_length=16000, stride=16000`（8秒窗，无重叠），`jitter=True`（随机偏移）
- 验证时：同 window 但 `jitter=False`
- 测试时：`window_length=None`（整段录音），`batch_size=1`
- 零 num_workers（Windows 上 HDF5 多进程有问题）

### 4.6 `cler.py`

- CLER (Classification Error Rate)：使用 Needleman-Wunsch 序列对齐算法
- 阈值参数：`THRESHOLD=0.35`, `DEBOUNCE=0.05s`, `TOLERANCE=(-0.05, 0.25)s`
- 这些参数与输出频率无关（单位是秒），所以 50Hz 和 200Hz 都适用

### 4.7 配置文件

**discrete_gestures_transfer.yaml**（顶层）：
- 指定 data_module、data_split、lightning_module
- 设置 data_location、seed、trainer 参数
- `pretrained_encoder_ckpt`：预训练 checkpoint 路径（null = 随机初始化）

**discrete_gestures_transfer_module.yaml**（模型配置）：
- 可以切换 `network._target_`：`Emg2PoseTdsGestureArchitecture`（迁移）或 `DiscreteGesturesArchitecture`（baseline）
- `freeze_backbone_epochs`：冻结轮数
- `backbone_lr_ratio`：backbone 学习率比例

**discrete_gestures_data_module.yaml**（数据配置）：
- `window_length: 16000` (8 秒)
- `batch_size: 4`（适配 RTX 2060 6GB）
- `num_workers: 0`（Windows 兼容性）

## 五、运行方式

### 环境准备

```bash
conda activate neuromotor  # 或 emg2pose
pip install pytorch-lightning hydra-core h5py numba numpy pandas tqdm
```

### 数据准备

```bash
# 下载全量手势数据 (31GB)
cd emg_nature/generic-neuromotor-interface-main/generic-neuromotor-interface-main
python -m generic_neuromotor_interface.scripts.download_data \
    --task discrete_gestures \
    --output-dir D:/emg/emg_nature/emg_data

# 下载 emg2pose 预训练 checkpoint
curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz" \
    -o emg2pose_checkpoints.tar.gz
tar -xzf emg2pose_checkpoints.tar.gz  # 得到 tracking_vemg2pose.ckpt
```

### 训练命令

```bash
cd D:/emg/emg_transfer

# Phase 2: 加载预训练 (默认)
python -m emg_transfer.train

# Phase 1 消融: 随机初始化
python -m emg_transfer.train \
    pretrained_encoder_ckpt=null

# 调整超参
python -m emg_transfer.train \
    lightning_module.freeze_backbone_epochs=10 \
    lightning_module.learning_rate=1e-4 \
    lightning_module.backbone_lr_ratio=0.01
```

### 输出文件

- 训练日志：`training_output.log`
- Checkpoints：`logs/YYYY-MM-DD/HH-MM-SS/lightning_logs/version_X/checkpoints/`
- Hydra 配置快照：`logs/.../hydra_configs/`

## 六、依赖关系

```
emg_transfer/
├── 独立，不依赖 emg_nature 源码 (import 路径全部改为 emg_transfer.*)
├── 懒加载 emg_nature.handwriting_utils (仅在 HandwritingModule 实例化时)
├── 懒加载 emg_nature.handwriting_utils (仅在 HandwritingTransform 实例化时)
└── 训练时需 emg2pose checkpoint 文件 (.ckpt)
```
