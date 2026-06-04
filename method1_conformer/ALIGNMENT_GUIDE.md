# emg_transfer 入门指南：从 emg2pose 到 emg_nature 的迁移学习

## 一句话总结

将 emg2pose 在大规模数据上预训练的 TDS 卷积 backbone（手部姿态估计），迁移到 emg_nature 的 9 类手势分类任务上。通过**冻结→解冻微调**和**差分学习率**，在小数据集上提升分类性能。

---

## 1. 背景：三个项目的关系

三个项目共用同一款 **16 通道 EMG 腕带**（2000Hz 采样率），具备直接迁移的数据基础。

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   emg2pose (NeurIPS 2024)          emg_nature (Nature 2024)    │
│   任务: 手部姿态估计                任务: 手势分类               │
│   数据: 193人, 370小时              数据: ~100人, 离散手势       │
│   输出: 20 DOF 关节角度             输出: 9 类手势标签            │
│                                                                 │
│   ┌──────────────────┐            ┌──────────────────────┐      │
│   │   TDS Backbone   │◄──迁移───►│  Conv1d + 3×LSTM     │      │
│   │   (全卷积编码器)   │           │  (原始分类模型)       │      │
│   └──────────────────┘            └──────────────────────┘      │
│            │                                                      │
│            └────────────┬──────────────┘                         │
│                         │                                         │
│                  ┌──────▼──────┐                                 │
│                  │ emg_transfer │  ← 本项目                       │
│                  │ TDS + 轻量头 │                                 │
│                  └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心问题一：输入长度为什么不一致？

### 2.1 数值差异

| 项目 | 训练输入长度 | 等效时间 |
|------|-------------|---------|
| emg2pose (tracking_vemg2pose) | **11,790** 采样点 | ~5.9 秒 |
| emg_nature (discrete_gestures) | **16,000** 采样点 | ~8.0 秒 |
| emg_transfer | **16,000** 采样点 | ~8.0 秒 |

emg_transfer 继承了 emg_nature 的 `window_length=16000`，但同时加载了在 11790 长度上预训练的 TDS backbone。

### 2.2 为什么不需要对齐？

**TDS backbone 是全卷积网络，没有全连接层限制输入尺寸。**

所有 `Conv1d` 层都使用 `padding=0`（因果卷积），这意味着网络可以接受**任意长度 ≥ left_context (1790)** 的输入：

```python
# emg2pose/networks.py — TDS 中所有卷积都是无 padding 的
nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
          stride=stride, padding=0)
```

### 2.3 不同输入长度下的内部计算

```
输入 11,790 采样点:                      输入 16,000 采样点:
──────────────────                      ──────────────────

Conv1dBlock(k=11,s=5)                   Conv1dBlock(k=11,s=5)
  (11790-11)//5 + 1 = 2356                (16000-11)//5 + 1 = 3198

Conv1dBlock(k=5,s=2)                    Conv1dBlock(k=5,s=2)
  (2356-5)//2 + 1 = 1176                  (3198-5)//2 + 1 = 1597

TdsStage1(k=17,s=4)                     TdsStage1(k=17,s=4)
  (1176-17)//4 + 1 = 290                  (1597-17)//4 + 1 = 396
  + TDS blocks (保持长度)                  + TDS blocks (保持长度)

TdsStage2(k=9,s=2)                      TdsStage2(k=9,s=2)
  (290-9)//2 + 1 = 141                    (396-9)//2 + 1 = 194
  + Linear(256→64)                        + Linear(256→64)

输出: (B, 64, 141) @25Hz                输出: (B, 64, 194) @25Hz
```

**结论：卷积层天然适配变长输入，不同长度产生不同时间步数，这是正常行为。**

emg_transfer 选择 16000 而非 11790，仅为了与 emg_nature 原始数据配置保持一致（8 秒窗口），没有技术约束。

---

## 3. 核心问题二：输出长度 / 频率如何对齐？

### 3.1 三个模型的输出差异

```
                      输出频率          stride         left_context
                      ────────         ──────         ────────────
emg2pose TDS:        25Hz             80              1790
emg_nature 原始:      200Hz            10              20
emg_transfer:        50Hz             40              1790
```

这是一个**三重不匹配**：不同的输出频率、不同的 stride、不同的 left_context。

### 3.2 对齐机制全景图

```
输入 EMG (B, 16, 16000) @2000Hz
│
│  ╔══════════════════════════════════════════╗
│  ║   TDS Backbone (预训练, 前5 epoch冻结)   ║
│  ║   left_context = 1790                    ║
│  ║   总下采样率 80×                          ║
│  ╚══════════════════╤═══════════════════════╝
│                     │
│              ┌──────▼──────┐
│              │ (B,64,194)  │  @25Hz  ← TDS 原始输出
│              └──────┬──────┘
│                     │
│         ★ ┌─────────▼─────────┐
│         ★ │  F.interpolate    │  线性插值: 25Hz → 50Hz
│         ★ │  size=356         │  target_len = (16000-1790-1)//40 + 1
│         ★ │  mode='linear'    │
│         ★ └─────────┬─────────┘
│                     │
│              ┌──────▼──────┐
│              │ (B,64,356)  │  @50Hz
│              └──────┬──────┘
│                     │
│              ┌──────▼──────┐
│              │  Permute    │  (B,C,T) → (B,T,C)
│              └──────┬──────┘
│                     │
│              ┌──────▼──────┐
│              │ 1-Layer LSTM│  64 → 128  (全新训练)
│              └──────┬──────┘
│                     │
│              ┌──────▼──────┐
│              │  Linear     │  128 → 9   (全新训练)
│              └──────┬──────┘
│                     │
│              ┌──────▼──────┐
│              │  Permute    │  (B,T,C) → (B,C,T)
│              └──────┬──────┘
│                     │
              输出 (B, 9, 356) @50Hz
```

### 3.3 五个对齐步骤逐一说明

#### 步骤 ①：选择 50Hz 作为目标输出频率

```python
# networks.py — Emg2PoseTdsGestureArchitecture.__init__()
self.stride = 40          # 2000Hz / 40 = 50Hz 输出
self.left_context = 1790  # 来自 TDS backbone
```

为什么是 50Hz？
- 比 emg2pose 的 25Hz 更密集（更有判别力）
- 比 emg_nature 的 200Hz 更稀疏（减少计算量）
- 40ms 脉冲窗口 ≈ 2 个输出步（便于捕捉 press/release 事件）

#### 步骤 ②：F.interpolate 线性插值（25Hz → 50Hz）

```python
# networks.py — forward()
features = self.encoder(x)                 # (B, 64, 194) @25Hz
T = x.shape[-1]                             # 16000
target_len = (T - 1790 - 1) // 40 + 1      # = 356
features = F.interpolate(
    features, size=target_len, mode='linear', align_corners=False
)                                           # (B, 64, 356) @50Hz
```

这是**最核心的桥接操作**：在时间维度上对 64 维特征做线性插值，从 194 步拉伸到 356 步。

#### 步骤 ③：目标标签切片

```python
# lightning.py — _step()
targets = targets[:, :, self.network.left_context :: self.network.stride]
#                    targets[:, :, 1790::40]
```

ground-truth 标签矩阵（2000Hz）被降采样到 50Hz，跳过前 1790 个样本（等同于 backbone 的感受野），然后每 40 步取一个。

#### 步骤 ④：新分类头替换姿态解码器

emg2pose 的 PoseDecoder 输出 20 维关节角度——对分类无意义。transfer 模型用全新的轻量头替换：

```python
# networks.py — __init__()
self.lstm = nn.LSTM(input_size=64, hidden_size=128,
                    num_layers=1, batch_first=True)
self.classifier = nn.Linear(128, 9)
```

| 对比 | emg2pose Decoder | emg_transfer Head |
|------|-----------------|-------------------|
| 类型 | SequentialLSTM + MLP | 1-Layer LSTM + Linear |
| 输入 | 64 维 TDS 特征 | 64 维 TDS 特征 |
| 输出 | 20 DOF 关节角度 | 9 类手势 logits |
| 参数量 | ~4M | ~100K |

#### 步骤 ⑤：预训练权重仅加载 encoder

```python
# networks.py — load_encoder_from_ckpt()
# 将 emg2pose checkpoint 中的 'model.network.layers.X.*'
# 重映射为 'encoder.layers.X.*'
# strict=False → 只加载 backbone 权重, head 保持随机初始化
```

### 3.4 完整的对齐公式

```
给定原始 EMG 长度 T:

1. TDS 输出步数:        N_feat = (T - 各层边界损失) / 80
                       对于 T=16000: N_feat ≈ 194

2. 插值目标步数:        N_out = (T - 1790 - 1) // 40 + 1
                       对于 T=16000: N_out = 356

3. 标签切片步数:        N_label = len(range(T)[1790::40])
                       = ceil((T - 1790) / 40)
                       对于 T=16000: N_label = 356

4. 验证: N_out == N_label ✓
```

---

## 4. 两模型对比总览

```
╔══════════════════════════════════════════════════════════════════════╗
║                        emg_nature 原始模型                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  输入: (B, 16, 16000)  EMG @2000Hz                                   ║
║        │                                                             ║
║   ┌────▼────────────────────────────────────────┐                    ║
║   │  ReinhardCompression (动态范围压缩)           │                    ║
║   └────┬────────────────────────────────────────┘                    ║
║        │ (B, 16, 16000)                                              ║
║   ┌────▼────────────────────────────────────────┐                    ║
║   │  Conv1d(16→512, k=21, s=10, p=0)            │  left=20           ║
║   └────┬────────────────────────────────────────┘                    ║
║        │ (B, 512, ~1598) @200Hz                                      ║
║   ┌────▼────────────────────────────────────────┐                    ║
║   │  ReLU → Dropout(0.1) → LayerNorm(512)       │                    ║
║   └────┬────────────────────────────────────────┘                    ║
║        │ Permute → (B, T, 512)                                        ║
║   ┌────▼────────────────────────────────────────┐                    ║
║   │  3-Layer LSTM(512→512, dropout=0.1)         │  ← 主体参数        ║
║   └────┬────────────────────────────────────────┘                    ║
║        │ (B, T, 512)                                                 ║
║   ┌────▼────────────────────────────────────────┐                    ║
║   │  LayerNorm(512) → Linear(512→9)             │                    ║
║   └────┬────────────────────────────────────────┘                    ║
║        │ (B, T, 9), Permute → (B, 9, T)                             ║
║                                                                      ║
║  输出: (B, 9, ~1598) @200Hz                                          ║
║  参数量: ~12M (主要来自 3 层 LSTM)                                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                     emg_transfer 迁移模型                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  输入: (B, 16, 16000)  EMG @2000Hz  ← 无 Reinhard (保持一致)        ║
║        │                                                             ║
║   ┌────▼──────────────────────────────────────────────┐              ║
║   │  TDS Backbone (来自 emg2pose 预训练)               │              ║
║   │                                                    │              ║
║   │  Conv1dBlock(16→256, k=11, s=5)   stride_cum=5    │              ║
║   │  Conv1dBlock(256→256, k=5, s=2)   stride_cum=10   │              ║
║   │  TdsStage1: Conv(k=17,s=4)        stride_cum=40   │              ║
║   │           + 2×TDSConv2dBlock(k=9)                 │              ║
║   │  TdsStage2: Conv(k=9,s=2)         stride_cum=80   │              ║
║   │           + 2×TDSConv2dBlock(k=5)                 │              ║
║   │           + Linear(256→64)                        │              ║
║   │                                                    │              ║
║   │  左感受野: 1790 samples (~0.9s)                    │              ║
║   └────┬───────────────────────────────────────────────┘              ║
║        │ (B, 64, ~194) @25Hz  ← 16000/80                             ║
║        │                                                             ║
║   ┌────▼──────────────────────────────────────────────┐              ║
║   │  F.interpolate(size=356, mode='linear')           │  ← 对齐核心  ║
║   └────┬──────────────────────────────────────────────┘              ║
║        │ (B, 64, 356) @50Hz                                          ║
║        │ Permute → (B, 356, 64)                                      ║
║   ┌────▼──────────────────────────────────────────────┐              ║
║   │  1-Layer LSTM(64→128)                             │  ← 轻量头    ║
║   │  Linear(128→9)                                     │              ║
║   └────┬──────────────────────────────────────────────┘              ║
║        │ Permute → (B, 9, 356)                                       ║
║                                                                      ║
║  输出: (B, 9, 356) @50Hz                                             ║
║  参数量: ~1.6M (backbone ~1.5M + head ~0.1M)                        ║
║                                                                      ║
║  训练策略:                                                            ║
║    阶段1 (epoch 0-4): 冻结 backbone, 只训练 head                      ║
║    阶段2 (epoch 5+):  解冻, backbone_lr=1e-5, head_lr=5e-4           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 5. 输出时长对比数值表

以 16000 采样点输入为例：

| 模型 | stride | left_context | 输出步数 | 输出频率 | 等效时长 |
|------|--------|-------------|---------|---------|---------|
| emg_nature 原始 | 10 | 20 | 1598 | 200Hz | ~7.99s |
| emg2pose TDS | 80 | 1790 | 194 | 25Hz | ~7.76s |
| emg_transfer | 40 | 1790 | 356 | 50Hz | ~7.12s |

```
公式: N_out = (T - left_context - 1) // stride + 1

emg_nature:   (16000 - 20 - 1) // 10 + 1 = 15979//10 + 1 = 1598
emg2pose:     经过逐层卷积等价于 (16000 - 1790) / 80 ≈ 194
emg_transfer: (16000 - 1790 - 1) // 40 + 1 = 14209//40 + 1 = 356
```

---

## 6. 快速上手

### 6.1 环境

```bash
conda create -n emg_transfer python=3.10 -y
conda activate emg_transfer
pip install torch pytorch-lightning hydra-core h5py numba numpy pandas tqdm
```

### 6.2 数据准备

```bash
# 1. 下载 emg_nature 手势数据 (~31GB)
cd D:/emg/emg_nature/generic-neuromotor-interface-main/generic-neuromotor-interface-main
python -m generic_neuromotor_interface.scripts.download_data \
    --task discrete_gestures \
    --output-dir D:/emg/emg_nature/emg_data

# 2. 确认 emg2pose checkpoint 存在
ls D:/emg/emg2pose1/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt
```

### 6.3 训练

```bash
cd D:/emg/emg_transfer

# 默认配置：加载预训练 backbone + 冻结 5 epoch + 差分 LR
python -m emg_transfer.train

# 消融实验：随机初始化 backbone（不加载预训练）
python -m emg_transfer.train pretrained_encoder_ckpt=null

# 运行 baseline（emg_nature 原始模型）
# 修改 config/lightning_module/discrete_gestures_transfer_module.yaml:
#   将 network._target_ 改为 emg_transfer.networks.DiscreteGesturesArchitecture
python -m emg_transfer.train

# 自定义超参
python -m emg_transfer.train \
    lightning_module.freeze_backbone_epochs=10 \
    lightning_module.learning_rate=1e-4 \
    trainer.max_epochs=300
```

### 6.4 预期结果

| 模型 | Val Accuracy | Test CLER |
|------|-------------|-----------|
| emg_nature Baseline (随机初始化) | ~65% | ~0.25 |
| emg_transfer (预训练 backbone) | ~78% | ~0.15 |
| emg_transfer (随机初始化) | ~55% | ~0.35 |

### 6.5 关键配置文件速查

```
config/discrete_gestures_transfer.yaml          ← 顶层入口
│
├── data_module: config/data_module/discrete_gestures_data_module.yaml
│   ├── window_length: 16000     ← 输入长度 (8秒)
│   ├── batch_size: 4            ← RTX 2060 6GB 上限
│   └── num_workers: 0           ← Windows HDF5 兼容
│
└── lightning_module: config/lightning_module/discrete_gestures_transfer_module.yaml
    ├── network._target_         ← 切换迁移模型/基线模型
    ├── freeze_backbone_epochs: 5  ← 冻结轮数
    ├── backbone_lr_ratio: 0.02   ← backbone LR = 5e-4 × 0.02 = 1e-5
    └── learning_rate: 5e-4       ← head LR
```

### 6.6 代码入口速查

| 文件 | 作用 | 关键类/函数 |
|------|------|------------|
| `emg_transfer/networks.py` | 模型定义 | `Emg2PoseTdsGestureArchitecture` (迁移), `DiscreteGesturesArchitecture` (基线) |
| `emg_transfer/lightning.py` | 训练逻辑 | `DiscreteGesturesModule._step()` (标签切片), `on_train_epoch_start()` (冻结/解冻), `configure_optimizers()` (差分LR) |
| `emg_transfer/train.py` | 训练入口 | `main()` Hydra 配置 → 加载权重 → 训练 |
| `emg_transfer/data.py` | 数据加载 | `WindowedEmgDataset` (滑窗读取 HDF5) |
| `emg_transfer/transforms.py` | 数据变换 | `DiscreteGesturesTransform` (EMG→Tensor, 事件→脉冲矩阵) |
| `emg_transfer/cler.py` | 评估指标 | `compute_cler()` (Needleman-Wunsch 序列对齐) |

---

## 7. 常见疑问

### Q: 为什么迁移模型用 50Hz 而不是 200Hz？
50Hz 是 emg2pose backbone 的 25Hz 和 emg_nature 的 200Hz 之间的折中。40ms 事件脉冲在 50Hz 下约 2 个输出步，足够区分，同时模型更轻量。

### Q: 为什么要去掉 Reinhard 压缩？
TDS backbone 在 emg2pose 原始任务中就是在**原始 EMG** 上预训练的，没有经过 Reinhard 压缩。保持一致才能让预训练权重发挥最大效果。

### Q: stride=40 和 left_context=1790 是怎么算出来的？
`stride=40` 是手动选择的（2000/40=50Hz）。`left_context=1790` 由 `TdsNetwork._get_left_context()` 根据所有卷积层的 kernel_size 和 stride 自动累积计算。

### Q: 输入长度变了，TDS 输出步数也变了，插值会出错吗？
不会。`target_len = (T - left_context - 1) // stride + 1` 是动态计算的，任何 `T ≥ 1790` 都能正确算出目标步数。`F.interpolate` 只关心 `size` 参数，不管输入有多长。

### Q: 为什么 baseline 的 left_context 只有 20？
因为 emg_nature 原始模型只有一层 Conv1d（k=21, s=10），感受野很小：left = kernel_size - 1 = 20。而 TDS 经过 8 层因果卷积累计了 1790 的感受野。
