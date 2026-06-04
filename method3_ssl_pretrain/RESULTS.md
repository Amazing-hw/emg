# 迁移学习实验结果分析

## 实验信息

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-10 |
| 硬件 | NVIDIA RTX 2060 (6GB VRAM) |
| 训练时长 | ~12.5 小时 (250 epochs) |
| 预训练权重 | tracking_vemg2pose.ckpt (emg2pose 官方) |
| 微调数据 | 全量 discrete_gestures (100 HDF5, 80 train / 10 val / 10 test) |

## 模型架构

```
EMG Input (B, 16, 16000) @2000Hz — 原始信号，无 Reinhard 压缩
    │
    ▼
TDS Backbone (emg2pose 预训练, 完整)
  Conv1d(16→256, k=11, s=5)
  Conv1d(256→256, k=5, s=2)
  TdsStage1(k=17, s=4, 2×TDS k=9)
  TdsStage2(k=9, s=2, 2×TDS k=5, out=64)
    │
    ▼  (B, 64, 178) @25Hz
F.interpolate → 50Hz (356 步)
    │
    ▼  (B, 64, 356) @50Hz
1-Layer LSTM(64→128)
    │
    ▼  (B, 128, 356)
Linear(128→9)
    │
    ▼
Output: (B, 9, 356) @50Hz — 9 类手势 logits
```

## 训练策略

| 阶段 | Epoch | Backbone LR | Head LR | 说明 |
|------|-------|-------------|---------|------|
| Phase 1 (冻结) | 0-4 | 0 | 5e-4 | 仅训练 LSTM + 分类头 |
| Phase 2 (微调) | 5-250 | 1e-5 | 5e-4 | 全量端到端微调 |

## 训练曲线

```
Epoch  |  Train Loss  |  Val Accuracy  |  阶段
-------|-------------|---------------|----------
    0  |     0.671   |      0.134    |  冻结
    1  |     0.031   |      0.151    |
    2  |     0.021   |      0.203    |
    3  |     0.016   |      0.351    |
    4  |     0.015   |      0.394    |  冻结结束
    5  |     0.014   |      0.457    |  解冻 ★
    7  |     0.011   |      0.461    |
    9  |     0.013   |      0.477    |
   10  |     0.015   |      0.490    |
   12  |     0.012   |      0.492    |
   16  |     0.011   |      0.501    |
   19  |     0.011   |      0.518    |
   24  |     0.008   |      0.528    |  ★ 最佳
 25-48| 0.008-0.011|   0.51-0.52    |  plateau
49-250| 0.008-0.010|   0.50-0.52    |  plateau
```

## 最终评估指标

| 指标 | 数值 |
|------|------|
| 最佳 val_accuracy | **0.5281** (epoch 24) |
| 最佳 checkpoint | epoch=24-step=87650.ckpt |
| Val loss (best ckpt) | 0.0158 |
| Test loss | 0.0116 |
| **Test CLER** | **0.1410 (14.1%)** |

## 结果分析

### 正面发现

1. **预训练特征可迁移性强**：backbone 冻结时，仅靠 4 个 epoch 训练分类头，accuracy 就达到 0.394。这说明 emg2pose 的 TDS encoder 学到的 EMG 时空表征对离散手势分类任务直接有用。

2. **解冻即见效**：Epoch 5 解冻 backbone 后，accuracy 从 0.394 跃升至 0.457（+16%），验证了端到端微调的必要性。

3. **快速收敛**：仅 24 epoch 就达到最佳 accuracy (0.528)，之后进入 plateau。相比原 emg_nature 模型（通常需要 100+ epoch），收敛速度显著加快。

4. **CLER 14.1%**：在 9 类手势上，模型对约 85.9% 的事件能正确分类。考虑到数据量较小且是 wristband EMG 的困难场景，这是可接受的结果。

### 待改进点

1. **Plateau 过早**：Epoch 24 后 accuracy 不再提升，可能原因：
   - 学习率过高（5e-4 for head, 1e-5 for backbone），需要更激进的学习率衰减
   - Batch size 过小（4），梯度噪声大
   - 需要 weight decay 正则化（当前未使用）
   - Backbone LR 比例 (0.02) 可能需要调整

2. **缺少对照组**：当前只有预训练版本的结果，无法量化"提升来自预训练"还是"提升来自更好的架构"。需要跑 Phase 1（随机初始化 TDS）作为对照。

3. **Val accuracy 与 test CLER 的差距**：val_accuracy (0.528) 对应的 test CLER (0.141)，两个指标衡量的是不同方面：
   - accuracy 是基于事件窗口内的 argmax 匹配（粗粒度）
   - CLER 是基于 Needleman-Wunsch 序列对齐的错误率（细粒度，含时序精度）

## 后续计划

1. **Phase 1 随机初始化对照组** — 量化预训练迁移的实际增益
2. **超参数调优** — 学习率、学习率衰减策略、batch size（梯度累积）
3. **架构消融** — 对比完整 TDS vs 截断 TDS vs 原始 Conv1d+LSTM
4. **数据增强优化** — 当前只用了 rotation augmentation
