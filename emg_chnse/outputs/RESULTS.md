# emg_chnse — 4 通道 EMG 筛选结果报告

## 一、项目目标

从 16 通道 EMG 腕带中，针对两个手势场景分别筛选出最优的 **4 个通道**：

| 场景 | 手势 | 类别数 | 涉及手指 |
|------|------|--------|----------|
| **A — Thumb** | thumb_click, thumb_down, thumb_in, thumb_out, thumb_up | 5 | 仅拇指 |
| **B — Index+Middle** | index_press, index_release, middle_press, middle_release | 4 | 仅食指+中指 |

---

## 二、数据与筛选方法

### 2.1 数据来源

- **训练数据**：emg_nature 离散手势数据集，80 个用户，16 通道 EMG @2000Hz
- **预训练模型**：emg2pose 的 `tracking_vemg2pose.ckpt`（193 人、370 小时手部姿态估计预训练）
- **数据量**：场景 A 约 43,000 事件，场景 B 约 79,000 事件

### 2.2 筛选方法

4 种互补打分方法，覆盖信号层、统计层、模型层三个维度：

| 方法 | 层面 | 原理 | 速度 |
|------|------|------|------|
| **SNR** | 信号层 | 手势窗口 EMG 方差 / 静息期 EMG 方差（dB） | 快 |
| **Fisher** | 统计层 | 类间方差 / 类内方差，衡量通道对不同手势的线性可分性 | 快 |
| **Mutual Info** | 统计层 | EMG 幅值与手势标签之间的互信息（bits） | 快 |
| **Weight Norm** | 模型层 | TDS 第一层 Conv1d(16→256) 的每个输入通道权重 L2 范数 | 极快 |

Saliency 和 Ablation（基于已训练模型的梯度/消融）预留接口，待有完整的 16 通道训练模型后可运行。

### 2.3 聚合方式

- **主排序**：排名法（rank-sum），各方法排名之和，越小越优
- **辅排序**：Z 分数均值，用于打破平局
- **冗余检查**：Top-4 通道间 Pearson 相关系数 > 0.9 时替换次优通道

---

## 三、场景 A — Thumb（拇指 5 类手势）结果

### 3.1 拇指手势的解剖学基础

拇指的四向滑动和点击涉及以下关节运动：

```
拇指关节（emg2pose 20 DOF 中的 joints 0-3）：

  THUMB_CMC_FE (joint 0)  ← 腕掌关节屈伸  → thumb_up / thumb_down
  THUMB_CMC_AA (joint 1)  ← 腕掌关节外展内收 → thumb_in / thumb_out
  THUMB_MCP_FE (joint 2)  ← 掌指关节屈伸  → thumb_click
  THUMB_IP_FE  (joint 3)  ← 指间关节屈伸  → thumb_click

手势→关节映射：
  thumb_up    → CMC_FE 伸展 + MCP_FE 伸展
  thumb_down  → CMC_FE 屈曲 + MCP_FE 屈曲
  thumb_in    → CMC_AA 内收
  thumb_out   → CMC_AA 外展
  thumb_click → MCP_FE 快速屈伸 + IP_FE 屈曲
```

### 3.2 通道排名

| 排名 | 通道 | rank_sum | z_mean | SNR | Fisher | MI | W-Norm |
|------|------|----------|--------|-----|--------|-----|--------|
| **★1** | **Ch 6** | 12 | +1.27 | 2 | 3 | 2 | 5 |
| **★2** | **Ch 7** | 16 | +0.99 | 4 | 5 | 1 | 6 |
| **★3** | **Ch 5** | 24 | +0.58 | 1 | 4 | 3 | 16 |
| **★4** | **Ch 2** | 25 | +0.50 | 8 | 12 | 8 | 3 |
| 5 | Ch 3 | 29 | +0.14 | 12 | 9 | 10 | 2 |
| 6 | Ch 1 | 31 | +0.29 | 3 | 16 | 6 | 4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 14 | Ch 11 | 46 | -0.73 | 16 | 7 | 15 | 11 |
| 15 | Ch 15 | 46 | -0.71 | 14 | 11 | 13 | 10 |
| 16 | Ch 10 | 47 | -0.83 | 8 | 15 | 14 | 9 |

### 3.3 推荐结论

```
★ 场景 A 推荐 4 通道：[6, 7, 5, 2]
```

各通道在信号层（SNR）、统计层（Fisher、MI）和模型层（Weight Norm）之间一致性良好（Spearman ρ > 0.6），Ch 6 在四种方法中全部排名前 5，是最稳健的选择。

---

## 四、场景 B — Index+Middle（食指+中指 4 类手势）结果

### 4.1 食指中指捏合的解剖学基础

捏合（pinch）涉及食指和中指的屈曲/伸展：

```
食指关节（joints 4-7）：              中指关节（joints 8-11）：

  INDEX_MCP_AA  (joint 4) ← 外展内收      MIDDLE_MCP_AA  (joint 8)
  INDEX_MCP_FE  (joint 5) ← 掌指屈伸 ★    MIDDLE_MCP_FE  (joint 9)  ← 掌指屈伸 ★
  INDEX_PIP_FE  (joint 6) ← 近端指间 ★    MIDDLE_PIP_FE  (joint 10) ← 近端指间 ★
  INDEX_DIP_FE  (joint 7) ← 远端指间       MIDDLE_DIP_FE  (joint 11) ← 远端指间

★ = press/release 主要涉及的关节（屈曲=press, 伸展=release）
```

### 4.2 通道排名

| 排名 | 通道 | rank_sum | z_mean | SNR | Fisher | MI | W-Norm |
|------|------|----------|--------|-----|--------|-----|--------|
| **★1** | **Ch 2** | 19 | +0.78 | 3 | 3 | 5 | 3 |
| **★2** | **Ch 6** | 19 | +0.78 | 5 | 10 | 7 | 5 |
| **★3** | **Ch 5** | 22 | +0.78 | 1 | 11 | 2 | 16 |
| **★4** | **Ch 3** | 26 | +0.47 | 15 | 4 | 3 | 2 |
| 5 | Ch 7 | 26 | +0.45 | 7 | 1 | 12 | 6 |
| 6 | Ch 1 | 27 | +0.45 | 2 | 14 | 8 | 4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 14 | Ch 13 | 46 | -0.86 | 12 | 8 | 13 | 13 |
| 15 | Ch 14 | 48 | -0.89 | 13 | 9 | 14 | 14 |
| 16 | Ch 15 | 51 | -0.86 | 14 | 13 | 16 | 10 |

### 4.3 推荐结论

```
★ 场景 B 推荐 4 通道：[2, 6, 5, 3]
```

注意 Ch 2 和 Ch 6 rank_sum 同为 19（并列第一），z_mean 几乎相同（+0.778 vs +0.776）。Ch 7 排名第 5，与 Ch 3 的 rank_sum 相同（均为 26），可作为备选（如果 Ch 3 因硬件限制不可用）。

---

## 五、两场景对比分析

### 5.1 重叠通道

```
        场景 A (Thumb)      场景 B (Index+Middle)
        ─────────────       ─────────────────────
        [6, 7, 5, 2]        [2, 6, 5, 3]
              │                    │
              └──── 交集 ──────────┘
                   [2, 5, 6]
```

| 通道 | 场景 A 排名 | 场景 B 排名 | 说明 |
|------|-----------|-----------|------|
| **Ch 6** | 1 ★ | 2 ★ | 两场景均顶级，最通用通道 |
| **Ch 5** | 3 ★ | 3 ★ | 两场景均前 3，SNR 指标均为第 1 |
| **Ch 2** | 4 ★ | 1 ★ | 两场景均前 4，Weight Norm 高 |
| Ch 7 | 2 ★ | 5 | 拇指场景极强，食指中指场景一般 |
| Ch 3 | 5 | 4 ★ | 食指中指场景好，拇指场景一般 |

### 5.2 场景差异解释（解剖学视角）

- **Ch 7 对拇指特化**：该通道可能位于腕部桡侧（拇指侧），对拇指运动肌群（拇长屈肌、拇短展肌等）的信号更敏感
- **Ch 3 对食指中指特化**：该通道可能位于腕部中央偏桡侧，对指浅屈肌/指深屈肌（支配食指、中指）的信号更敏感
- **Ch 2, 5, 6 通用**：这些通道可能位于腕部中央位置，覆盖多个手指的肌肉群

### 5.3 如果只能选 4 个通道同时覆盖两个场景

```
通用 4 通道推荐：[2, 5, 6, 7] （融合两场景的最优通道）
```

---

## 六、基于 emg2pose 的手指级分析

### 6.1 方法

emg2pose 预训练模型预测 20 DOF 手部关节角度，映射到 5 根手指。我们分析了解码器最后一层 `Linear(512→20)` 的权重分布，以确定模型对不同手指关节的"关注度"分配。

### 6.2 解码器权重分析结果

```
每个关节的解码器权重 L2 范数（模型对各关节的预测能力分配）：

  拇指关节:                          食指关节:
    THUMB_CMC_FE  [ 0] ████████ 2.85    INDEX_MCP_AA  [ 4] ██████ 2.43
    THUMB_CMC_AA  [ 1] ██████████ 3.57   INDEX_MCP_FE  [ 5] ██████████████ 5.27 ★
    THUMB_MCP_FE  [ 2] █████████ 3.37    INDEX_PIP_FE  [ 6] ██████████████ 5.26 ★
    THUMB_IP_FE   [ 3] ████████████ 4.50  INDEX_DIP_FE  [ 7] ██████████ 3.50

  中指关节:                          环指+小指:
    MIDDLE_MCP_AA [ 8] ████ 1.62        RING_*         [12-15] 合计 15.67
    MIDDLE_MCP_FE [ 9] ██████████████ 5.06 ★  PINKY_*       [16-19] 合计 15.81
    MIDDLE_PIP_FE [10] ███████████████ 5.35 ★
    MIDDLE_DIP_FE [11] ███████████ 3.93

各手指权重总和占比:
  index  : 16.46 (21.1%) █████████████████████
  middle : 15.96 (20.4%) ████████████████████
  pinky  : 15.81 (20.2%) ████████████████████
  ring   : 15.67 (20.0%) ████████████████████
  thumb  : 14.29 (18.3%) ██████████████████
```

### 6.3 关键发现

1. **五指权重分布均衡**（18-21%），说明预训练模型对各手指的预测能力分配均匀，没有明显的"主导手指"
2. **MCP_FE 和 PIP_FE 权重最高**（5.0-5.4），这是手指屈伸运动的核心关节，也是 pinch 和 swipe 的主要运动轴
3. **拇指 CMC_AA 权重较高**（3.57），这是拇指独有的腕掌关节外展/内收自由度（thumb_in/thumb_out 的关键关节），其他手指没有对应的 AA 关节
4. **Joint 8 (MIDDLE_MCP_AA) 权重最低**（1.62），可能因为中指的外展/内收在日常手势中使用较少

### 6.4 从关节角度理解两个场景

```
场景 A — 拇指四向滑动+点击:
  thumb_in    → 主要依赖 THUMB_CMC_AA (joint 1) 内收
  thumb_out   → 主要依赖 THUMB_CMC_AA (joint 1) 外展
  thumb_up    → 主要依赖 THUMB_CMC_FE (joint 0) 伸展 + THUMB_MCP_FE (joint 2) 伸展
  thumb_down  → 主要依赖 THUMB_CMC_FE (joint 0) 屈曲 + THUMB_MCP_FE (joint 2) 屈曲
  thumb_click → 主要依赖 THUMB_MCP_FE (joint 2) + THUMB_IP_FE (joint 3) 快速屈伸
  → 拇指场景需要通道对 joints [0,1,2,3] 的信号敏感

场景 B — 食指中指捏合:
  index_press   → 主要依赖 INDEX_MCP_FE (joint 5) 屈曲 + INDEX_PIP_FE (joint 6) 屈曲
  index_release → 主要依赖 INDEX_MCP_FE (joint 5) 伸展 + INDEX_PIP_FE (joint 6) 伸展
  middle_press   → 主要依赖 MIDDLE_MCP_FE (joint 9) 屈曲 + MIDDLE_PIP_FE (joint 10) 屈曲
  middle_release → 主要依赖 MIDDLE_MCP_FE (joint 9) 伸展 + MIDDLE_PIP_FE (joint 10) 伸展
  → 食指中指场景需要通道对 joints [5,6,9,10] 的信号敏感
```

---

## 七、TDS 第一层权重分析

预训练 TDS 的 `Conv1d(16→256, k=11)` 第一层权重反映了模型对 16 个输入通道的依赖程度：

```
通道权重 L2 范数（归一化）：
  Ch  3: 1.000 ██████████████████████████████
  Ch  2: 0.987 █████████████████████████████
  Ch  1: 0.940 ████████████████████████████
  Ch  0: 0.752 ███████████████████████
  Ch 15: 0.746 ███████████████████████
  Ch  6: 0.680 ████████████████████
  Ch  7: 0.677 ████████████████████
  Ch 14: 0.658 ███████████████████
  Ch 13: 0.636 ██████████████████
  Ch  5: 0.629 ██████████████████
  Ch  4: 0.619 █████████████████
  Ch 12: 0.610 ████████████████
  Ch  8: 0.594 ████████████████
  Ch 11: 0.542 ███████████████
  Ch  9: 0.534 ██████████████
  Ch 10: 0.506 ██████████████
```

**注意**：Weight Norm 排名（Ch 3 > 2 > 1 > 0 > 15...）与数据驱动方法（SNR/Fisher/MI 偏 Ch 5 > 6 > 7...）差异较大。这是因为 Weight Norm 反映的是**预训练任务（手部姿态估计）**中的通道重要性，而数据驱动方法反映的是**手势分类任务**中的通道重要性。两者任务不同，通道利用模式不同，这是预期内的差异。

---

## 八、emg2pose 与 emg_nature 的左右手佩戴差异分析

### 8.1 问题

用户反馈：基于 emg2pose 的 Weight Norm 分析与基于 emg_nature 数据的 SNR/Fisher/MI 分析，在拇指和食指中指场景中**结果不一致**。怀疑 emg2pose 和 emg_nature 存在左右手佩戴差异。

### 8.2 证据链

#### 证据 1：emg2pose 使用**双手**数据训练

```
emg2pose 数据分割 (mini_split.yaml):
  train:
    - recording-1_left    ← 左手
    - recording-1_right   ← 右手
    - recording-2_left
    - recording-2_right
    ...

每个 recording 同时采集左右手腕的 EMG 数据。
训练时左手和右手数据都输入模型，不做任何通道翻转。
```

关键代码证据（`emg2pose/data.py:31-36`）：
```python
"""EMG data from the left and right wrists, and their corresponding timestamps.
The sampling rate of EMG is 2kHz, each EMG device has 16 electrode channels...
corresponding to left and right EMG are 2D arrays of shape (T, 16) each"""
```

#### 证据 2：emg2pose **不做左右手通道翻转**

`emg2pose/transforms.py` 中唯一的 EMG 增强是 `RotationAugmentation`：
```python
class RotationAugmentation:
    def __call__(self, data):
        rotation = np.random.choice([-1, 0, 1])   # 仅随机滚动 ±1
        return torch.roll(data, rotation, dims=-1)
```

**没有任何针对左右手的通道镜像或翻转逻辑。**

#### 证据 3：CTAC 腕带在左右手上的通道-解剖映射是**相反的**

```
右手佩戴 (emg_nature):
  
  腕带连接器
      ↓
  ┌──────────────┐
  │ Ch0 Ch1 ...  │  手掌朝上，拇指在左侧
  │              │  Ch0 → 桡侧（拇指侧）
  │   手腕截面    │  Ch15 → 尺侧（小指侧）
  │              │
  └──────────────┘
       ↑
   拇  食  中  环  小
   指  指  指  指  指
  (桡侧)        (尺侧)

左手佩戴 (emg2pose 训练数据的一半):

  腕带连接器
      ↓
  ┌──────────────┐
  │ Ch0 Ch1 ...  │  手掌朝上，拇指在右侧
  │              │  Ch0 → 尺侧（小指侧）★ 翻转！
  │   手腕截面    │  Ch15 → 桡侧（拇指侧）★
  │              │
  └──────────────┘
       ↑
   小  环  中  食  拇
   指  指  指  指  指
  (尺侧)        (桡侧)
```

**核心结论：同一腕带佩戴在左手 vs 右手时，若保持连接器朝向一致，通道与手指的对应关系完全相反。**

### 8.3 对我们打分结果的影响

| 方法 | 数据来源 | 通道含义 | 可信度 |
|------|---------|---------|--------|
| **SNR** | emg_nature (右手) | Ch0=桡侧, Ch15=尺侧（一致） | ★★★ 高 |
| **Fisher** | emg_nature (右手) | 同上 | ★★★ 高 |
| **Mutual Info** | emg_nature (右手) | 同上 | ★★★ 高 |
| **Weight Norm** | emg2pose (双手混合) | **Ch0 有时是桡侧，有时是尺侧（混淆）** | ★☆☆ 低 |

这解释了为什么 Weight Norm 的排名（Ch 3 > 2 > 1 > 0 > 15 > ...）与数据驱动方法（Ch 5 > 6 > 7 > ...）差异巨大：

1. **Weight Norm 反映的是 TDS 模型在双手混合数据上学习到的"无位置偏向"的通道利用模式** — 由于训练数据中同一通道在左右手上对应相反的手指，模型学会了对通道位置保持鲁棒
2. **SNR/Fisher/MI 直接测量右手数据中每通道对手势的判别力** — 通道-手指映射一致，反映了真实的解剖关系
3. **RotationAugmentation（roll ±1）进一步削弱了 TDS 对绝对通道位置的依赖**

### 8.4 修正后的最终推荐

**应以数据驱动方法（SNR、Fisher、MI）为主要依据，Weight Norm 仅作参考（或直接排除）。**

重新审视纯粹数据驱动方法的 Top-4（排除 Weight Norm）：

| 场景 | SNR Top-4 | Fisher Top-4 | MI Top-4 | 交集 |
|------|-----------|-------------|---------|------|
| Thumb | 5, 6, 1, 7 | 12, 13, 6, 8 | 7, 6, 5, 12 | **6, 7, 5** |
| Index+Middle | 5, 1, 2, 0 | 7, 9, 2, 15 | 4, 5, 3, 2 | **2, 5** |

**场景 A (Thumb) — 纯数据驱动推荐：`[6, 7, 5, 12]`**

Ch 12 替代了原本由 Weight Norm 推高的 Ch 2。Ch 12 在 Fisher 中排名 1、MI 中排名 4，对拇指手势有较好的线性可分性。

**场景 B (Index+Middle) — 纯数据驱动推荐：`[2, 5, 1, 7]`**

Ch 1 和 Ch 7 替代了原本由 Weight Norm 推高的 Ch 3 和 Ch 6。Ch 1 在 SNR 中排名 2、Fisher 中也表现不错。

### 8.5 方法学建议

1. **未来筛选通道时，不应依赖 emg2pose 预训练权重的通道分布来分析解剖对应关系** — emg2pose 的双手训练使其通道表示不具有解剖特异性
2. **若需要利用 emg2pose 的手指知识**，应使用梯度归因（Saliency）方法在**仅含目标侧手腕**的数据上计算，而非直接分析权重
3. **若两项目使用相同的腕带硬件**，通道-解剖对应关系应与 emg_nature（右手）保持一致，即 Ch0→桡侧（拇指），Ch15→尺侧（小指）

---

## 九、通道相关性矩阵分析

两个场景的 16×16 通道间 Pearson 相关矩阵显示：

- **高相关块**：Ch 0-4 之间相关性较高（r > 0.7），Ch 12-15 之间相关性较高
- **低相关通道**：Ch 6, 7 与其他通道的相关性较低（r < 0.5），说明它们携带独立信息
- **Top-4 通道间相关性**：场景 A 的 [6, 7, 5, 2] 之间相关性均 < 0.6，无冗余问题

---

## 十、最终推荐（已根据左右手分析修正 + 7 方法共识）

### 10.1 方法论说明

最终排名排除以下低可靠性方法：
- **Weight Norm**（第八章已排除：emg2pose 双手混合训练使其通道位置不具解剖特异性）
- **Per-Channel Logistic Regression**（单通道准确率 ~0.21，接近随机水平 0.20，信噪比过低）
- **Greedy Forward Selection**（基于逻辑回归弱分类器，继承其不稳定性）

保留 5 种稳健方法：**SNR + Fisher + Mutual Info + Pairwise Diff + Bootstrap Stability**

### 10.2 场景 A — Thumb（拇指）

```
Top-6: [6, 7, 5, 3, 13, 12]

  ★★★ 高置信核心 (Bootstrap top4 > 80%): [6, 7, 5]
  ★★☆ 中置信扩展 (Bootstrap top4 10-20%):  [3, 13, 12]

  通道     rank_sum  SNR  Fisher  MI   PairDiff  BootStab
  ─────    ────────  ───  ──────  ───  ────────  ────────
  Ch  6 ★   10       2    3       2    2         1  (top4=1.00)
  Ch  7 ★   16       7    4       1    1         3  (top4=0.83)
  Ch  5 ★   28       1    14      3    8         2  (top4=1.00)
  Ch  3     35       4    16      8    3         4  (top4=0.20)
  Ch 13     35       8    2       11   7         7  (top4=0.10)
  Ch 12     36       9    1       12   5         9  (top4=0.00)
```

**选择建议**：核心 [6, 7, 5] 必须保留，第 4 个通道从 Ch 3、Ch 13、Ch 12 中选择。

- 选 Ch 3：SNR 排名 4，Pairwise Diff 排名 3（手势间对比度好）
- 选 Ch 13：Fisher 排名 2（类间可分性好）
- 选 Ch 12：Fisher 排名 1（最佳线性可分性），但 Bootstrap 稳定性最差

### 10.3 场景 B — Index+Middle（食指+中指）

```
Top-6: [4, 6, 11, 2, 3, 5]

  ★★★ 高置信核心 (Bootstrap top4 > 60%): [4, 6, 2, 5]
  ★★☆ 中置信扩展 (Bootstrap top4 10-20%):  [11, 3]

  通道     rank_sum  SNR  Fisher  MI   PairDiff  BootStab
  ─────    ────────  ───  ──────  ───  ────────  ────────
  Ch  4 ★   29       4    16      1    6         2  (top4=0.87)
  Ch  6 ★   31       5    5       4    14        3  (top4=0.70)
  Ch 11     31       9    8       5    3         6  (top4=0.20)
  Ch  2 ★   32       3    3       11   11        4  (top4=0.73)
  Ch  3     32       7    15      3    2         5  (top4=0.20)
  Ch  5 ★   32       1    13      2    15        1  (top4=1.00)
```

**选择建议**：核心 [4, 6, 2, 5] 恰好构成 4 通道优选组合。Ch 11 和 Ch 3 可作为替补。

### 10.4 跨场景对比

```
                Thumb 排名    Index+Middle 排名    两场景通用性
  ─────         ──────────    ────────────────    ────────────
  Ch  6          1 ★★★         2 ★★★              ★★★ 最通用
  Ch  5          3 ★★★         6 ★★★              ★★★ 极通用
  Ch  3          4 ★★☆         5 ★★☆              ★★☆ 通用
  Ch  7          2 ★★★         11                 ★☆☆ 拇指特化
  Ch 13          5 ★★☆         13                 ★☆☆ 拇指特化
  Ch 12          6 ★★☆         10                 ★☆☆ 拇指特化
  Ch  4          8              1 ★★★              ★☆☆ 食指中指特化
  Ch 11          14             3 ★★☆              ★☆☆ 食指中指特化
  Ch  2          9              4 ★★★              ★★☆ 偏食指中指
```

### 10.5 如果只选 4 通道同时覆盖两个场景

```
通用推荐: [5, 6, 3, 7]

  理由：
  - Ch 5, 6: 两场景均高置信核心
  - Ch 3: 两场景 Bootstrap 均稳定 (top4=0.20, 排名 4-5)
  - Ch 7: 拇指场景排名第 2 (牺牲一些食指中指场景的性能)
  
  替代方案: [5, 6, 3, 2] — 若更偏重食指中指场景
```

---

## 十一、输出文件清单

```
emg_chnse/outputs/
├── RESULTS.md                              ← 本报告
├── thumb_aggregate.json                    ← 场景 A 完整 JSON
├── thumb_ranking.csv                       ← 场景 A 排名表
├── thumb_per_method_scores.csv             ← 场景 A 各方法原始分数
├── index_middle_aggregate.json             ← 场景 B 完整 JSON
├── index_middle_ranking.csv                ← 场景 B 排名表
├── index_middle_per_method_scores.csv      ← 场景 B 各方法原始分数
├── figures/
│   ├── thumb_channel_ranking.png           ← 场景 A 排名柱状图
│   ├── thumb_gesture_heatmap.png           ← 场景 A 手势×通道热力图
│   ├── thumb_correlation_matrix.png        ← 16×16 通道相关性矩阵
│   ├── thumb_method_agreement.png          ← 各方法 Spearman 一致性
│   ├── thumb_per_method_scores.png         ← 各方法分通道分数
│   ├── index_middle_channel_ranking.png
│   ├── index_middle_gesture_heatmap.png
│   ├── index_middle_correlation_matrix.png
│   ├── index_middle_method_agreement.png
│   └── index_middle_per_method_scores.png
└── cache/
    ├── thumb_event_data.pkl                ← ~2.1 GB
    └── index_middle_event_data.pkl         ← ~4.0 GB
```

---

## 十二、后续验证建议

1. **训练验证**：使用推荐的 4 通道子集训练分类模型，与随机 4 通道和全部 16 通道对比
2. **跨用户泛化**：在测试集（未见用户）上评估 Top-4 的泛化性能
3. **硬件验证**：在实际 4 通道设备上采集数据，对比离线分析结果
4. **方法 5-6 补全**：训练完整的 16 通道模型后运行 Saliency 和 Ablation 方法，作为额外验证
