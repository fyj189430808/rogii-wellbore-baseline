# P2-F01 连续 GR offset 路径设计

## 目标

本实验只验证一个问题：D01 中逐块看起来很弱的五尺度 NCC，加入整段连续性和曲率约束后，能否形成一条稳定优于冻结 PF 中心的路径。

它不是新的地层曲面实验，也不改变 LightGBM。诊断通过后，才把固定路径和质量列作为一个新特征组加入 P2B00。

## 为什么中心固定为 PF ANCC

二阶段全部 773 井中：

- PF ANCC 独立路径 RMSE：`14.2433 ft`；
- Beam mean 独立路径 RMSE：`15.7736 ft`；
- PF 真值落在 ±40 ft：`97.5471%`；
- Beam mean 真值落在 ±40 ft：`97.4530%`。

因此首版只使用：

```text
center_tvt = last_visible_tvt + pf_ancc_delta
```

P2B00 的 `pred_tvt` 不能作中心，因为它是 LightGBM OOF 输出，再送入新 LightGBM 会变成 stacking。Beam 中心留作独立敏感性实验，不在首版中择优。

## 数据流

```text
当前井 MD/GR/TVT_input
+ 对应 Typewell TVT/GR
+ 冻结 pf_ancc_delta
        ↓
隐藏段 PF 中心 TVT
        ↓
每 50 ft、41 个 offset、5 个尺度的 NCC 得分张量
        ↓
支持度加权的 block × offset 观测代价
        ↓
二阶状态动态规划
        ↓
前向累计代价 + 后向回溯 + 前后总代价余量
        ↓
offset 控制点线性插值回逐行
        ↓
PF 中心 + offset = 连续修正路径
```

隐藏 TVT 不进入上述流程。合法路径先保存并生成指纹，评分脚本随后单独读取隐藏 TVT。

## 得分张量

一口井定义：

```text
H = 隐藏行数
T = 50 ft 控制块数量
K = 41 个 offset
S = 5 个平滑尺度
```

合法缓存保存：

```text
row_index       [H]
row_to_block    [H]
block_md_mid    [T]
offsets_ft      [K]
ncc_scores      [S, T, K]
pair_counts     [S, T, K]
```

缓存中严格禁止出现 `TVT`、surface、真实 offset、真实 rank 或任何 oracle 误差。

## 观测代价

对尺度 `s`：

```text
scale_loss = (1 - NCC_s) / 2
```

对每个 block 和 offset，只平均有效尺度。支持度定义为公共有效行比例乘以有效尺度比例。最终：

```text
cost = support × mean(scale_loss)
     + (1 - support) × (offset / 40)^2
```

GR 完整时主要听五尺度形态；GR 缺失或 Typewell 超出支撑时平滑回到 PF 零 offset。长 GR 缺口不插值为平线。

## 连续状态

状态为 `(offset, change)`：

- offset：`-40, -38, ..., 40 ft`；
- change：`-4, -2, 0, 2, 4 ft/50ft`；
- 新 offset 必须等于旧 offset 加新 change；
- 新旧 change 之差绝对值不超过 `2 ft`；
- 虚拟初始状态固定 `(0, 0)`。

这些都是硬约束，不再搜索一阶或二阶惩罚系数。动态规划输出全局最小代价路径；后向表用于计算每个控制点经过不同 offset 的全局代价余量。

## 对照与晋级

实验卡中的合成轨迹、可见前缀注入、全缺失回零是程序和信号正对照。隐藏段循环平移 GR 是信息负对照，逐块独立最佳 offset 是连续性负对照。

只有所有预注册门槛通过，才进入固定单模 LightGBM 的正式特征实验。失败只否定本文件中冻结的状态网格、NCC 表示和硬约束组合。

