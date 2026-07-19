# P2-S01：严格 outer-fold EGFDU 地层面路径

## 要回答的问题

只使用每个 outer fold 的训练井地层面，能否预测验证井沿 XY 方向的结构变化，再由当前井可见前缀把结构面转换为 TVT 路径。

地层面血缘审计已经确认：

```text
U = TVT + Z ≈ surface + Typewell marker
```

因此目标井不需要 Typewell 的 Geology。先预测结构面 `surface_hat(X,Y)`，再在可见前缀估计常数：

```text
offset = median(TVT_input + Z - surface_hat)
TVT_surface = surface_hat + offset - Z
```

## 为什么先用这个简单版本

旧高分 notebook 已经固定使用“每井中位代表点 + 最近 10 井局部平面”。它比 F07R 的轨迹距离、方位筛选、前缀外推和 top-8 聚合少很多环节，能最直接回答《一阶段总结》提出的空间曲面问题。

六个 surface 近乎平行。首版固定 `EGFDU`，因为 773/773 口训练井都有有效 surface 和 marker；不同时加入六个重复标签。

## 冻结算法

### outer-train source 表

每折只取 `fold != outer_fold` 的井。每口 source 井生成一个等权代表点：

```text
source_x = median(X)
source_y = median(Y)
source_surface = median(EGFDU)
```

### 目标井控制点

用 XY 累计水平距离每 50 ft 选一个控制点，并强制保留首行、最后可见行和末行。局部平面只在控制点求解，之后按水平距离线性插值到逐行，避免数百万次小矩阵求解。

### 局部平面

查询最近 10 口 source 井。以当前查询点为原点、邻井距离中位数为尺度：

```text
surface_j = c0 + c1 * dx_j / scale + c2 * dy_j / scale
weight_j = 1 / (distance_j / scale + 0.001)
```

使用加权最小二乘。预测值是查询点处的截距 `c0`。矩阵秩不足时回退为加权中位数。这样避免直接用数百万量级绝对坐标造成病态。

### 前缀校准与隐藏路径

合法路径只读取目标井的 `MD/X/Y/Z/TVT_input`。用全部可见前缀残差中位数校准绝对偏移，然后只输出隐藏行：

```text
surface_tvt = surface_hat + prefix_offset - Z
surface_tvt_delta = surface_tvt - last_visible_tvt
```

同时保存最近 source 距离、第 10 source 距离、局部平面加权 RMSE、矩阵秩和前缀校准 RMSE，供后续判断支撑质量。

## 三条证据

1. 主路径：严格 outer-fold EGFDU 局部平面。
2. 空间负对照：每折固定打乱 source 的 `surface` 与 `X/Y` 对应关系。
3. 公式正对照：目标井自身 EGFDU 只在合法路径完成后单独读取，验证公式上限，不进入正式特征。

## 防泄漏

- source 表只包含 `fold != outer_fold`。
- 目标合法读取列不含 `TVT` 和六个 surface。
- 正对照、真值评分和合法路径分别调用。
- 后续若进入 LightGBM，outer-train 井的训练特征也必须在查询时排除该井自身，不能使用入样本 surface。
- 修改目标井隐藏 TVT 或 surface 后，合法路径必须逐位不变。

## 晋级

本诊断达到实验卡全部门槛后，才生成五套“outer-valid 严格排除 + outer-train leave-one-well-out”的正式特征缓存，并按冻结单模 LightGBM 门槛运行。

