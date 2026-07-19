# P2-S02：dense EGFDU 相对梯度路径

## P2-S01 为什么失败

P2-S01 的公式正对照只有 0.0068 ft，但合法路径为 41.99 ft。它把每口 source 井压成一个中位点，再用 10 个点外推目标井整条轨迹。最坏井的预测 surface 范围达到 500～800 ft，而且方向常与真实相反。

新版本不再预测绝对 surface 高度。目标井最后可见点已经给出准确结构坐标：

```text
U_anchor = TVT_last + Z_last
```

真正未知的是此后地层面沿目标井 XY 如何变化。

## 冻结表示

### dense source

每个 outer-train 井沿累计 XY 水平距离每 50 ft 保留一个 `(X,Y,EGFDU)` 控制点。每折仍完整排除 outer-valid 井。

### 每个目标控制点的局部梯度

目标井也每 50 ft 取控制点。对每个查询点：

1. 每口 source 井最多取一个离查询点最近的控制点；
2. 再选最近的 10 口不同井；
3. 使用与 S01 相同的查询点中心化、距离归一化权重拟合局部平面；
4. 丢弃平面绝对截距，只保留换回原始坐标单位后的 `gradient_x/gradient_y`。

### outer-train 梯度上限

对每个 outer fold，只用 source 井自身相邻 50 ft 控制点计算：

```text
abs_directional_slope = abs(delta_surface) / horizontal_distance
```

固定取这些合法斜率的 P99 作为梯度模长上限。局部梯度超出时只缩小模长，不改变方向；矩阵秩不足时梯度回退为 0。P99 不在验证分数上搜索。

### 相对路径积分

以最后可见点的相对 surface 为 0，对相邻目标控制点使用梯形积分：

```text
delta_surface_j
  = delta_surface_(j-1)
  + 0.5 * (gradient_(j-1) + gradient_j) dot (xy_j - xy_(j-1))

TVT_surface
  = last_visible_tvt
  + delta_surface
  - (Z - Z_anchor)
```

这一步天然删除跨井绝对 surface 截距和 Typewell 常数基准。

## 合法质量量

- 最近与第 10 邻井轨迹点距离；
- 局部平面残差、秩和条件数；
- raw/clipped 梯度模长与裁剪比例；
- 10 个邻点的最大方位空缺角，用于表示查询点是否被空间支撑包围；
- 相邻目标控制点的邻井集合更换率；
- 可见前缀回测 RMSE。

这些量只作诊断；本阶段不训练门控模型。

## 负对照

对每口 source 井单独做：

```text
negative_surface = 2 * median(source_surface) - source_surface
```

它保留井的位置、绝对中位高程和变化幅度，只反转井内地层倾角。若真实梯度路径没有优于该负对照，说明当前局部梯度聚合没有识别正确方向。

## 晋级

达到实验卡四条门槛后，才生成 outer-train leave-one-well-out 的正式特征缓存并接入冻结单模 LightGBM。失败只关闭本固定梯度表示。

