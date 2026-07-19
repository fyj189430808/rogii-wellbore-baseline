# P2-S03：最近 dense 轨迹点 surface 路径

## 固定算法

每折完整排除验证井，只使用 P2-S02 已冻结的 50 ft outer-train EGFDU 点。目标井每 50 ft 取控制点，对每个控制点只找一个最近 source 轨迹点：

```text
surface_hat_i = EGFDU(nearest legal source point to target XY_i)
offset = median(TVT_input + Z - surface_hat) on visible prefix
TVT_surface_i = surface_hat_i + offset - Z_i
```

控制点之间按累计水平距离线性插值。该方法没有局部平面、梯度、积分上限、邻井数或可调权重，因此不会把跨井高度差外推成数百英尺斜率。

同时保存最近轨迹距离、source 井切换比例、前缀校准 RMSE 和井内梯度反转负对照。达到实验卡门槛才允许进入 LightGBM 特征阶段。

## 防泄漏

- 每个 outer fold 的 dense 点只来自另外四折。
- 目标合法函数白名单为 `MD/X/Y/Z/TVT_input`。
- 目标 TVT 和自身 EGFDU 仅在路径完成后评分。
- dense cache SHA 固定，防止静默复用其他表示。

