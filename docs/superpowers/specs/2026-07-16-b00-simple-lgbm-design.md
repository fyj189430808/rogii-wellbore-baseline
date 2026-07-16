# B00 最简合法单模 LightGBM 设计

## 1. 目标

在冻结的 `spatial_pad_1000_v1` 五折上，训练一个且仅一个 LightGBM 回归模型，得到第一份可复现、无同井训练覆盖、无邻井标签泄漏的单模 CV。

这个实验只回答一个问题：只使用当前水平井自身的基础几何、隐藏段进度和原始 GR，固定 LightGBM 能达到多少 CV。

它不追求复现历史缓存中的 `10.4733`。历史结果使用普通按井 GroupKFold、旧缓存和更复杂特征，不能作为本实验的严格对照。

## 2. 不同方案及选择

考虑过三个方案：

1. 当前井几何、进度和原始 GR：信息最基础，同时保留比赛的主要测井信号。
2. 只使用几何和进度：更纯粹，但完全丢弃 GR，不适合作为后续 GR 特征实验的共同基线。
3. 加入 PF、Beam 和 Typewell：可能更强，但构建复杂、当前干净项目尚不能复现，也不属于“最简单”实验。

采用方案 1。模型、target、fold 和后处理从此冻结，后续实验只增加一个命名特征组。

## 3. 输入与输出

每口训练井读取：

- `MD`：测量深度，单位 ft；
- `X`、`Y`、`Z`：井轨迹坐标，项目当前按 ft 解释；
- `GR`：水平井伽马测井值，可缺失；
- `TVT_input`：可见前缀中的 TVT，隐藏评价段为空；
- `TVT`：只用于构造训练 target 和最终验证评分。

固定评价区为：

```python
hidden_mask = horizontal_df["TVT_input"].isna()
```

模型预测目标为：

```text
target_delta = hidden_TVT - last_visible_TVT_input
```

最终绝对 TVT 为：

```text
pred_tvt = last_visible_TVT_input + predicted_delta
```

## 4. 精确特征表

模型固定使用以下 12 个特征，不自动扫描或追加其他列：

| 特征名 | 公式 | 单位 | 作用 |
|---|---|---:|---|
| `last_visible_tvt` | 最后一个非空 `TVT_input` | ft | 当前井的可见 TVT 锚点 |
| `md_since_visible_end` | `MD - last_visible_MD` | ft | 隐藏点离预测起点多远 |
| `hidden_fraction` | `(MD - first_hidden_MD) / max(last_hidden_MD - first_hidden_MD, 1)` | 无量纲 | 隐藏段相对进度 |
| `x_current` | 当前行 `X` | ft | 当前空间位置 |
| `y_current` | 当前行 `Y` | ft | 当前空间位置 |
| `z_current` | 当前行 `Z` | ft | 当前井眼垂向位置 |
| `dx_from_visible_end` | `X - last_visible_X` | ft | 相对起点的 X 位移 |
| `dy_from_visible_end` | `Y - last_visible_Y` | ft | 相对起点的 Y 位移 |
| `dz_from_visible_end` | `Z - last_visible_Z` | ft | 相对起点的 Z 位移 |
| `dxy_from_visible_end` | `sqrt(dx^2 + dy^2)` | ft | 平面位移长度 |
| `gr_raw` | 当前行原始 `GR` | 原数据单位 | 当前点的直接测井信号 |
| `gr_missing` | 原始 `GR` 缺失时为 1，否则为 0 | 0/1 | 告诉模型 GR 是否可用 |

`gr_raw` 保留原始 NaN，由 LightGBM 原生处理。除上述 12 列外，不允许使用 Well ID、行 ID、surface、Typewell、PF、Beam、邻井、旧模型预测或任何隐藏 TVT 派生特征。

## 5. 固定模型

只使用 `lightgbm.LGBMRegressor`。参数完全读取 `rogii_clean/configs/lgbm_feature_baseline_v1.json`：

- `n_estimators=1734`；
- `learning_rate=0.00934485794382918`；
- `num_leaves=64`；
- `random_state=29`；
- 无 early stopping；
- 行权重全部为 1；
- 不训练第二个模型；
- 不进行参数搜索；
- 不做输出平滑、收缩、融合或投影。

## 6. 固定五折数据流

1. 读取冻结注册表 `artifacts/folds/spatial_pad_1000_v1.csv`，并核对 SHA-256。
2. 逐井读取 horizontal CSV，只对自然隐藏行构造上述 12 个特征和 target。
3. 将 773 口井合并为 3,783,989 条隐藏行样本。
4. 对每个 outer fold，训练集只包含其他四折的完整井，验证集只包含当前折完整井。
5. 每折重新创建一个相同参数的 LightGBM；不拟合任何跨折标准化或邻井统计。
6. 把预测 delta 加回该井 `last_visible_tvt`，写入该折 OOF 位置。
7. 五折完成后，使用统一评分器计算 carry、micro、macro、每折、median、P90、worst 和胜井率。

这里“一套单模”指模型家族、参数和特征固定为一套；交叉验证仍必须每折独立训练一个实例，否则会把验证井标签带入模型。

## 7. 运行顺序

1. 用 3 口井检查特征公式、shape、缺失值和 target。
2. 运行固定 fold 0，记录训练行数、验证行数、耗时、内存、特征重要性和指标。
3. 如果 fold 0 没有数据错位、非有限预测或泄漏检查失败，继续运行其余四折。
4. 支持按 fold 保存预测和模型，中断后只重跑缺失 fold。
5. 五折全部存在且配置指纹一致后再合并 OOF 和评分。

fold 0 分数差不会自动停止实验；只要实现检查通过，仍运行完整五折，因为各折 carry RMSE 已显示明显差异。

## 8. 错误检查与防泄漏

运行前必须断言：

- 注册表为 773 井、290 pad、3,783,989 个隐藏行；
- 注册表 SHA-256 与冻结配置一致；
- 每口井至少有一个可见点和一个隐藏点；
- 每口井 `TVT_input` 是连续的“可见前缀 + 隐藏后缀”；
- 每口井隐藏行数与注册表一致；
- 同一个 pad 只属于一个 fold；
- 训练井与验证井没有交集；
- 模型特征名精确等于上述 12 列；
- 除允许缺失的 `gr_raw` 外，特征和 target 全部有限；
- OOF 每一行恰好预测一次。

防泄漏测试必须证明：删除或打乱验证井隐藏 `TVT` 后，验证特征逐位不变。隐藏 `TVT` 只允许在 target 构造和最终评分两个位置出现。

## 9. 测试设计

新增以下测试：

1. 人工小井特征公式测试：手算 12 个特征并逐项比较。
2. GR 缺失测试：`gr_raw` 保持 NaN，`gr_missing` 正确为 1。
3. 隐藏标签隔离测试：改变隐藏 `TVT` 不得改变任何特征。
4. 固定特征清单测试：多列输入不能让 surface 等列自动进入模型。
5. fold 隔离测试：任一 fold 的训练井、验证井和 pad 均无交集。
6. 小型端到端测试：使用很少的树训练合成数据，确保 delta 能正确还原为绝对 TVT。

## 10. 实验产物

实验编号为 `B00_simple_lgbm_v1`，保存：

```text
artifacts/B00_simple_lgbm_v1/
├── config.json
├── metrics.json
├── per_well.csv
├── predictions.parquet
├── feature_list.json
├── parameter_list.json
├── runtime.json
├── feature_importance.csv
├── fold_0/
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
└── conclusion.md
```

大型预测、模型和缓存只保存在本地，不提交 GitHub。实验登记摘要写入 `experiments/registry.jsonl`。

## 11. 成功与失败的解释

本实验成功的最低标准是：完整、可复现、无泄漏地得到五折 OOF，而不是必须超过某个分数。

若分数优于 carry，只能说明这 12 个特征和固定 LGBM 有效；不能说明 LGBM 参数最优。

若分数不如 carry，只能否定这组最简特征与固定参数的组合，不能否定 GR、几何或 LightGBM 整个方向。
