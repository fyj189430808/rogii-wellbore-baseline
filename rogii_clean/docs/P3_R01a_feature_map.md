# P3-R01a 井级特征映射

## 1. 第一版只解决什么问题

第一版只预测每口井两个残差系数：

```text
residual(t) = a0 + a1 * t
t = (MD - hidden_start_MD) / (hidden_end_MD - hidden_start_MD)
```

- `a0`：整条基础路径的常数偏差，单位 ft；
- `a1`：从隐藏段开头到结尾累计增加的线性偏差，单位 ft；
- 最终路径：`P2P02_pred_tvt + predicted_a0 + predicted_a1 * t`。

训练单位是一口井，不是隐藏段的一行。开发集最终应是 `657 行 × 特征数`。

## 2. 最重要的数据边界

### 2.1 每个 outer fold 都必须重算的动态量

以下内容和 P2-P02 基础预测有关，不能直接复用现有全局五折 OOF：

1. 外层验证井的 P2-P02 预测；
2. 外层训练井的纯内层 OOF P2-P02 预测；
3. 用内层 OOF 残差拟合的训练井 `a0`、`a1`；
4. 从实际 P2-P02 预测路径计算的开头 100、200、500 ft 的 `U_pred` 坡度。

原因：例如外层验证 fold 0 时，现有 fold 1 的全局 OOF 模型曾使用 fold 0 标签训练。若直接拿它的残差训练 R01a，会把 fold 0 信息间接传给 R01a。

### 2.2 只需要计算一次的静态合法量

以下特征只依赖本井测试时可见信息、冻结 PF、冻结 Beam 或 Typewell，可以按井缓存一次：

- PF 五条冻结路径及其分歧；
- PF 与 Beam 的分歧；
- GR 有效率、最长缺口、有效证据数、固定温度 ESS；
- 可见前缀 `U=TVT_input+Z` 的坡度；
- 井轨迹几何；
- Typewell 与可见前缀的匹配质量。

## 3. 推荐首版特征

### 3.1 PF 路径，共 8 列

来源：

```text
artifacts/P2_P01_multiseed_pf_mean_v1/legal_cache/<well_id>.parquet
```

原始列：

```text
last_visible_tvt
pf128_mean_tvt
pf128_scale_3_delta
pf128_scale_5_delta
pf128_scale_8_delta
pf128_scale_12_delta
pf128_seed_std
row_index
```

五条路径分别是：

```text
mean_path = pf128_mean_tvt
scale3_path = last_visible_tvt + pf128_scale_3_delta
scale5_path = last_visible_tvt + pf128_scale_5_delta
scale8_path = last_visible_tvt + pf128_scale_8_delta
scale12_path = last_visible_tvt + pf128_scale_12_delta
```

首版保存：

| 特征 | 公式 |
|---|---|
| `pf_mean_end_delta` | `mean_path[-1] - last_visible_tvt` |
| `pf_s3_end_delta` | `scale3_path[-1] - last_visible_tvt` |
| `pf_s5_end_delta` | `scale5_path[-1] - last_visible_tvt` |
| `pf_s8_end_delta` | `scale8_path[-1] - last_visible_tvt` |
| `pf_s12_end_delta` | `scale12_path[-1] - last_visible_tvt` |
| `pf_mean_average_slope` | `(mean_path[-1]-mean_path[0]) / hidden_MD_span` |
| `pf_end_delta_range` | 五条末端 delta 的 `max-min` |
| `pf_average_slope_range` | 五条全段平均坡度的 `max-min` |

不把十个两两差全部放进首版岭回归，因为它们是五个末端值的线性组合，没有新增线性信息。若以后换小 LightGBM，可再登记加入十个显式差值。

### 3.2 PF 与 Beam 分歧，共 2 列

Beam 来源：

```text
artifacts/F05a_deterministic_candidate_cache_v1/per_well/<well_id>.parquet
```

使用 `beam_mean_d`，它是相对最后可见 TVT 的 Beam 平均路径。令：

```text
pf_delta = pf128_mean_tvt - last_visible_tvt
gap = pf_delta - beam_mean_d
```

首版保存：

| 特征 | 公式 |
|---|---|
| `pf_beam_end_gap` | `gap[-1]` |
| `pf_beam_abs_gap_growth` | 后半段 `mean(abs(gap))` 减前半段对应均值 |

### 3.3 随机路径分散，共 1 列

| 特征 | 公式 |
|---|---|
| `pf_seed_std_end` | 冻结缓存 `pf128_seed_std[-1]` |

现有完整 128 seed 路径缓存尚未覆盖全部 657 口开发井，因此首版不等待真正的 `P90-P10` 高低分支距离。现有 `pf128_seed_std` 是最快的合法替代量。

### 3.4 GR 证据和 ESS，共 4 列

来源：

```text
artifacts/P3_D01_pf_observation_weight_audit_v1/per_well.csv
```

必须白名单读取，禁止把同表中的 `p2p02_rmse`、各路径 RMSE、oracle 等目标诊断列读进特征表。

| 特征 | 公式或原始列 |
|---|---|
| `gr_observed_fraction` | `observed_gr_fraction` |
| `gr_longest_gap_fraction` | `longest_gr_gap_md_ft / max(hidden_MD_span, 1)` |
| `gr_effective_evidence_fraction` | `effective_observation_count / hidden_rows` |
| `pf_scale8_normalized_ess` | `scale_8_normalized_effective_sample_size` |

### 3.5 可见前缀结构坡度，共 4 列

来源是原始水平井中 `TVT_input` 有限的可见前缀。已有函数：

```text
src/f01_features.py::fit_visible_u_trend
```

令 `U_visible = TVT_input + Z`，分别在最后 100、200、500 ft 上对 `U_visible ~ MD` 做 OLS：

```text
visible_u_slope_100
visible_u_slope_200
visible_u_slope_500
visible_u_slope_100_minus_500
```

最后一列是最便宜的倾角变化/曲率断点代理。

### 3.6 P2-P02 基础预测开头坡度，共 4 列，动态重算

对每个 outer/inner 实际生成的基础预测，令：

```text
U_pred = P2P02_pred_tvt + Z_hidden
```

在隐藏段开头 100、200、500 ft 上拟合 `U_pred ~ MD`：

```text
base_u_slope_100
base_u_slope_200
base_u_slope_500
base_u_slope_100_minus_500
```

这四列不能放进全局静态缓存；它们必须跟随当前 inner/outer 的基础预测生成。

### 3.7 井轨迹，共 5 列

`azimuth_deg` 可直接复用：

```text
artifacts/P3_shadow_holdout_v1/metadata.csv
```

首版保存：

| 特征 | 公式 |
|---|---|
| `azimuth_sin` | `sin(azimuth_deg)` |
| `azimuth_cos` | `cos(azimuth_deg)` |
| `hidden_dxy_per_md` | 隐藏段首尾水平距离 / MD span |
| `hidden_abs_dz_per_md` | 隐藏段首尾 `abs(dZ)` / MD span |
| `hidden_tortuosity` | 隐藏段逐步三维弦长之和 / 首尾三维弦长，退化时置 1 |

这些量只使用测试时提供的 MD/XYZ。

### 3.8 Typewell 前缀匹配质量，共 4 列

现成合法诊断：

```text
artifacts/RF03_D0_prefix_alignment_v1/offset_scores.csv
artifacts/RF03_D0_prefix_alignment_v1/per_well_margins.csv
src/f03a_prefix_reliability_features.py::build_well_prefix_reliability_features
```

首版取：

```text
f03a_tail_zero_raw_ncc
f03a_tail_zero_affine_mae
f03a_tail_valid_pair_fraction
f03a_tail_ncc_margin
```

只使用最后 1000 ft 可见前缀，更贴近预测起点。

## 4. 首版 shape

上述首版总计 32 列：

```text
PF 8
+ PF-Beam 2
+ seed 分散 1
+ GR/ESS 4
+ 可见前缀 U 4
+ 动态基础预测 U 4
+ 井轨迹 5
+ Typewell 匹配 4
= 32
```

每个外层 fold 大致为：

```text
X_outer_train: [约 525, 32]
y_outer_train: [约 525, 2]
X_outer_valid: [约 131, 32]
```

32 列对岭回归不算多；正则化会处理共线性。第一版不要增加 8 个控制点、逐行特征或邻井先验。

## 5. 缺失值与标准化

每个 outer fold 内分别执行：

1. 把正负无穷改成 NaN；
2. 只用 outer-train 井计算每列中位数；
3. 用该中位数填充 outer-train 和 outer-valid；
4. 只对确实出现缺失的原始列追加一个缺失指示列；
5. `StandardScaler` 只在 outer-train 上拟合；
6. `a0` 与 `a1` 分别训练一个岭回归，避免两个目标尺度互相影响。

推荐使用 `Pipeline(SimpleImputer(add_indicator=True), StandardScaler(), Ridge(...))`，并确保 pipeline 只 `fit` outer-train。

## 6. 首版明确延后的内容

- 邻井 `a0/a1` 先验：它需要在每个 outer fold 内只用 inner-OOF 目标重建，首版会拖慢且增加泄漏面；
- 128 条 seed 的真实高低分支分位差：当前共享路径缓存未覆盖全部开发井；
- 二次项或四控制点：只有线性 R01a 通过后再做；
- 将修正路径塞回逐行 LightGBM：第一步直接评价修正路径本身。

