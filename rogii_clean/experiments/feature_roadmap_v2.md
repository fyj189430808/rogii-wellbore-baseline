# 单模 LightGBM 特征实验路线（Feature Roadmap）

> 版本：`feature_roadmap_v2`  
> 主目标：在固定单模 LightGBM、固定空间 CV、固定 target 和冻结 PF/Beam 的前提下，逐组验证测试期合法特征的真实增量价值。  
> 核心原则：**每次只改变一个特征组；先做边际价值，再做条件价值；负结果只否定当前实现，不扩大解释。**

---

## 0. 全局实验合同

### 0.1 固定不变项

所有实验统一使用：

- CV：`spatial_pad_1000_v1`
- target：`TVT - last_known_TVT`
- 模型：单个 LightGBM
- LightGBM 参数：冻结
- 树数：`1734`
- seed：`29`
- 评价行：自然隐藏段，即 `TVT_input.isna()`
- 无 early stopping
- 无模型融合
- 无 stacking
- 无 learned trajectory
- 无 model package
- 无同井训练 TVT、surface、contact 或完整轨迹检索
- 不改变 PF 参数、scale、hold、粒子数和 seed 集
- 不改变 target、fold、训练样本范围和评价样本范围
- 每次只新增一个定义明确的特征组

### 0.2 固定对照

每个实验统一比较：

1. carry-forward
2. 冻结 PF
3. B0 单模 LightGBM
4. B0 + 当前唯一新增特征组

### 0.3 统一晋级标准

一个特征组只有同时满足以下条件才晋级：

- fold 0 改善至少 `0.15 ft`
- fold 1 与 fold 0 同方向
- 五折 micro RMSE 改善至少 `0.10 ft`
- 至少 `4/5` 折同方向
- 井级聚类 bootstrap 的 95% CI 上界小于 0
- P90 逐井 RMSE 不恶化超过 `0.5 ft`
- 相应负对照不产生类似改善
- 收益不能仅来自极小覆盖切片

### 0.4 停止规则

出现以下任一情况立即停止当前实验：

- fold、井数、评价行数或 row hash 与 B0 不一致
- 新特征读取验证井隐藏 TVT 或其派生量
- 冻结 PF/Beam cache 与当前 fold 不匹配
- 负对照得到相近改善
- fold 0 明显恶化
- 主要收益只来自少量井或少量有效行
- 当前实现的 oracle 也没有足够空间

### 0.5 Bootstrap 统一定义

每口井保存：

```text
well_id
n_rows
sse_b0
sse_new
rmse_b0
rmse_new
```

井级 bootstrap 每次以井为单位重采样，并重新计算：

\[
RMSE=\sqrt{\frac{\sum_w SSE_w}{\sum_w n_w}}
\]

最终统计：

\[
\Delta RMSE=RMSE_{new}-RMSE_{B0}
\]

不得直接对逐井 RMSE 差求均值代替比赛 micro RMSE。

---

## 1. 可复现性与数据血缘合同

### 1.1 B0 必须冻结为不可变对象

B0 必须保存以下 hash：

```text
baseline_id
fold_hash
train_well_hash
train_row_hash
eval_row_hash
feature_list_hash
lightgbm_config_hash
pf_cache_hash
beam_cache_hash
raw_data_hash
code_commit
```

推荐固定：

```text
baseline_id = B0_v1
```

后续任何 B0 特征调整都必须升级版本号，不得覆盖原基线。

### 1.2 PF/Beam 必须严格折外

每口训练井使用的冻结 PF/Beam 预测必须来自不包含该井的合法外层折。

每个 cache 至少记录：

```json
{
  "well_id": "...",
  "outer_fold": 0,
  "fit_wells_hash": "...",
  "pf_config_hash": "...",
  "seed_list_hash": "...",
  "prediction_row_hash": "..."
}
```

### 1.3 隐藏真值不变性测试

每个特征组都必须通过：

1. 删除验证井隐藏 TVT 后，特征完全不变
2. 随机修改验证井隐藏 TVT 后，特征完全不变
3. 删除训练 surface 后，非 F07 相关特征完全不变
4. 固定 seed 重跑，特征 hash 和预测 hash 一致
5. 验证井、验证 pad 不进入 learned statistics 来源

### 1.4 等维负对照

每个真实特征组必须配一个等列数或同结构负对照，例如：

- 井内 block shift
- 井内循环平移
- 同分布跨井置乱
- 保留缺口长度分布的 mask 重排
- 保留 Typewell 自相关的 block permutation

判定要求：

\[
B0+F_{real}<B0+F_{negative}
\]

---

## 2. 实验阶段设计

整个路线分为两阶段。

### 2.1 S 系列：边际价值

每组独立与 B0 比较：

```text
S01 = B0 + F01
S02a = B0 + F02a
S02b = B0 + F02b
S03a = B0 + F03a
...
```

目标是回答：

> 该信息源单独加入 B0 后是否有稳定增量？

### 2.2 C 系列：条件价值

边际实验完成后，按五折改善排序逐步组合：

```text
C01 = B0 + 最强特征组
C02 = C01 + 第二强特征组
C03 = C02 + 第三强特征组
```

每次仍然只改变一个特征组。

目标是回答：

> 该特征在已有强特征存在时，是否仍提供独立信息？

不得直接把全部晋级组一次性合并。

---

# 3. B0：private-safe 单模基线

## 实验编号

`B0`

## 实验名称

冻结单模 LightGBM 基线

## 唯一假设

当前已理解的合法几何、原始 GR、冻结 PF/Beam 和 Typewell 全局摘要，足以建立一个稳定、可复现、private-safe 的单模参考点。

## 合法输入

- 当前井：`MD/X/Y/Z/GR/TVT_input`
- 当前 Typewell：`TVT/GR`
- 冻结 OOF PF/Beam
- outer-train 安全统计
- 明确列入 `feature_list.json` 的固定特征

## 明确禁止

- 同井训练副本
- 验证井 surface
- 验证井隐藏 TVT
- contact reconstruction
- 全量 formation imputer
- learned/OOF model prediction stacking

## 正对照

B0 应至少学到相对 carry、基础几何和冻结 PF/Beam 的有效关系。

## 负对照

将 target 在井间置乱后，性能必须明显崩溃。

## 成功标准

- 完整五折 OOF
- 相同 seed 重跑结果一致
- 所有 fold 行数、井数、特征列表完全一致
- artifacts 可复现

## 产物

```text
config.json
feature_list.json
feature_lineage.json
predictions.parquet
per_well.csv
metrics.json
runtime.json
cache_manifest.json
leakage_tests.json
```

---

# 4. F01：可见前缀 U 的多窗口倾角与稳定性

## 实验编号

`F01`

## 实验名称

可见前缀 `U=TVT_input+Z` 倾角与稳定性

## 唯一假设

最后可见段的局部构造倾角、曲率和稳定性，对隐藏段 TVT 残差有额外信息。

## 新增特征

按最近 `50/100/200/500/1000 ft`：

```text
robust U slope
OLS U slope
slope std
short-long slope difference
positive-slope ratio
curvature
second-difference MAD
last-window vs full-prefix slope difference
linear extrapolation delta
```

所有窗口按 MD 英尺定义，不按行数定义。

## 正对照

在可见前缀内部早切点回放时，倾角外推应优于 carry。

## 负对照

- 井内打乱 MD 顺序后重算
- 所有 slope 置零
- 将整套 slope 特征在相近井之间置乱

## Oracle

用隐藏真实 U 拟合最佳直线、锚定样条和分段线，仅报告上限。

## 失败解释边界

失败只否定当前窗口、公式和单模使用方式，不否定前缀 U 信息源。

---

# 5. F02：GR 可观测性与信号质量

F02 拆成两个相互独立的实验，避免将“是否有观测”和“观测是否稳定”混为一组。

## F02a：纯 GR 缺失几何

### 唯一假设

模型需要知道某一行及其邻域的 GR 是否真正可观测。

### 新增特征

仅使用原始缺失 mask，不使用 GR 数值：

```text
gr_is_observed
gap_length_ft
distance_to_left_observed_ft
distance_to_right_observed_ft
relative_position_inside_gap
valid_fraction_50ft
valid_fraction_100ft
valid_fraction_200ft
leading_gap_flag
trailing_gap_flag
interpolated_between_two_observations_flag
forward_extrapolated_flag
backward_extrapolated_flag
well_hidden_valid_fraction
well_longest_gap_ft
```

### 实现要求

- 所有距离按 MD 英尺
- 不把实测、内插、前向填充、后向填充合并成同一状态
- 不对缺失状态填 0 表示“稳定”

### 正对照

PF 绝对误差应随 gap length 增大、valid fraction 降低、距最近实测点变远而变差。

### 负对照

- 井内循环平移完整 mask
- 以完整缺口 block 为单位重排
- 保持每井缺失率和缺口长度分布

### Oracle

逐行知道真值时，在 PF/carry/B0 中理想选择，只报告可靠性上限。

## F02b：实测 GR 信号质量

### 唯一假设

即使 GR 非缺失，不同区间的噪声、尖峰、平坦和非平稳程度也不同。

### 新增特征

只在真实观测点上计算：

```text
local GR MAD
first-difference MAD
second-difference MAD
local quantile range
spike fraction
flat-line fraction
local autocorrelation
local trend
early-vs-late distribution shift
local stationarity score
```

每个统计同时保存：

```text
effective_observed_count
effective_observed_fraction
```

### 实现要求

- 禁止在插值后的 GR 上计算“信号质量”
- 支持不足时置 NaN，不填 0
- LightGBM 自行处理 NaN

### 正对照

低质量分箱中的 PF/B0 误差应稳定更高。

### 负对照

在相同观测率和相似 GR 方差的井之间置乱质量特征。

---

# 6. F03：Typewell 对当前井的可信度与多尺度对齐

F03 拆为“前缀可信度”和“隐藏段候选得分面”两个阶段。

## F03a：可见前缀上的 Typewell 可信度

### 唯一假设

在评价隐藏段之前，应先判断 Typewell 是否适合当前水平井。

### 合法构造

在可见前缀上，真实 `TVT_input` 已知，可合法比较：

\[
GR_h(MD)\quad\text{与}\quad GR_t(TVT_{input})
\]

### 新增特征

```text
prefix raw NCC
prefix 2ft NCC
prefix 5ft NCC
prefix 10ft NCC
prefix 20ft NCC
prefix robust MAE
prefix derivative NCC
valid matched pair count
valid matched fraction
```

拟合：

\[
GR_h \approx a\,GR_t+b
\]

并新增：

```text
affine_a
affine_b
calibrated_MAE
calibrated_NCC
raw_vs_calibrated_gain
```

按连续 MD block 再计算：

```text
block NCC median
block NCC std
block best-offset median
block best-offset std
early-vs-late NCC difference
last_200ft vs full-prefix reliability
```

### 正对照

真实前缀 TVT 位置应稳定优于 `±10/20 ft` 错位。

### 负对照

- Typewell GR 循环平移
- 50–100 ft block permutation
- 使用 GR 分布接近但错误的 Typewell

### 价值

该实验回答：

> 当前 Typewell 是可靠模板、局部可靠模板，还是基本不可靠模板？

## F03b：多尺度 Typewell offset 得分面

### 唯一假设

低频 GR 形态和整个 offset 得分面，比原始逐点误差和单一最佳 offset 更稳定。

### 新增特征

在冻结 PF 路径和固定 TVT offset 网格上计算：

```text
raw robust MAE
affine-calibrated MAE
2/5/10/20ft NCC
first-derivative NCC
best offset
best score
non-local second-best offset
second-best score
peak separation
peak width
scale agreement
```

### 非极大值抑制

第二峰必须与第一峰相隔至少 `5 ft` 或一个估计峰宽。相邻 offset 格点不得被当成两个独立地质模式。

### 得分面软统计

对 offset 分数 \(S_k\) 构造：

\[
p_k=\frac{\exp(-S_k/T)}{\sum_j\exp(-S_j/T)}
\]

新增：

```text
soft_offset_mean
soft_offset_std
offset_entropy
score_skewness
best_basin_mass
second_basin_mass
inter_basin_distance
```

### 支持量

所有相似度必须附带：

```text
valid_pair_count
valid_pair_fraction
longest_valid_run
number_of_valid_blocks
```

### 负对照

- 保留 Typewell 自相关的循环平移
- 相似频谱 block permutation
- 错误但统计性质相近的 Typewell 区间

### Oracle

隐藏真实 TVT 下的最佳 offset、最佳 scale 和可辨识上限，只作诊断。

---

# 7. F04：branch-aware self-template

## 实验编号

`F04`

## 实验名称

当前井可见前缀的 `GR(TVT_input)` 自模板

## 唯一假设

同一工具、同一口井的前缀模板，在幅值和噪声模式上可能优于配对 Typewell。

## 核心风险

水平井可能多次经过同一 TVT，因此 `TVT → GR` 不一定是单值函数。

## 模板定义

至少按以下维度分支：

```text
TVT bin
sign(dTVT_input/dMD)
continuous prefix segment ID
```

即：

\[
GR_{self}=f(TVT,\operatorname{sign}(dTVT/dMD),segment)
\]

## 新增特征

```text
candidate TVT coverage flag
nearest observed TVT distance
same-direction support count
opposite-direction support count
number of visits
within-bin GR MAD
self-template predicted GR
self-template robust MAE
self-template NCC
self-template best offset
self-template peak separation
self vs typewell score difference
```

## 实现要求

- F03 与 F04 共用同一 scorer
- 唯一变化是模板来源
- 支持量和不确定性必须显式输入
- 多次经过同一 TVT 时不得简单平均后忽略离散度

## 正对照

前缀前半建模板，预测后半可见前缀，应优于困难错误模板。

## 负对照

选择以下条件相近但井 ID 不同的错误模板：

```text
GR 均值接近
GR 方差接近
Typewell 指纹接近
已知 TVT 范围接近
```

## 最低覆盖门槛

预先冻结，例如：

- 至少覆盖 20% 评价行
- 至少覆盖 30% 井
- 每个比较窗口至少 30 个真实有效点

## Oracle

隐藏真实 TVT 下评估 self-template 在高覆盖子集的理论上限。

---

# 8. F05：冻结 PF 的候选分布与内部不确定性

F05 先验证“候选路径几何”，再验证 raw ESS/entropy。

## F05a：候选路径分布几何

### 唯一假设

PF 候选的多模式、跨度和分歧增长，能描述模型什么时候容易整体走错。

### 新增井级特征

```text
endpoint D median
endpoint D IQR
endpoint D MAD
endpoint D full span
number of endpoint modes
largest inter-mode gap
largest mode mass
second mode mass
candidate pairwise distance median
candidate pairwise distance P90
PF mean vs candidate medoid distance
weighted mean vs weighted median distance
PF vs Beam divergence
```

### 新增行级特征

```text
row-wise candidate std
row-wise q10-q90 width
row-wise multimodality score
distance from PF mean to candidate median
```

### 分歧增长特征

\[
g=\operatorname{slope}\left(\operatorname{Std}_k(TVT_{k,i})\text{ 对 }MD_i\right)
\]

并保存：

```text
dispersion_at_start
dispersion_at_middle
dispersion_at_end
dispersion_growth_rate
late_divergence_fraction
```

### 正对照

候选跨度、多模式数和后段分歧增长应与 PF 绝对误差有稳定关系。

### 负对照

跨井置乱候选几何特征，但匹配：

```text
hidden length
GR valid fraction
PF baseline error stratum proxy
```

### Oracle

匹配候选结构的随机平滑路径池，与真实 PF 候选池比较 oracle。

随机池必须匹配：

```text
endpoint span
平均平滑度
二阶差分方差
路径相关长度
与 PF 均值的距离
```

## F05b：归一化 PF 内部统计

### 唯一假设

在候选路径几何之外，PF 内部权重退化和重采样过程仍有额外信息。

### 新增特征

```text
ESS / particle_count
mean ESS / valid_GR_count
minimum normalized ESS
fraction ESS < 0.1
resampling_count / trajectory_length
weight_collapse_count / valid_observation_count
loglik_gap / valid_observation_count
entropy / log(particle_count)
time_since_last_informative_update
valid_update_fraction
```

### 实现要求

不得直接使用未归一化的 raw ESS、raw entropy、raw cumulative likelihood 和 raw resampling count。

### 停止条件

若 F05a 无任何稳定信号，则 F05b 降低优先级或取消。

---

# 9. F06：horizon-matched prefix holdout

## 实验编号

`F06`

## 实验名称

合法候选的可见前缀伪 holdout 可靠性

## 唯一假设

候选在可见前缀内部多个预测 horizon 上的表现，能预测其在自然隐藏后缀上的可靠性。

## 关键改动

不再仅使用比例 cut：

```text
0.50 / 0.65 / 0.75
```

同时加入按物理距离定义的 horizon：

```text
250 ft
500 ft
1000 ft
```

根据前缀长度选择可支持的最长 horizon。

## Cut-specific 重建

每个 cut 都必须重新生成候选：

```text
cut 之前的 TVT_input：允许使用
cut 之后的 TVT：禁止使用
cut 之后的 GR：允许使用
```

不得使用完整前缀生成的候选回头评分早期 cut。

## 新增特征

对 carry、几何、PF、Beam、Typewell 候选分别保存：

```text
cut RMSE
gain vs carry
gain vs PF
rank
valid coverage
```

跨 cut 聚合：

```text
median gain
worst-cut gain
gain std
gain slope vs horizon
rank Spearman consistency
rank Kendall consistency
winner-family stability
same-family top2 rate
```

重点特征：

\[
\frac{\Delta gain}{\Delta horizon}
\]

用于识别“短期有效、长期快速失稳”的候选族。

## 覆盖要求

每个候选在伪 holdout 上必须满足：

```text
coverage >= 95%
```

否则该 cut 分数无效，不得只在少量有限预测点上计算 RMSE。

## 正对照

早期 cut 的 gain 和 rank 应预测下一段可见前缀误差。

## 负对照

在以下条件相近的井之间置乱：

```text
prefix length
hidden length
GR missing rate
candidate span
```

## Oracle

自然隐藏后缀候选排名与 prefix-cut 排名的一致性，只作诊断。

---

# 10. F07R：邻井相对 ΔU 结构先验

## 实验编号

`F07R`

## 实验名称

严格 outer-fold 的邻井相对 \(\Delta U\) 趋势

## 原 F07 的调整

不再尝试直接预测绝对 surface：

\[
U_{target}(X,Y)
\]

改为转移邻井相对走势：

\[
\Delta U_{neighbor}(s)=U_{neighbor}(s)-U_{neighbor}(s_0)
\]

其中：

\[
U=TVT+Z
\]

## 唯一假设

在排除验证 pad 后，空间接近、方向相似、轨迹长期平行的训练水平井，仍可能提供当前井长期低频结构趋势。

## 邻井选择

不得只用 median X/Y。

至少综合：

```text
trajectory minimum distance
azimuth difference
parallel overlap length
along-track distance
cross-track distance
heel distance
toe distance
```

## 新增特征

```text
neighbor_relative_U_prior
neighbor_relative_slope
neighbor_curvature
neighbor_MAD
nearest_trajectory_distance
azimuth_difference
parallel_overlap_length
support_well_count
support_effective_weight
neighbor_disagreement
```

## 当前井前缀校准

对每个邻井 prior，在可见前缀上拟合：

\[
U_{input}-U_{neighbor\ prior}=a+bs
\]

隐藏段使用：

\[
U_{prior,calibrated}=U_{neighbor\ prior}+a+bs
\]

## 数据来源

正式特征优先直接由 outer-train 的 `U=TVT+Z` 构造，不依赖六个 surface。

surface 仅保留用于 lineage 审计和 oracle 诊断。

## 严格 fold-safe 要求

为 outer-train 中每口井生成训练特征时，必须排除：

- 该井本身
- 该井所在 pad
- 必要时进入 purge 距离的轨迹

验证井只能使用 outer-train pads。

## 正对照

在验证井可见前缀上，校准后的邻井相对 prior 应优于全局常数和未校准邻井 prior。

## 负对照

- 置乱 XY
- 置乱井级 U 路径
- 保留距离但置乱方位角
- 使用近但不平行的错误邻井

## 停止条件

- surface lineage 未审计
- 有效邻井覆盖过低
- 前缀校准后仍不能重建可见段
- 负对照同样改善
- fold 0 明显恶化

---

# 11. Typewell 指纹压力测试

该测试不替代主 CV，但所有 F03/F04 晋级方案都必须额外报告。

## 审计对象

识别：

```text
完全重复 Typewell
裁剪版 Typewell
仅 TVT 平移的 Typewell
高相似 GR 指纹 Typewell
```

## 两套结果

```text
primary: spatial_pad_1000_v1
strict: spatial + Typewell fingerprint grouped CV
```

## 解释规则

若 primary 改善、strict 不改善，则结论写为：

> 该特征主要利用了比赛数据中重复或高度相似的 Typewell 模板，对全新 Typewell 的泛化证据不足。

这不必被视为非法，但必须明确收益来源。

---

# 12. 推荐执行顺序

| 顺序 | 实验 | 成本 | 优先级 |
|---:|---|---:|---:|
| 1 | F01：前缀 U 倾角 | 低 | 已进行 |
| 2 | F02a：纯 GR 缺失几何 | 低 | 高 |
| 3 | F02b：实测 GR 信号质量 | 低 | 中高 |
| 4 | F03a：前缀 Typewell 可信度 | 中低 | 很高 |
| 5 | F03b：多尺度 offset 得分面 | 中 | 高 |
| 6 | F04：branch-aware self-template | 中 | 高 |
| 7 | F05a：PF 候选分布几何 | 中 | 中高 |
| 8 | F05b：归一化 ESS/熵 | 中高 | 中低 |
| 9 | F06：horizon-matched prefix holdout | 中高 | 中高 |
| 10 | F07R：邻井相对 ΔU prior | 高 | 最后 |

## 不建议按原样运行

- 原 F05：直接使用 raw ESS、raw entropy、raw likelihood gap
- 原 F07：绝对 surface KNN 或全局 X/Y 局部平面
- 任何 top-1 候选硬选择
- 任何未经过完整 OOF 的输出融合

---

# 13. 条件组合阶段

边际实验完成后，按以下流程构建最终单模特征集。

## C01

```text
B0 + 五折改善最大的单组
```

## C02

```text
C01 + 第二强组
```

要求第二组相对 C01 仍达到统一晋级标准。

## C03

```text
C02 + 第三强组
```

若边际有效但条件无效，说明该组与已有组冗余，不进入最终集合。

## 推荐组合顺序原则

优先保留不同信息源：

1. 前缀结构
2. GR 可观测性
3. Typewell 可信度
4. self-template
5. PF 候选几何
6. prefix holdout
7. 邻井相对结构

不要只按单组分数把高度同源的特征全部加入。

---

# 14. 统一 Artifact 规范

每个实验目录至少包含：

```text
config.json
feature_list.json
feature_definition.json
feature_lineage.json
feature_quality.csv
cache_manifest.json
leakage_tests.json
negative_control_metrics.json
predictions.parquet
per_well.csv
per_fold.csv
slice_metrics.csv
metrics.json
runtime.json
bootstrap_replicates.parquet
conclusion.md
```

## feature_quality.csv

至少包含：

```text
feature
dtype
unit
source
finite_rate
unique_count
mean
std
p01
p50
p99
well_constant_rate
correlation_with_existing_feature
```

## feature_lineage.json

每列记录：

```text
raw source
是否使用 TVT_input
是否使用 hidden GR
是否使用 Typewell
是否使用 PF/Beam
是否需要 outer-train fit
fit wells hash
transform version
unit
```

## leakage_tests.json

至少记录：

```text
hidden TVT deletion invariance
hidden TVT mutation invariance
surface deletion invariance
outer-fold source exclusion
PF cache fold match
row hash match
fixed-seed reproducibility
negative-control status
```

---

# 15. 明确不做

在本路线完成前，不进行：

- LightGBM 参数搜索
- 树数调整
- PF 参数调整
- target 更换
- fold 更换
- CatBoost、TCN、Transformer、TabICL 或其他模型
- stacking
- 输出级融合
- learned model package
- 同井训练 TVT、surface 或 contact
- 用 fold 0 一次失败否定整个信息源
- 用 oracle 结果作为可提交性能
- 用硬编码历史分数替代本次运行结果

---

# 16. 当前主线判断

这条路线的核心不是继续堆表格统计，而是逐步回答五个问题：

\[
\text{GR 是否可观测}
\]

\[
\text{Typewell 对当前井是否可信}
\]

\[
\text{当前井 self-template 是否更可靠}
\]

\[
\text{PF 候选何时开始分叉和多模态}
\]

\[
\text{前缀回测能否预测远端失效}
\]

如果这些测试期合法信号能稳定改善同一个冻结单模 LightGBM，再进入条件组合阶段。

如果不能，则应把下一轮研究重点转向：

- PF 候选生成本身
- GR likelihood 的分段可靠性
- 多尺度观测模型
- 相对结构先验

而不是继续增加同源静态特征。

---

# 17. 最终路线摘要

```text
B0
└── F01 前缀 U
    └── F02a 缺失几何
        └── F02b 实测 GR 质量
            └── F03a Typewell 前缀可信度
                └── F03b 多尺度 offset landscape
                    └── F04 branch-aware self-template
                        └── F05a PF 候选几何
                            └── F05b 归一化内部统计
                                └── F06 horizon-matched prefix holdout
                                    └── F07R 邻井相对 ΔU
```

所有实验先做 S 系列边际筛查；通过后再按 C 系列验证条件增量。

最终保留的不是“特征数量最多”的方案，而是：

> 在严格相同数据合同下，能够跨折、跨井、跨困难切片稳定改善 B0，且负对照不能复制该收益的最小特征集合。
