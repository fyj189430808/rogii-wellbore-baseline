# P3-D01 PF 观测证据与随机路径权重审计设计

## 1. 目标与批准来源

本设计落实已经批准的《三阶段突破路线与实验执行说明》中 `P3-D01`。用户在路线冻结后明确回复“继续”，因此本设计不再重复请求是否开始。

唯一问题是：

> 当前 PF 把插值 GR 当作真实观测，并用整井累计似然和固定温度给 128 条随机路径加权；这种做法是否让权重锐度被井长、GR 缺失和似然尺度系统性带偏？

本实验是只读诊断，不修改 P3B00 的 41 个特征，不训练 LightGBM，不生成正式特征，不打开 116 口影子井的目标或误差。

## 2. 三种实现方案

### 方案 A：只使用 P2-P01 旧缓存近似审计

旧 runtime 已保存 `pf_best_ll_per_row`、`pf_ll_spread` 和 `pf_gr_sigma`。这种方法最快，但没有 128 个原始累计似然，因此无法精确计算温度 3/5/8/12 下的 ESS、最大权重和熵，不能回答 D01 的核心问题。

### 方案 B：重跑并保存全部 128 条逐行路径

这种方法信息最完整，但 657 口井会产生约 4 亿个路径点，缓存体积和 I/O 都没有必要。P2-P01 已经保存四条温度路径，D01 不需要再次永久保存全部路径。

### 方案 C：精确重放 PF，只保存紧凑似然缓存

复用冻结的 P2-P01 PF 内核和参数，每井仍运行 500 粒子、128 个 seed，但只永久保存 128 个累计对数似然以及合法 GR 统计。逐行 seed 路径只在内存中短暂存在，不写盘。四条温度路径的误差从既有 P2-P01 合法路径缓存和开发井 OOF 真值重新计算。

采用方案 C。它能精确回答 D01，同时保持缓存小、可恢复，并且不改变旧 PF 代码。

## 3. 数据边界

### 合法生成阶段

每口开发井只读取：

```text
horizontal_well: MD, Z, GR, TVT_input
typewell:        TVT, GR
```

使用固定来源：

- fold：`balanced_well_5fold_v1`
- PF 参数：`P2_P01_multiseed_pf_mean_v1`
- PF 内核：`src/p2_p01_multiseed_pf.py`
- 影子井名单：`P3_shadow_holdout_v1`

在合法缓存全部生成前，不读取 `TVT`、P2-P02 误差、逐井 RMSE 或最佳温度。

### 诊断评分阶段

合法缓存完成后，只从 P2-P02 OOF Parquet 中筛选 657 口开发井，再读取：

```text
well_id, fold, row_index, target_tvt, pred_tvt
```

四条固定温度路径来自既有 P2-P01 `legal_cache`。影子井必须在 Arrow 扫描阶段排除，返回表与影子井交集必须为 0。

## 4. 每井合法统计

### GR 证据数量

- `hidden_rows`：自然隐藏行数。
- `observed_gr_rows`：隐藏段原始 GR 非缺失行数。
- `interpolated_gr_rows`：当前 PF 会通过插值或回填得到 GR 的行数。
- `observed_gr_fraction`：`observed_gr_rows / hidden_rows`。
- `longest_gr_gap_rows`：隐藏段最长连续 GR 缺失行数。
- `longest_gr_gap_md_ft`：最长连续缺失段覆盖的 MD 范围。
- `current_likelihood_update_count`：当前实现每一隐藏行都更新权重，因此等于 `hidden_rows`。
- `effective_observation_count`：保守地定义为真实观测行数，即 `observed_gr_rows`；插值不创造新的独立测量。

### 可见前缀 GR 标定

- `gr_sigma`：当前 PF 的值，即可见 GR 缺失填 0 后计算残差标准差，再裁剪到 10～60 API。
- `observed_only_gr_sigma`：只用可见段真实 GR 计算并同样裁剪的对照。
- `affine_gr_slope`、`affine_gr_intercept`：在可见真实 GR 上拟合

```text
horizontal_GR = slope × typewell_GR(TVT_input) + intercept
```

这两个量只用于诊断，不改变本次 PF 输入。

### seed 累计似然

- `seed_ll_mean`
- `seed_ll_std`
- `seed_ll_range`
- `ll_per_observed_row = seed_ll_mean / observed_gr_rows`

每口井另外保存长度为 128 的 `seed_log_likelihoods` 紧凑缓存，便于复算和后续 PF02 使用。该缓存不得混入正式特征目录。

## 5. 温度权重公式

对温度 `T ∈ {3, 5, 8, 12}`：

```text
weight_s = exp((LL_s - max(LL)) / T)
weight_s = weight_s / sum(weight_s)
```

保存：

```text
scale_T_ess        = 1 / sum(weight_s²)
scale_T_max_weight = max(weight_s)
scale_T_entropy    = -sum(weight_s × log(weight_s))
scale_T_normalized_entropy = entropy / log(128)
```

ESS 必须位于 `[1, 128]`；均匀似然应得到 ESS 128、最大权重 1/128、归一化熵 1。

## 6. 真值只读分析

每口开发井重新计算：

- P2-P02 OOF RMSE；
- P2-P01 无权均值路径 RMSE；
- 温度 3/5/8/12 路径 RMSE；
- 事后最佳温度及其 oracle RMSE。

“最佳温度”明确标为 oracle，只衡量温度自适应的表示空间，不进入正式特征。

分析包含：

1. ESS 与隐藏行数、真实 GR 比例、最长缺口的 Spearman 相关；
2. ESS 与 P2-P02、PF 无权均值路径误差的相关；
3. 以上相关在五个 fold 的方向和幅度；
4. 按隐藏长度、GR 有效率、最长缺口和 P2-P02 难度四分位分箱；
5. 困难井是否更常出现 ESS 小于 12.8 的权重塌缩；
6. 容易井中加权路径是否比无权均值路径恶化超过 0.25 ft；
7. 事后最佳温度相对最佳全局固定温度的 pooled oracle 余量。

相关的零假设对照使用固定随机种子在井间打乱合法统计 1,000 次，报告观测相关在置换分布中的双侧比例。置换只用于诊断显著性，不选择每井温度。

## 7. 预登记解释规则

### 支持优先做 PF01

满足任一条件即认为缺失感知观测值得正式测试：

- 最长缺口或插值比例与 PF 无权均值路径 RMSE 的整体 Spearman 绝对值至少 0.20，且至少 4/5 折方向一致；
- 高缺失四分位的 PF 无权均值路径 RMSE 比低缺失四分位至少差 0.50 ft，且至少 4/5 折同方向；
- 当前 `gr_sigma` 与 observed-only 对照的中位相对差至少 10%。

### 支持随后做 PF02

必须同时满足：

1. 某个固定温度的 ESS 与井长或 GR 有效率整体 Spearman 绝对值至少 0.25，至少 4/5 折方向一致，置换比例不高于 0.05；
2. 事后最佳温度相对最佳全局固定温度的 pooled 路径 RMSE 至少改善 0.25 ft。

### 不能由 D01 否定的方向

若这些规则不通过，只能说明当前合法统计没有稳定解释固定权重问题，不能否定 PF、GR、多峰路径或 D00 已支持的严格 OOF 低维残差路线。

## 8. 运行结构与恢复

支持三种模式：

```text
smoke   第一口开发井
fold01  开发集 folds 0～1
all     全部 657 口开发井
```

每井保存一个独立 likelihood cache 和 runtime JSON。缓存指纹覆盖配置、fold、影子名单、PF 核心、D01 核心和 runner。中断后只重算缺失或指纹不匹配的井。

旧 P2-P01 记录显示全 773 口累计约 8.28 核心小时；8 个 worker 的 657 口开发集预计约 45～70 分钟。

## 9. 产物

```text
rogii_clean/artifacts/P3_D01_pf_observation_weight_audit_v1/
├── config.json
├── legal_likelihood_cache/<well_id>.npz
├── legal_runtime/<well_id>.json
├── per_well.csv
├── per_fold_correlations.csv
├── overall_correlations.csv
├── binned_metrics.csv
├── scale_path_metrics.csv
├── summary.json
├── runtime.json
└── conclusion.md
```

不生成 `feature_list.json`、模型文件或正式预测。

## 10. 测试

1. 先写失败测试，再实现每个新函数；
2. 软最大值、ESS、熵使用手算小例子；
3. GR 缺失段使用小型 DataFrame 验证行数和最长缺口；
4. 隐藏 TVT 改写后合法统计和似然必须逐位不变；
5. shadow 井不能进入 generation task、目标表或输出；
6. 缓存指纹不匹配必须重算；
7. smoke 重放的四条温度路径与 P2-P01 冻结缓存最大差必须为 0；
8. 完整运行前后都运行 D01 聚焦测试，并在交付前运行三阶段相关回归测试。

## 11. 自审结果

- 没有 TBD、TODO 或未定义字段；
- `effective_observation_count` 已明确为真实 GR 行数，不与当前 PF 更新次数混淆；
- oracle 最佳温度与正式输入物理分离；
- 影子集只提供井号名单，不读取目标或误差；
- 本实验不修改模型、41 列特征、fold 或评价区；
- 方案范围只覆盖 D01，没有提前实现 PF01/PF02。
