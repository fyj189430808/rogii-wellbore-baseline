# Task 2 独立复核：登记 RF01 稳定性审计

复核日期：2026-07-17

## 结论

- 规格符合性：**通过**。
- 文档质量：**批准**。
- Task 3 前置状态：登记材料可以作为实现输入；但必须先执行卡片中已经预注册的评价行键/row hash 对齐检查。该检查是对既有缺口的正确处理，不是本次复核新增的实验条件。

## 复核范围与方法

本次完整阅读了 `.superpowers/sdd/task-2-review-package.md`，并逐项核对以下依据和产物：

- 需求：`.superpowers/sdd/task-2-brief.md`。
- 项目规则：`AGENTS.md` 第 9 章。
- 实施计划：`docs/superpowers/plans/2026-07-16-agents-and-rf01-stability-audit.md` 的 Task 2。
- 实验卡：`rogii_clean/experiments/RF01_stability_audit_v1_card.md`。
- 机器配置：`rogii_clean/configs/rf01_stability_audit_v1.json`。
- 实施报告：`.superpowers/sdd/task-2-report.md`。
- 真实来源：RF01a 源配置、冻结 LightGBM 配置、RF01a 特征缓存及 folds 0–1 产物、B00 OOF 预测、固定 fold 注册表。

复核只做文件读取、JSON/Parquet 元数据检查和 SHA-256 复算；未运行训练、未修改 runner、未使用 Git。除本复核文件外，没有修改项目文件。

## 规格符合性核对

### 1. 审计身份和研究边界

实验卡与 JSON 一致锁定：

- `experiment_id = RF01_stability_audit_v1`；
- `experiment_type = diagnostic_only`；
- 固定对象为 `RF01a_huber_slopes_v1`；
- 复用 folds `[0, 1]`，只新训练 folds `[2, 3, 4]`；
- 原 RF01 八个版本继续“不晋级”，审计结果不得用于事后改判或追认晋级；
- 不重选其余 RF01 版本，不调整窗口、特征、LightGBM 参数、1,734 棵树、seed、fold、目标、评价行、评分代码或后处理，也不重跑或覆盖 folds 0–1；
- 审计问题限定为 fold 1 是否为唯一异常折、fold 0 是否才是异常折；若其余折方向混合，也只能报告不支持这两种单一异常折解释。

这些内容符合 brief、计划 Task 2 和 `AGENTS.md` 第 9 章，没有发现事后选择、调参、重新打开晋级判断或重跑 folds 0–1 的暗示。

### 2. 四条预注册选择理由

实验卡和 JSON 的 `pre_registered_selection_reasons` 均保存了同一组、同一顺序的四条理由：

1. 公式最简单；
2. fold 1 损失最小；
3. 不含派生程度更高的差值、曲率和无界外推；
4. 选择依据不使用 folds 2–4。

既有 `RF01_u_slope_route_summary_v1/metrics.csv` 也支持第二条理由：RF01a 在八个 RF01 候选中 fold 1 相对 B00 的损失最小，为 0.1295 ft。没有使用 folds 2–4 结果进行选择的迹象。

### 3. JSON 必需字段和路径

`rf01_stability_audit_v1.json` 可正常解析。brief 要求的全部固定字段和 `pre_registered_selection_reasons` 均存在且取值正确。下列相对路径均以 `rogii_clean` 为基准并实际存在：

- `configs/rf01a_huber_slopes_v1.json`；
- `artifacts/RF01a_huber_slopes_v1`；
- `artifacts/RF01a_huber_slopes_v1/feature_cache.parquet`；
- `configs/lgbm_feature_baseline_v1.json`；
- `artifacts/folds/spatial_pad_1000_v1.csv`；
- `artifacts/B00_simple_lgbm_v1/predictions.parquet`。

`source_artifact_dir`、`reused_folds` 和 `new_folds` 足以让 Task 3 按固定约定解析原 folds 0–1 与新 folds 2–4；配置没有要求审计脚本重新生成或选择特征。

### 4. 与真实 RF01a、模型和 fold 合同的一致性

独立核验结果如下：

| 核验项 | 真实值 | 结论 |
|---|---|---|
| RF01a `experiment_id` | `RF01a_huber_slopes_v1` | 一致 |
| RF01a `feature_version` | `rf01_huber_slopes_17_v1` | 一致 |
| RF01a 固定模型特征 | 12 个 B00 特征 + 5 个 Huber slope，共 17 个 | 与实验卡逐列一致 |
| 模型配置 | `configs/lgbm_feature_baseline_v1.json` | 一致 |
| 树数与 seed | `n_estimators=1734`，各固定 seed 为 29 | 一致 |
| early stopping | `false` | 一致 |
| fold 注册表实际 SHA-256 | `0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab` | 与源配置、审计配置一致 |
| RF01a 特征缓存 | 3,783,989 行，773 井，含全部 17 个固定模型特征 | 一致 |
| RF01a fold 0 预测 | 757,050 行，147 井，全部标为 fold 0 | 与 runtime/B00 一致 |
| RF01a fold 1 预测 | 756,990 行，155 井，全部标为 fold 1 | 与 runtime/B00 一致 |
| B00 OOF | 3,783,989 行，773 井；五折行数与 RF01a 缓存一致 | 一致 |

实际 fold 行数依次为 757,050、756,990、756,649、757,061、756,239，总计 3,783,989。没有发现路径、井数、行数、特征版本、模型配置或 fold hash 被错误登记。

### 5. row hash 顾虑处理

RF01a 源配置和 B00 配置都没有单独保存评价 row hash。实验卡没有把特征缓存 fingerprint、文件 hash 或临时计算值冒充固定评价 row hash，而是明确登记“本次不生成替代值”，并要求 Task 3/4 在任何训练前：

1. 从固定 `well_id, fold, row_index` 行键计算确定性指纹；
2. 对齐 RF01a 特征缓存、既有 folds 0–1 与 B00；
3. 同时验证行数、行键唯一性和 `target_tvt` 一致；
4. 任一不一致均作为合同异常停止，不得记成实验负结果。

这种限定符合 review package 的要求。它既没有擅自制造“已冻结”的 hash，也没有把检查推迟到训练之后。

### 6. Task 2 副作用边界

当前不存在 `rogii_clean/artifacts/RF01_stability_audit_v1`。实验卡明确说明 Task 2 只登记实验卡和配置，不训练、不创建 artifact；报告也没有暗示已经运行模型或修改 registry。现有状态与 Task 2 的限定一致。

## 文档质量评价

实验卡中文清楚，先说明独立诊断身份，再分别登记唯一问题、四条理由、禁止事项、冻结模型/特征/参数、数据血缘、CV/评价行、对照、覆盖、停止边界和结果解释。它覆盖了审计前必须知道的输入、输出和失败语义。

JSON 字段命名明确，所有路径和 fold 集均可直接供 Task 3 使用；源配置负责提供固定 17 列和 target，冻结模型配置负责提供完整参数，避免在审计配置中静默复制出第二套可漂移定义。实施报告把已验证事实、未创建的产物和唯一遗留顾虑分开陈述，未把 folds 2–4 的未知结果写成事实。

因此文档质量批准。

## 问题清单

### 严重

无。

### 重要

无。

### 轻微

无。

## 后续执行约束

Task 3 可以按登记配置开始实现，但实际训练 folds 2–4 之前必须先通过上述 row-key/target 对齐与确定性 row hash 检查。若该检查失败，应按实验卡既定规则停止整条审计；不得通过改行集、补造 canonical hash、重跑 folds 0–1 或换用其他 RF01 版本恢复。
