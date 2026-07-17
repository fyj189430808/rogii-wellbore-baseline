# Task 2 实施报告：登记 RF01 稳定性审计

## 结论

状态：`DONE_WITH_CONCERNS`

Task 2 要求的实验卡和机器可读配置已经创建并通过字段、双文件语义、源 RF01a、冻结模型与实际 fold 文件哈希校验。本子任务没有训练模型、没有修改 runner、没有执行 Git，也没有创建 `artifacts/RF01_stability_audit_v1`。

## 本子任务写入的文件

1. `H:\kaggle\716\rogii_clean\experiments\RF01_stability_audit_v1_card.md`
2. `H:\kaggle\716\rogii_clean\configs\rf01_stability_audit_v1.json`
3. `H:\kaggle\716\.superpowers\sdd\task-2-report.md`（本报告）

两个正式登记文件在实施前均不存在，因此没有覆盖既有实验卡或配置。

## 实验卡内容核对

实验卡已经明确锁死以下语义：

- 实验编号为 `RF01_stability_audit_v1`。
- 实验类型为独立诊断，不参与原 RF01 晋级。
- 固定对象为 `RF01a_huber_slopes_v1`。
- 复用 folds 为 `[0, 1]`，只新运行 folds `[2, 3, 4]`。
- 原 RF01 八个版本继续“不晋级”，不得事后改判或追认晋级。
- 不选择其他 RF01 版本，不调整窗口、17 个固定特征、LightGBM 参数、1,734 棵树、seed、fold、目标、评价行、评分代码或后处理。
- 审计问题固定为：fold 1 是唯一异常折，还是 fold 0 才是异常折？
- 四条预注册选择理由与任务 brief 逐字一致。
- 数据血缘、固定模型、17 个特征、目标、fold hash、B00 对照、覆盖门槛、合同异常停止条件和结果解释边界均已登记。

## JSON 必需字段验证

`rf01_stability_audit_v1.json` 可由 Python `json.loads` 解析。以下字段与 brief 的固定值逐项相等：

- `experiment_id = RF01_stability_audit_v1`
- `experiment_type = diagnostic_only`
- `source_experiment_id = RF01a_huber_slopes_v1`
- `source_config = configs/rf01a_huber_slopes_v1.json`
- `source_artifact_dir = artifacts/RF01a_huber_slopes_v1`
- `source_feature_cache = artifacts/RF01a_huber_slopes_v1/feature_cache.parquet`
- `model_config = configs/lgbm_feature_baseline_v1.json`
- `fold_registry = artifacts/folds/spatial_pad_1000_v1.csv`
- `fold_registry_sha256 = 0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab`
- `b00_prediction_file = artifacts/B00_simple_lgbm_v1/predictions.parquet`
- `expected_wells = 773`
- `expected_rows = 3783989`
- `reused_folds = [0, 1]`
- `new_folds = [2, 3, 4]`
- `original_rf01_decision = not_promoted_and_unchanged`
- `allow_reselection = false`
- `allow_parameter_change = false`
- `pre_registered_selection_reasons` 恰好包含预注册的四条理由，顺序一致。

## 与 RF01a 和冻结合同的一致性验证

读取 `configs/rf01a_huber_slopes_v1.json` 与 `configs/lgbm_feature_baseline_v1.json` 后，断言结果如下：

- RF01a `experiment_id` 与审计 `source_experiment_id` 一致。
- RF01a `feature_version` 为固定值 `rf01_huber_slopes_17_v1`。
- RF01a `feature_columns` 恰为 17 列，实验卡逐列登记一致。
- RF01a `model_config` 与审计配置一致，均为 `configs/lgbm_feature_baseline_v1.json`。
- RF01a 的 fold 注册表路径、fold SHA-256、预期井数和预期行数均与审计配置一致。
- 冻结模型配置版本为 `lgbm_feature_baseline_v1`，`n_estimators=1734`、`random_state=29`、`early_stopping=false`。
- 对实际 `artifacts/folds/spatial_pad_1000_v1.csv` 重新计算 SHA-256，结果为 `0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab`，与源配置和审计配置都一致。
- 源 RF01a 特征缓存、原 fold 0/1 预测、B00 预测和固定 fold 注册表均存在；这些检查只读取文件，没有运行模型。

## 双文件语义验证

自动断言实验卡与 JSON 同时包含并一致表达：

- 实验编号 `RF01_stability_audit_v1`；
- 固定对象 `RF01a_huber_slopes_v1`；
- 复用 folds `[0, 1]`；
- 新 folds `[2, 3, 4]`；
- 独立诊断、不参与原 RF01 晋级；
- 不得事后改判或追认晋级；
- 四条预注册选择理由。

最终验证输出为：

```text
PASS json_parse_and_required_fields
PASS card_config_semantics
PASS source_feature_version=rf01_huber_slopes_17_v1
PASS source_model_and_fold_contract
PASS actual_fold_sha256=0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab
PASS no_audit_artifact_created
```

第一次内联验证曾因 Windows PowerShell 把通过标准输入传给 Python 的中文测试字面量替换成 `?` 而失败；对文件内实际 Unicode 码点检查后确认产物无损。验证脚本改用 ASCII `\u` 转义重跑，同一组断言全部通过，没有为绕过测试而修改产物内容。

## 副作用检查

- `H:\kaggle\716\rogii_clean\artifacts\RF01_stability_audit_v1` 不存在。
- 未调用任何训练脚本或模型入口。
- 未修改 runner。
- 未执行任何 Git、worktree、分支、提交、推送或 PR 操作。
- 未修改 registry；Task 2 只做运行前登记，诊断结果应在后续实际审计完成后登记。

## 顾虑与后续恢复条件

唯一顾虑：源 `rf01a_huber_slopes_v1.json` 没有单独保存评价 row hash，Task 2 brief 的 JSON 必需字段也没有提供该值，因此本次没有猜测或制造一个 row hash。实验卡已明确登记这一缺口。后续 Task 3/4 在任何训练前必须从固定 `well_id, fold, row_index` 行键计算一致的评价行指纹，并与 B00/RF01a 对齐验证；若行数、行键、目标或 row hash 不一致，应作为合同异常停止整条审计，而不能记成实验负结果。
