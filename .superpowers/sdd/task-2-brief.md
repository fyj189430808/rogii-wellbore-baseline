# Task 2：登记 RF01 稳定性审计

## 任务位置

只创建实验卡和机器可读配置，不训练模型，不修改 runner，不使用 Git。

## 必读文件

1. `H:\kaggle\716\docs\superpowers\plans\2026-07-16-agents-and-rf01-stability-audit.md` 中 Task 2
2. `H:\kaggle\716\AGENTS.md` 第9章
3. `H:\kaggle\716\rogii_clean\configs\rf01a_huber_slopes_v1.json`
4. `H:\kaggle\716\rogii_clean\configs\lgbm_feature_baseline_v1.json`

## 创建文件

- `H:\kaggle\716\rogii_clean\experiments\RF01_stability_audit_v1_card.md`
- `H:\kaggle\716\rogii_clean\configs\rf01_stability_audit_v1.json`

## 固定语义

- 实验类型：独立诊断，不参与原 RF01 晋级。
- 原 RF01 八版本继续“不晋级”，不得事后改判。
- 固定对象：`RF01a_huber_slopes_v1`。
- 复用 folds：0、1。
- 只新训练 folds：2、3、4。
- 不选择其他 RF01 版本，不调整窗口、特征、LightGBM参数、fold、目标、评价行或评分。
- 审计只回答：fold1是唯一异常折，还是fold0才是异常折。

## 预先锁死的四个选择理由

1. 公式最简单。
2. fold1损失最小。
3. 不含派生程度更高的差值、曲率和无界外推。
4. 选择依据不使用 folds 2–4。

## JSON 必需字段

```json
{
  "experiment_id": "RF01_stability_audit_v1",
  "experiment_type": "diagnostic_only",
  "source_experiment_id": "RF01a_huber_slopes_v1",
  "source_config": "configs/rf01a_huber_slopes_v1.json",
  "source_artifact_dir": "artifacts/RF01a_huber_slopes_v1",
  "source_feature_cache": "artifacts/RF01a_huber_slopes_v1/feature_cache.parquet",
  "model_config": "configs/lgbm_feature_baseline_v1.json",
  "fold_registry": "artifacts/folds/spatial_pad_1000_v1.csv",
  "fold_registry_sha256": "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab",
  "b00_prediction_file": "artifacts/B00_simple_lgbm_v1/predictions.parquet",
  "expected_wells": 773,
  "expected_rows": 3783989,
  "reused_folds": [0, 1],
  "new_folds": [2, 3, 4],
  "original_rf01_decision": "not_promoted_and_unchanged",
  "allow_reselection": false,
  "allow_parameter_change": false
}
```

配置还要保存四条选择理由，字段名使用 `pre_registered_selection_reasons`。

## 验证

- JSON 可解析。
- 两个文件中的实验编号、固定对象、复用折和新折一致。
- source RF01a 的 `feature_version` 必须是 `rf01_huber_slopes_17_v1`。
- source RF01a 的模型配置和 fold hash 必须与审计配置一致。
- 不创建 artifact，不运行模型。

## 报告

写入：`H:\kaggle\716\.superpowers\sdd\task-2-report.md`

包含创建文件、字段验证、与RF01a一致性验证、顾虑。最终回复 DONE 或 DONE_WITH_CONCERNS 和一句验证摘要。
