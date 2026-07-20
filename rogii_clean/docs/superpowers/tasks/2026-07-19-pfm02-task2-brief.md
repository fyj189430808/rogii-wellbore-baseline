# P3-PFM02 Task 2：44 列冻结 LightGBM runner

## 任务位置

Task 1 会生成 657 口无标签三模式合法缓存。本任务把三列 delta 追加到 P3B00 原 41 列，用现有 P3-PF02 正式训练骨架运行单模 LightGBM。

## 只允许新增

- `rogii_clean/configs/p3_pfm02_direct_mode_paths_v1.json`
- `rogii_clean/scripts/run_p3_pfm02_direct_mode_paths_cv.py`
- `rogii_clean/tests/test_run_p3_pfm02_direct_mode_paths_cv.py`
- 报告：`rogii_clean/docs/superpowers/tasks/2026-07-19-pfm02-task2-report.md`

不要修改旧 runner、训练器、PFM01/PF03/P3B00。不要运行正式训练。

## 实现骨架

以 `scripts/run_p3_pf02_target_ess_lgbm_cv.py` 为直接骨架，尽量复用其：开发/影子反连接、Arrow 过滤、B00+F05a 加载、旧 P01 五路径合并、自然键/anchor 合并、冻结模型校验、`train_fold`、配对 P3B00 评分、folds01/full5 门槛、bootstrap、产物与自动停止。

## 冻结正式特征

`P3B00_FEATURES` 必须与 `artifacts/P2_P02_multiscale_pf_paths_v1/feature_list.json` 的 41 列逐位一致，末尾只追加：

```python
NEW_MODE_FEATURES = [
    "pf_mode_low_delta",
    "pf_mode_middle_delta",
    "pf_mode_high_delta",
]
```

共 44 列。禁止 direction、mass、seed_count、p2_position、separation、oracle、target、surface、geology 等额外特征。

## 冻结模型与数据

- `configs/lgbm_feature_baseline_v1.json` 全参数逐项相同：单 `LGBMRegressor`、1734 树、seed29、无 early stopping、行等权。
- fold `balanced_well_5fold_v1`；排除 shadow 后 657 井、3,211,872 行。
- fold井数 131/132/131/132/131；行数 651881/630395/645557/649717/634322。
- 基线 `P3B00_group5_p2p02_v1`，开发井配对 OOF micro `10.272146267501086`。
- baseline OOF 路径：`artifacts/P2_P02_multiscale_pf_paths_v1/predictions.parquet`，SHA `8109514eb125f8beda2559f39e1d9f54396d41fd38fa76114423c8dd6bca4d37`。
- 原 41 列来源、fold、shadow、model config 哈希全部照抄现有 PF02 target 配置和 runner 的冻结合同。
- 三模式来源：`artifacts/P3_PFM02_mode_paths_v1/legal_cache` 和 `legal_runtime`；schema 为 Task 1 的八列合同；只接受统一非空 fingerprint 和 `hidden_tvt_read=false`。

## 配置与门槛

实验编号 `P3_PFM02_direct_mode_paths_v1`，默认输出同名 artifact。CLI 只允许 `--folds 0,1` 或 `--folds all`。

沿用现 P3-PF02 成功条件：fold01 pooled 改善≥0.20、最差折≥-0.10；full5 改善≥0.10、改善折≥4、后3折改善≥2、最差折≥-0.25、bootstrap CI high<0、胜井率≥0.55、P90恶化≤0.20、top5%收益占比≤0.60。`all` 也必须先重算 folds01，未晋级就不训练2–4。

## 输出

标准 config、feature_list(44列)、parameter_list、leakage_audit、逐折 model/预测/runtime/importance、folds01/full5 predictions/metrics/per_well/per_fold、runtime、conclusion。结论必须明确只能否定当前三路径直接追加实现。

## TDD 必测

1. 先写测试并看到因新 runner 缺失而 RED；记录命令和错误。
2. `build_formal_feature_names()` 精确 41+3=44、顺序固定且无重复。
3. config/model任何参数、树数、seed、training policy、fold/井/行数变化都失败。
4. 三模式 cache 缺井、错 schema、错 fold、错行数、重复键、非统一 fingerprint、runtime 缺失/读过真值、shadow 重叠均失败。
5. 合并按 `(well_id,row_index)` 一对一，保持原行序，float32 anchor 不等立即失败，三列必须有限。
6. 不能读取 PFM01 `oracle/`，不能让 forbidden 摘要进入 44 列。
7. folds01 自动停止和 `all` 晋级行为复用原逻辑并有测试。
8. 旧 PF02 target runner 的相关测试仍通过。

完成后运行专项 pytest、相关旧测试和 py_compile。不要 Git commit。报告 RED/GREEN、文件、自审和担忧；消息只简短返回状态。
