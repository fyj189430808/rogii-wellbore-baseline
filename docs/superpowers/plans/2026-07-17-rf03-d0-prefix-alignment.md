# RF03-D0 Prefix Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不读取隐藏 TVT、不训练模型的前提下，完成 773 口井可见前缀 Typewell 的 0 ft 对 `±10/20 ft` 错位正对照，并生成可复核的正式诊断产物。

**Architecture:** 保留 `src/rf03_prefix_alignment.py` 作为纯计算模块；把 `scripts/diagnose_rf03_prefix_alignment.py` 改成只读取冻结 JSON 的薄运行入口。运行器先验证数据与 fold 合同，再按井生成 offset 分数，最后由独立汇总函数生成井级、fold 级、支持量和 bootstrap 产物。

**Tech Stack:** Python 3、pandas、NumPy、pytest、Parquet；不训练 LightGBM，不使用 Git。

## Global Constraints

- 用户已确认 `B00_simple_lgbm_v1` 就是 `B0`；本实验不建立其他基线。
- 只读取水平井 `MD/GR/TVT_input` 和 Typewell `TVT/GR`。
- offset 固定为 `(-20,-10,0,10,20)` ft。
- scope 固定为 `all_visible` 与 `tail_1000ft`。
- 共同有效点最低为 50。
- `spatial_pad_1000_v1` 与其 SHA-256 固定不变。
- 不允许 CLI 改 offset、scope、最低点数或通过门槛。
- 诊断产物必须声明不是 TVT 模型预测。
- 不修改 B0、LightGBM runner、模型参数、目标或历史实验。

---

### Task 1: 冻结 RF03-D0 配置并补全运行合同

**Files:**
- Create: `rogii_clean/configs/rf03_d0_prefix_alignment_v1.json`
- Modify: `rogii_clean/experiments/RF03_D0_prefix_alignment_card.md`
- Modify: `rogii_clean/scripts/diagnose_rf03_prefix_alignment.py`
- Test: `rogii_clean/tests/test_rf03_prefix_alignment.py`

**Interfaces:**
- Consumes: `load_and_validate_registry(path)`、`score_prefix_offsets(...)`、`build_prefix_alignment_margins(...)`。
- Produces: `load_and_validate_diagnostic_config(path) -> dict`、`run_alignment_for_registry(registry, train_dir, minimum_points) -> tuple[pd.DataFrame, pd.DataFrame]`、`build_summary(offset_scores, margins) -> dict`。

- [ ] **Step 1: 写配置冻结失败测试**

测试必须断言：

```python
config = load_and_validate_diagnostic_config(config_path)
assert config["experiment_id"] == "RF03_D0_prefix_alignment_v1"
assert config["baseline_id"] == "B00_simple_lgbm_v1"
assert config["offsets_ft"] == [-20.0, -10.0, 0.0, 10.0, 20.0]
assert config["minimum_points"] == 50
assert config["hidden_tvt_loaded"] is False
```

并验证修改 offset、scope、最低点数、fold hash 或 baseline_id 时立即报错。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```powershell
& 'D:\anaconda\python.exe' -m pytest tests\test_rf03_prefix_alignment.py -q
```

Expected: 因配置读取/校验接口不存在而失败。

- [ ] **Step 3: 创建冻结配置**

配置固定包含：

```json
{
  "experiment_id": "RF03_D0_prefix_alignment_v1",
  "experiment_type": "diagnostic_only",
  "baseline_id": "B00_simple_lgbm_v1",
  "fold_registry": "artifacts/folds/spatial_pad_1000_v1.csv",
  "fold_registry_sha256": "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab",
  "expected_wells": 773,
  "offsets_ft": [-20.0, -10.0, 0.0, 10.0, 20.0],
  "scopes": {"all_visible": null, "tail_1000ft": 1000.0},
  "minimum_points": 50,
  "bootstrap_resamples": 2000,
  "bootstrap_seed": 42,
  "hidden_tvt_loaded": false
}
```

- [ ] **Step 4: 最小实现配置校验与纯运行接口**

CLI 只接受：

```text
--config
--mode smoke|full
```

`smoke` 固定取注册表按 `well_id` 排序后的前三口井并写到独立 `_smoke` 目录；`full` 固定处理全部 773 井。不得接受 offset、scope 或 minimum-points 覆盖参数。

- [ ] **Step 5: 更新实验卡**

明确写入：

```text
baseline_id = B00_simple_lgbm_v1 = B0
诊断不训练模型
诊断不读取隐藏 TVT
通过只允许进入后续负对照/F03a，不代表 CV 已改善
```

- [ ] **Step 6: 运行聚焦测试确认 GREEN**

Run:

```powershell
& 'D:\anaconda\python.exe' -m pytest tests\test_rf03_prefix_alignment.py -q
```

Expected: 全部通过。

---

### Task 2: 生成诊断专用的完整产物

**Files:**
- Modify: `rogii_clean/scripts/diagnose_rf03_prefix_alignment.py`
- Test: `rogii_clean/tests/test_rf03_prefix_alignment.py`

**Interfaces:**
- Consumes: Task 1 的 `offset_scores`、`margins` 和冻结配置。
- Produces: `build_diagnostic_artifacts(...) -> dict[str, Path]`、`bootstrap_alignment_margins(margins, n_resamples=2000, seed=42) -> pd.DataFrame`。

- [ ] **Step 1: 写产物合同失败测试**

测试固定检查：

```python
required = {
    "config.json",
    "feature_list.json",
    "feature_definition.json",
    "feature_lineage.json",
    "feature_quality.csv",
    "cache_manifest.json",
    "leakage_tests.json",
    "negative_control_metrics.json",
    "predictions.parquet",
    "offset_scores.csv",
    "per_well.csv",
    "per_well_margins.csv",
    "per_fold.csv",
    "slice_metrics.csv",
    "metrics.json",
    "summary.json",
    "runtime.json",
    "bootstrap_replicates.parquet",
    "conclusion.md",
}
assert required.issubset({path.name for path in output_dir.iterdir()})
```

`predictions.parquet` 必须包含 `artifact_role="diagnostic_scores_not_tvt_predictions"` 对应的字段说明，且其数据逐位等于 `offset_scores.csv` 的诊断分数，不得出现 `pred_tvt`。

- [ ] **Step 2: 写 bootstrap 失败测试**

对一个四井合成 margins 表，固定 seed 42，验证：

```python
replicates = bootstrap_alignment_margins(margins, 2000, 42)
assert len(replicates) == 4000  # 两个 scope × 2000 次
assert set(replicates["scope"]) == {"all_visible", "tail_1000ft"}
assert set(replicates["seed"]) == {42}
```

每次重采样必须以井为单位，同时保存 NCC/MAE 的零 offset 胜率和 median margin；bootstrap 只作稳定性诊断，不改变固定通过门槛。

- [ ] **Step 3: 运行测试确认 RED**

Run:

```powershell
& 'D:\anaconda\python.exe' -m pytest tests\test_rf03_prefix_alignment.py -q
```

Expected: 缺少 artifact/bootstrap builder 而失败。

- [ ] **Step 4: 实现诊断产物 builder**

固定语义：

- `per_well.csv` 与 `per_well_margins.csv`：一井 × scope 一行；
- `per_fold.csv`：总体与五折 × 两个 scope；
- `slice_metrics.csv`：支持量分箱与两个 scope；
- `negative_control_metrics.json`：四个错位 offset 的实际得分，不把尚未运行的循环平移写成 passed；
- `leakage_tests.json`：记录 hidden TVT 未加载、固定列白名单、公共支持量、fold hash、代码/input hash；
- `cache_manifest.json`：记录 773 对 horizontal/typewell 输入文件 hash、注册表 hash、脚本和核心模块 hash；
- `feature_lineage.json`：记录每个诊断量只来自可见前缀或 Typewell；
- `runtime.json`：井数、跳过井数、有效井数、耗时和代码指纹。

- [ ] **Step 5: 实现原子落盘**

先写到同级临时目录，全部 schema/hash 验证通过后再一次性移动到正式目录；失败时不得留下看似完整的正式结果。

- [ ] **Step 6: 运行聚焦与全套测试**

Run:

```powershell
& 'D:\anaconda\python.exe' -m pytest tests\test_rf03_prefix_alignment.py -q
$tempPath = 'H:\kaggle\716\.pytest_tmp_rf03_d0_20260717_1'
& 'D:\anaconda\python.exe' -m pytest tests --basetemp $tempPath -q
```

Expected: 聚焦测试和全套测试全部通过。

---

### Task 3: 小样本与完整 773 井运行

**Files:**
- Create: `rogii_clean/artifacts/_smoke/RF03_D0_prefix_alignment_v1/`
- Create: `rogii_clean/artifacts/RF03_D0_prefix_alignment_v1/`
- Modify: `rogii_clean/experiments/registry.jsonl`

**Interfaces:**
- Consumes: Task 2 的冻结 runner。
- Produces: RF03-D0 完整诊断、固定结论和一条 registry 记录。

- [ ] **Step 1: 运行三井 smoke**

Run:

```powershell
& 'D:\anaconda\envs\fyj\python.exe' scripts\diagnose_rf03_prefix_alignment.py --config configs\rf03_d0_prefix_alignment_v1.json --mode smoke
```

Expected: 只处理三个固定井；不读取 `TVT`；生成独立 smoke 产物。

- [ ] **Step 2: 复核 smoke**

检查所有 offset 在同一井/scope 下 `n_points` 完全相同，`predictions.parquet` 不含 `target_tvt`、`pred_tvt` 或隐藏 TVT。

- [ ] **Step 3: 运行完整诊断**

Run:

```powershell
& 'D:\anaconda\envs\fyj\python.exe' scripts\diagnose_rf03_prefix_alignment.py --config configs\rf03_d0_prefix_alignment_v1.json --mode full
```

Expected: 日志每 50 口井报告进度，最终处理 773 井并原子生成正式目录。

- [ ] **Step 4: 独立复算通过规则**

不调用 runner 的 `build_summary`，直接从 `per_well_margins.csv` 重新计算两个 scope × 总体/五折的胜率和 median margin，确认 `summary.json` 逐项一致。

- [ ] **Step 5: 根据预注册规则写结论**

只有以下逻辑：

```python
diagnostic_passes = ncc_passes_all_scopes_and_folds or mae_passes_all_scopes_and_folds
```

不得在看到结果后挑 scope、fold、offset 或指标。

- [ ] **Step 6: 登记结果**

向 `experiments/registry.jsonl` 追加一条 UTF-8 JSON，至少包含：

```json
{
  "experiment_id": "RF03_D0_prefix_alignment_v1",
  "experiment_type": "diagnostic_only",
  "baseline_id": "B00_simple_lgbm_v1",
  "wells": 773,
  "diagnostic_passes": false,
  "artifact_dir": "artifacts/RF03_D0_prefix_alignment_v1"
}
```

`diagnostic_passes` 使用真实结果，不得硬编码示例值。

- [ ] **Step 7: 最终验证与独立复核**

重新运行全套测试、artifact 合同检查、隐藏 TVT 列扫描和 summary 独立复算。若通过规则为真，下一项单独登记循环平移负对照；若为假，归档当前实现并继续路线图下一个预注册实现，不得事后调参数。

---

## Plan Self-Review

- Spec coverage：固定基线、合法输入、offset、scope、门槛、smoke、完整运行、产物和停止结论均有对应任务。
- Placeholder scan：无 `TBD`、`TODO` 或结果后再决定的实现。
- Type consistency：核心表统一使用 `well_id/fold/scope/offset_ft/raw_ncc/affine_median_ae/n_points`；margins 与 bootstrap 都按井和 scope 汇总。
- Scope：只完成 RF03-D0；循环平移负对照和 F03a 在 D0 结果出来后另立实验卡，不在本计划中偷偷实现。
- Git：按用户规则不创建 worktree、不提交、不推送。
