# RF03-D0 Task 2 Brief

## 任务位置

这是 `docs/superpowers/plans/2026-07-17-rf03-d0-prefix-alignment.md` 的 Task 2。

Task 1 已完成并通过独立审查。当前只补齐诊断产物、井级 bootstrap 和原子落盘；不得运行真实三井 smoke 或 773 井正式诊断，不得登记实验结果，不得进入循环平移负对照或 F03a。

## 已冻结且不得改变的事实

- 用户已明确：`B00_simple_lgbm_v1` 就是 `B0`，本任务不得创建其他 B0。
- 本诊断无模型，不训练 LightGBM，不改变 B0 的 12 个特征。
- 只读取水平井 `MD/GR/TVT_input` 与 Typewell `TVT/GR`。
- offset 固定为 `[-20,-10,0,10,20]` ft。
- scope 固定为 `all_visible`、`tail_1000ft`。
- 共同有效点最低 50。
- 通过规则固定为：raw NCC 或 affine-MAE 至少一个指标，在两个 scope 各自的 overall 与五个 fold、共 12 组中，0 ft 最佳率全部严格 `> 50%`。
- bootstrap 只描述稳定性，不能改变以上通过规则。
- 不使用 Git 或 worktree。

## 允许修改的文件

- `H:\kaggle\716\rogii_clean\scripts\diagnose_rf03_prefix_alignment.py`
- `H:\kaggle\716\rogii_clean\tests\test_rf03_prefix_alignment.py`
- 完成后写报告：`H:\kaggle\716\.superpowers\sdd\rf03-d0-task-2-report.md`

除非测试确实暴露纯计算错误，否则不要修改 `src/rf03_prefix_alignment.py`。不得修改配置、实验卡、B0、LightGBM runner、历史 artifact 或 registry。

## 必须保留和新增的接口

保留 Task 1 已有接口及语义：

```python
load_and_validate_diagnostic_config(...)
validate_run_inputs(...)
run_alignment_for_registry(...)
build_summary(...)
```

新增并公开：

```python
def bootstrap_alignment_margins(
    margins: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame: ...

def build_diagnostic_artifacts(...) -> dict[str, Path]: ...
```

函数参数必须显式命名，不能用 `*args`、`**kwargs` 隐藏数据来源。

## TDD 顺序

必须先增加会失败的测试并保存 RED 输出，再实现生产代码。不要先把函数写好再补测试。

### 1. 完整产物合同

用合成的 `offset_scores`、`margins`、registry 和临时输入文件调用 builder。成功目录必须至少包含：

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
offset_scores.csv
per_well.csv
per_well_margins.csv
per_fold.csv
slice_metrics.csv
metrics.json
summary.json
runtime.json
bootstrap_replicates.parquet
conclusion.md
```

正式目录在 builder 完全成功前不得存在。

### 2. predictions.parquet 的含义必须防误读

- 它保存 offset 诊断分数，不是 TVT 预测。
- `predictions.parquet` 与 `offset_scores.csv` 都必须带常量字段：

```text
artifact_role = diagnostic_scores_not_tvt_predictions
```

- 两者按固定键排序后数据逐位一致。
- 禁止出现 `pred_tvt`、`TVT_true`、`target`、`label` 或水平井隐藏 `TVT`。
- 固定键至少为：`well_id/fold/scope/offset_ft`。

### 3. 井级 bootstrap 合同

对每个 scope 分别从该 scope 的有效井中有放回抽样；一行代表一口井，不能按 offset 行抽样。

固定输出：

```text
replicate
scope
seed
sampled_wells
ncc_zero_best_rate
ncc_median_margin
mae_zero_best_rate
mae_median_margin
```

四井、两个 scope、2000 次时必须恰好 4000 行；`replicate` 每个 scope 都是 0..1999；seed 全部为 42。同样输入与 seed 必须逐位复现，不同 seed 至少有一个抽样统计发生变化。

若某 scope 没有有效井，必须明确报错，不能生成看似成功的空 bootstrap。bootstrap 结果仅写入稳定性诊断，不参与 `diagnostic_passes`。

### 4. 固定汇总表语义

- `per_well.csv`：一井 × scope 一行，包含 0 ft 得分、最佳错误 offset 得分、最佳标志、margin、支持点数和 fold。
- `per_well_margins.csv`：一井 × scope 一行，必须与用于 `build_summary`/bootstrap 的 margins 数据逐位一致。
- `per_fold.csv`：两个 scope ×（overall + folds 0..4），固定 12 行；overall 的 fold 用 `-1`。
- `slice_metrics.csv`：只按支持点数做预先固定的描述性切片，区间固定为 `[50,100)`、`[100,250)`、`[250,500)`、`[500,1000)`、`[1000,+inf)`；空切片也保留并写有效井数 0。切片不能参与通过规则。
- `feature_quality.csv`：报告两个 scope 中 raw NCC、affine-MAE、两种 margin 的有效井数、缺失井数和有限值比例；不得读取 target。
- `metrics.json`：保存固定门槛、12 组统计和最终布尔结论。
- `summary.json`：保存 `build_summary` 的完整结果；必须与 `metrics.json` 中同名结论一致。

### 5. 文档型 JSON 的最低内容

- `feature_list.json`：明确这里只有诊断量，不是送入模型的特征；列出 raw NCC、affine-MAE、两种 margin、最佳标志和支持点数。
- `feature_definition.json`：逐项写中文含义、公式、单位、方向（越大/越小越好）和使用位置。
- `feature_lineage.json`：逐项写出来源列；所有量均标记 `uses_hidden_horizontal_tvt=false`、`requires_model_fit=false`、`test_time_legal=true`。
- `negative_control_metrics.json`：只记录四个已运行的固定错误 offset；循环平移必须写 `not_run`，不得写 passed/failed。
- `leakage_tests.json`：至少记录水平井精确列白名单、Typewell 精确列白名单、隐藏 TVT 未加载、公共支持量、registry hash 匹配、禁止预测列扫描以及整体 pass/fail。

### 6. cache_manifest 与 runtime

`cache_manifest.json` 至少记录：

- registry 的路径、SHA-256、井数；
- 本次选定的每口井 horizontal/typewell 文件路径、SHA-256、文件大小；
- 冻结配置、runner、核心模块的 SHA-256；
- offset、scope、minimum_points、bootstrap 次数和 seed；
- artifact_role。

smoke 合成测试只记录被选中的文件；未来 full 必须能记录 773 对文件。

`runtime.json` 至少记录：运行模式、注册井数、选择井数、成功井数、跳过井数、各 scope 有效井数、开始/结束时间、总秒数和代码指纹。测试不得依赖真实时钟的精确值。

### 7. 原子落盘

- 先在目标目录同级创建唯一 staging 目录。
- 所有文件写完后，重新读取关键 CSV/Parquet/JSON，核对 schema、行数、hash、artifact_role 和禁止列。
- 只有全部验证通过，才把 staging 一次性改名为正式目录。
- 正式目录若已存在必须拒绝覆盖。
- 任意写入或验证异常都要清理 staging，并保证正式目录不存在。
- 增加故障注入测试：在写到一半时主动抛异常，断言正式目录和 staging 均不存在。

## 运行边界

本 Task 只能运行合成单测、聚焦测试和全套测试，禁止调用真实 CLI：

```powershell
Set-Location H:\kaggle\716\rogii_clean
& 'D:\anaconda\python.exe' -m pytest tests\test_rf03_prefix_alignment.py --basetemp <工作区内唯一目录> -q
& 'D:\anaconda\python.exe' -m pytest tests --basetemp <工作区内另一个唯一目录> -q
& 'D:\anaconda\python.exe' -m py_compile scripts\diagnose_rf03_prefix_alignment.py
```

不得生成：

```text
rogii_clean/artifacts/RF03_D0_prefix_alignment_v1
rogii_clean/artifacts/_smoke/RF03_D0_prefix_alignment_v1
```

不得追加 `experiments/registry.jsonl`。

## 完成报告

写入 `.superpowers/sdd/rf03-d0-task-2-report.md`，至少包含：

- RED 的真实失败输出；
- 新增接口和产物语义；
- 聚焦测试、全套测试和 py_compile 的真实输出；
- 原子落盘故障注入证据；
- 正式/smoke artifact 不存在的证据；
- 未运行真实数据、未修改 B0/配置/实验卡/registry 的证据；
- 自查和剩余顾虑。

最终只回复 `DONE`、`DONE_WITH_CONCERNS`、`NEEDS_CONTEXT` 或 `BLOCKED` 加一行摘要。
