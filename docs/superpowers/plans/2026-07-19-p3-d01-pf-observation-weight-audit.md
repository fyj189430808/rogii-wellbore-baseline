# P3-D01 PF Observation Weight Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不读取影子井目标、不修改正式模型的前提下，精确重放 657 口开发井的冻结 128-seed PF，审计 GR 缺失、似然尺度、固定温度权重和路径误差之间的关系。

**Architecture:** 新增一个纯数值核心模块，负责 GR 证据统计、稳定 softmax、ESS/熵和单井 PF 重放；新增一个 runner，负责影子过滤、逐井紧凑缓存、开发 OOF 评分、跨折相关与产物写出。旧 P2-P01 PF 内核和 P2-P02 预测只读复用，不修改。

**Tech Stack:** Python 3、NumPy 1.26.4、Pandas、Numba 0.60.0、PyArrow、pytest。

## Global Constraints

- 基线固定为 `P3B00_group5_p2p02_v1`，micro RMSE `10.305704992073148`。
- 主 CV 固定为 `balanced_well_5fold_v1`，完整井为验证单位。
- 影子集固定为 `P3_shadow_holdout_v1` 的 116 口井，隐藏目标与误差保持未打开。
- 合法生成只读取水平井 `MD/Z/GR/TVT_input` 和 Typewell `TVT/GR`。
- PF 固定为 P2-P01 的 500 粒子、128 seed、seed base 0 和全部原参数。
- 本实验只做诊断，不训练 LightGBM，不生成正式特征或正式预测。
- 用户负责 Git；本计划不创建 worktree、不提交、不推送。
- 每个新函数必须先出现会按预期失败的测试，再写生产实现。

---

### Task 1: 权重数学和 GR 证据纯函数

**Files:**
- Create: `rogii_clean/tests/test_p3_d01_pf_observation_audit.py`
- Create: `rogii_clean/src/p3_d01_pf_observation_audit.py`

**Interfaces:**
- Produces: `seed_weights(log_likelihoods, temperature) -> np.ndarray`
- Produces: `summarize_weight_scale(log_likelihoods, temperature) -> dict[str, float]`
- Produces: `hidden_gr_evidence_statistics(horizontal_well) -> dict[str, float | int]`
- Produces: `visible_prefix_gr_diagnostics(horizontal_well, typewell, parameters) -> dict[str, float | int | bool]`

- [ ] **Step 1: 写稳定 softmax、ESS 和熵的失败测试**

```python
def test_uniform_seed_likelihoods_have_full_effective_path_count() -> None:
    log_likelihoods = np.zeros(128, dtype=np.float64)
    report = summarize_weight_scale(log_likelihoods, temperature=3.0)
    assert report["ess"] == pytest.approx(128.0)
    assert report["max_weight"] == pytest.approx(1.0 / 128.0)
    assert report["normalized_entropy"] == pytest.approx(1.0)


def test_weight_summary_is_stable_for_large_negative_likelihoods() -> None:
    log_likelihoods = np.array([-100000.0, -100001.0, -100010.0])
    weights = seed_weights(log_likelihoods, temperature=5.0)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]
```

- [ ] **Step 2: 运行 RED**

Run: `python -m pytest rogii_clean/tests/test_p3_d01_pf_observation_audit.py -q --basetemp .pytest_tmp_p3_d01_core_red`

Expected: `ModuleNotFoundError: No module named 'src.p3_d01_pf_observation_audit'`。

- [ ] **Step 3: 最小实现权重函数**

```python
def seed_weights(log_likelihoods: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(log_likelihoods, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("累计似然必须是一维有限数组")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("温度必须为正数")
    scaled = (values - values.max()) / float(temperature)
    weights = np.exp(scaled)
    weights /= weights.sum()
    return weights


def summarize_weight_scale(log_likelihoods: np.ndarray, temperature: float) -> dict[str, float]:
    weights = seed_weights(log_likelihoods, temperature)
    positive = weights > 0.0
    entropy = -float(np.sum(weights[positive] * np.log(weights[positive])))
    return {
        "ess": float(1.0 / np.sum(np.square(weights))),
        "max_weight": float(weights.max()),
        "entropy": entropy,
        "normalized_entropy": float(entropy / np.log(len(weights))),
    }
```

- [ ] **Step 4: 添加 GR 缺失和前缀标定失败测试**

使用一个含 6 个隐藏行、缺失运行长度 2 的小表，断言 `observed_gr_rows=4`、`interpolated_gr_rows=2`、`longest_gr_gap_rows=2`、`effective_observation_count=4`。另用线性构造 `horizontal_GR = 2 × typewell_GR + 5`，断言 affine slope 2、intercept 5；把隐藏 `TVT` 列任意改写，结果必须完全不变。

- [ ] **Step 5: 实现 GR 统计并运行 GREEN**

实现时只索引 `MD/GR/TVT_input` 和 Typewell `TVT/GR`。当前 `gr_sigma` 必须通过旧 `prepare_particle_filter_inputs` 获得，observed-only sigma 只使用可见且 GR 非空的行。运行同一聚焦测试直到全部通过。

---

### Task 2: 单井精确 PF 重放与紧凑似然缓存

**Files:**
- Modify: `rogii_clean/tests/test_p3_d01_pf_observation_audit.py`
- Modify: `rogii_clean/src/p3_d01_pf_observation_audit.py`

**Interfaces:**
- Consumes: P2-P01 `prepare_particle_filter_inputs`、`_kernel_arguments`、`particle_filter_all_seeds_numba`
- Produces: `run_pf_likelihood_audit(horizontal_well, typewell, parameters) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]`
- Produces: `write_likelihood_cache(path, seed_log_likelihoods, fingerprint) -> None`
- Produces: `read_likelihood_cache(path, expected_fingerprint) -> np.ndarray`

- [ ] **Step 1: 写单井小参数重放失败测试**

小测试使用 8 个粒子、3 个 seed 和 5 个隐藏行，断言：

```python
seed_ll.shape == (3,)
set(paths) == {"mean", "scale_3", "scale_5", "scale_8", "scale_12"}
all(len(path) == 5 for path in paths.values())
report["hidden_rows"] == 5
```

测试同时传入含伪造隐藏 `TVT` 的水平井，改写该列后 `seed_ll`、paths 和合法 report 必须逐位相同。

- [ ] **Step 2: 运行 RED**

Expected: `ImportError` 或 `AttributeError`，因为 `run_pf_likelihood_audit` 尚未定义。

- [ ] **Step 3: 实现最小单井审计**

```python
prepared = prepare_particle_filter_inputs(horizontal_well, typewell, parameters)
seed_predictions, seed_ll = particle_filter_all_seeds_numba(**_kernel_arguments(prepared))
mean_path = seed_predictions.mean(axis=0)
scale_paths = {
    f"scale_{int(scale)}": seed_weights(seed_ll, scale) @ seed_predictions
    for scale in (3.0, 5.0, 8.0, 12.0)
}
```

最终 report 合并 GR 统计、似然均值/标准差/范围、每个温度的 ESS/max/entropy。逐行路径只返回调用者，不写入永久缓存。

- [ ] **Step 4: 写缓存指纹失败测试**

先写 3 个似然和指纹 `abc`，用 `abc` 读取应逐位相同；用 `def` 读取必须抛 `ValueError`；缓存内容不得出现 target/error/rmse/oracle 字段。

- [ ] **Step 5: 实现原子 NPZ 缓存并运行 GREEN**

缓存只包含 `seed_log_likelihoods` 和 `_cache_fingerprint`。写临时文件后替换；读取时验证长度、有限值、指纹和 SHA 所需内容。

---

### Task 3: 影子隔离、可恢复 runner 与 smoke 复现

**Files:**
- Create: `rogii_clean/tests/test_diagnose_p3_d01_pf_observation_weights.py`
- Create: `rogii_clean/scripts/diagnose_p3_d01_pf_observation_weights.py`
- Create: `rogii_clean/configs/p3_d01_pf_observation_weight_audit_v1.json`

**Interfaces:**
- Produces: `load_development_registry(fold_path, shadow_path) -> pd.DataFrame`
- Produces: `read_legal_inputs(raw_train_dir, well_id) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `generate_one_well(task) -> dict[str, Any]`
- Produces: `run_legal_generation(tasks, workers) -> tuple[list[dict], list[dict]]`
- Produces CLI modes `smoke`, `fold01`, `all`

- [ ] **Step 1: 写 shadow 排除和合法列失败测试**

构造 5 井 fold 表和 2 井 shadow 表，断言返回 3 井且交集为 0。用临时 CSV 调 `read_legal_inputs`，断言水平井只含 `MD/Z/GR/TVT_input`，Typewell 只含 `TVT/GR`。

- [ ] **Step 2: 运行 RED**

Expected: `ModuleNotFoundError: No module named 'scripts.diagnose_p3_d01_pf_observation_weights'`。

- [ ] **Step 3: 实现配置、注册表和任务生成**

配置固定：

```json
{
  "experiment_id": "P3_D01_pf_observation_weight_audit_v1",
  "fold_version": "balanced_well_5fold_v1",
  "expected_total_wells": 773,
  "expected_shadow_wells": 116,
  "expected_development_wells": 657,
  "workers": 8,
  "likelihood_scales": [3.0, 5.0, 8.0, 12.0],
  "model_training": false,
  "formal_feature_output": false,
  "shadow_target_access": false
}
```

另外保存当前已核对 SHA：fold `a70bc21e...a241c`、shadow `7fde7c16...83c1`、P2-P01 config `f4e9ba4e...243b`、PF core `b636982d...0782`、P2-P02 predictions `8109514e...d37`。

- [ ] **Step 4: 写 cache hit/resume 失败测试**

用临时目录生成一口小井后第二次运行，断言第二次 `cache_hit=True`；改变 fingerprint 后必须重算；runtime 明确写 `hidden_tvt_read=false` 和合法列名单。

- [ ] **Step 5: 实现线程池和逐井恢复**

每完成一井打印 `完成数/总数、井号、缓存或新算、秒数`。任一失败写入 errors 并最终非零退出，不得带着缺井结果进入评分。

- [ ] **Step 6: 运行真实 smoke**

Run:

```powershell
python rogii_clean/scripts/diagnose_p3_d01_pf_observation_weights.py --mode smoke
```

读取同井 P2-P01 legal cache，比较 mean/scale 3/5/8/12，要求最大绝对差为 0。若不为 0，停止全量运行并保存 smoke 诊断。

---

### Task 4: 开发 OOF 评分、相关分析和路线判断

**Files:**
- Modify: `rogii_clean/tests/test_diagnose_p3_d01_pf_observation_weights.py`
- Modify: `rogii_clean/scripts/diagnose_p3_d01_pf_observation_weights.py`

**Interfaces:**
- Produces: `load_development_targets(prediction_path, shadow_ids) -> pd.DataFrame`
- Produces: `score_development_paths(...) -> pd.DataFrame`
- Produces: `spearman_report(frame, x, y, permutations, seed) -> dict[str, float]`
- Produces: `analyze_diagnostic(per_well, config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]`

- [ ] **Step 1: 写目标过滤和逐井路径 RMSE 失败测试**

临时 Parquet 同时放开发井和 shadow 井，返回表只能含开发井。构造两条手算路径，验证 SSE、pooled RMSE 和 oracle best scale；输出中必须保留 `best_scale_is_oracle=True`。

- [ ] **Step 2: 运行 RED**

Expected: 缺少目标加载或分析函数。

- [ ] **Step 3: 实现 Arrow 过滤和 P2-P01 路径连接**

目标读取列固定为 `well_id/fold/row_index/target_tvt/pred_tvt`。每井 P2-P01 cache 只读取 `row_index/last_visible_tvt/pf128_mean_tvt/pf128_scale_*_delta`，按键一对一连接。

- [ ] **Step 4: 写相关、跨折和置换失败测试**

使用完全单调的 10 井表，断言 Spearman 为 1；固定 seed 重复调用，置换比例完全一致；五折方向统计必须按 fold 单独计算，不能用全表结果复制。

- [ ] **Step 5: 实现汇总和预登记判断**

产出：

```text
per_well.csv
per_fold_correlations.csv
overall_correlations.csv
binned_metrics.csv
scale_path_metrics.csv
summary.json
runtime.json
conclusion.md
```

判断严格使用设计文档中的 PF01/PF02 阈值。结论必须分事实、推断、未验证、只能否定、下一步。

- [ ] **Step 6: 运行聚焦测试和三阶段回归测试**

Run:

```powershell
python -m pytest rogii_clean/tests/test_p3_d01_pf_observation_audit.py rogii_clean/tests/test_diagnose_p3_d01_pf_observation_weights.py rogii_clean/tests/test_freeze_p3_b00.py rogii_clean/tests/test_p3_shadow_holdout.py rogii_clean/tests/test_p3_d00_residual_structure.py rogii_clean/tests/test_diagnose_p3_d00_residual_structure.py -q --basetemp .pytest_tmp_p3_d01_preflight
```

Expected: 0 failures。

---

### Task 5: 完整 657 井运行、产物核验和状态登记

**Files:**
- Modify: `rogii_clean/experiments/phase3_roadmap.md`
- Modify: `rogii_clean/experiments/current_state.json`
- Modify: `rogii_clean/experiments/registry.jsonl`

**Interfaces:**
- Consumes: Task 3/4 的 CLI 和缓存
- Produces: 完整 `P3_D01_pf_observation_weight_audit_v1` 产物与下一实验编号

- [ ] **Step 1: 运行完整开发集**

Run:

```powershell
python rogii_clean/scripts/diagnose_p3_d01_pf_observation_weights.py --mode all
```

运行前打印 657 口井、8 workers、预计 45～70 分钟、输出目录和同命令恢复方式。每口井结束时输出进度。

- [ ] **Step 2: 核验产物合同**

断言：

```text
per_well 井数 = 657
开发自然隐藏行 = 3,211,872
shadow overlap = 0
每口井 seed_ll 数量 = 128
ESS 全部位于 [1, 128]
五折均出现且井数 = 131/132/131/132/131
没有 model.txt、feature_list.json 或正式 predictions.parquet
```

- [ ] **Step 3: 更新路线图、状态和 registry**

`current_state.json` 的 latest 改为 D01，记录 PF01/PF02 支持判断和下一项。registry 追加一条完整 JSON 记录；不得改写旧实验结论。

- [ ] **Step 4: 最终新鲜验证**

重新运行 Task 4 的完整测试命令，解析所有 JSON/JSONL，运行 `git diff --check`。只有命令退出 0 后才能汇报完成。

## Plan Self-Review

- Spec coverage：合法生成、128 似然、四温度 ESS、影子隔离、路径评分、跨折分析、置换对照和路线判断均有对应任务。
- Placeholder scan：没有 TBD、TODO、模糊的“适当处理”或未定义接口。
- Type consistency：核心返回 `np.ndarray + dict + path dict`；runner 缓存只保存一维 128 似然；评分只消费开发目标和冻结四温度路径。
- Scope：只实现 D01，不提前改 PF01/PF02，不训练模型。
