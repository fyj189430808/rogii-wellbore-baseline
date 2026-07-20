# P3-PFS01 Fixed-Lag Particle Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改旧 PF 基线的情况下，新增能用未来 250/500/1000 ft GR 修正较早粒子状态的三条合法路径，并按“原 41 列 + 新三列 = 44 列”的冻结单模 LightGBM 合同评价。

**Architecture:** 新建独立 Numba 内核，复用 P2-P01 的输入准备和 scale8 聚合。内核逐 seed 保存最大 1000 ft 的活动祖先历史，重采样时同步重排历史；runner 先生成逐井无标签缓存，再单独做诊断/模型阶段。

**Tech Stack:** Python 3、NumPy 1.26.4、Numba 0.60.0、pandas、pytest、LightGBM（只在路径 smoke 通过后复用冻结入口）。

## Global Constraints

- 不修改 `src/p2_p01_multiseed_pf.py` 的旧内核。
- 500 粒子、128 seeds、全部状态转移/观测/重采样参数冻结。
- lag 固定为 250/500/1000 ft；seed 聚合固定 scale8。
- shadow 116 井保持关闭；路径生成阶段不得读取隐藏 TVT。
- 不做 Git 操作，由用户自行处理。

---

### Task 1: 固定滞后内核与数值合同

**Files:**
- Create: `src/p3_pfs01_fixed_lag_smoothing.py`
- Create: `tests/test_p3_pfs01_fixed_lag_smoothing.py`

**Interfaces:**
- Consumes: `prepare_particle_filter_inputs()` 返回的冻结 PF 参数字典。
- Produces: `particle_filter_fixed_lag_all_seeds_numba(..., lag_distances_ft) -> (filtered_paths, smoothed_paths, final_ll)`，其中 shape 分别为 `[S,H]`、`[L,S,H]`、`[S]`。

- [ ] **Step 1: 先写失败测试**

测试必须覆盖：非法 lag；合成单 seed 的未来证据能改变旧状态；尾段回退；lag=0 与旧内核一致；隐藏 TVT 不属于函数输入。

- [ ] **Step 2: 运行 RED**

```powershell
python -m pytest -q tests/test_p3_pfs01_fixed_lag_smoothing.py
```

预期：模块或目标函数不存在而失败。

- [ ] **Step 3: 最小实现**

实现冻结 PF 的同序随机调用；每个 seed 保存 `filtered_path[H]`、`active_history[R,500]`，其中 `R` 根据 MD 与 1000 ft 的最大活动行数计算。重采样时必须对所有活动历史行应用与当前粒子相同的 `source_indices`。到达 lag 后用当前权重计算祖先 `U-Z_old`；未定稿尾段复制 `filtered_path`。

- [ ] **Step 4: 运行 GREEN 与旧内核回归测试**

```powershell
python -m pytest -q tests/test_p3_pfs01_fixed_lag_smoothing.py tests/test_p2_p01_multiseed_pf.py
python -m py_compile src/p3_pfs01_fixed_lag_smoothing.py
```

预期：全部通过。

---

### Task 2: 合法逐井缓存 runner

**Files:**
- Create: `scripts/run_p3_pfs01_fixed_lag_particle_smoothing.py`
- Create: `configs/p3_pfs01_fixed_lag_particle_smoothing_v1.json`
- Modify: `tests/test_p3_pfs01_fixed_lag_smoothing.py`

**Interfaces:**
- Consumes: 原始训练井合法列、Typewell TVT/GR、Task 1 内核、固定 fold/shadow registry。
- Produces: `artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_cache/<well>.parquet`，列为 `well_id/fold/row_index/last_visible_tvt/pfs_lag250_delta/pfs_lag500_delta/pfs_lag1000_delta/_cache_fingerprint`。

- [ ] **Step 1: 先写失败测试**

覆盖 shadow 排除、只读合法 CSV 列、缓存指纹、原子写入、行键、重复运行一致性和隐藏 TVT 扰动不变。

- [ ] **Step 2: 运行 RED**

```powershell
python -m pytest -q tests/test_p3_pfs01_fixed_lag_smoothing.py
```

- [ ] **Step 3: 实现 smoke/all/resume CLI**

```text
--mode smoke     固定三井并打印逐井耗时
--mode all       657 开发井，逐井原子缓存，可续跑
--force          仅重算当前 scope
```

合法阶段必须保存 `hidden_tvt_read=false`，并打印每口井隐藏行数、最大活动历史行数、运行秒数和预计剩余时间。

- [ ] **Step 4: 三井 smoke**

```powershell
python scripts/run_p3_pfs01_fixed_lag_particle_smoothing.py --mode smoke --force
```

验收：三井输出有限、行键一致、尾段回退正确、总耗时不超过 10 分钟。

---

### Task 3: 路径只读诊断与冻结 44 列模型合同

**Files:**
- Modify: `scripts/run_p3_pfs01_fixed_lag_particle_smoothing.py`
- Create: `tests/test_p3_pfs01_model_contract.py`

**Interfaces:**
- Consumes: 全部合法缓存，以及只在缓存完整后读取的隐藏 TVT。
- Produces: 路径自身 RMSE；44 列特征表，在原 P3B00 41 列后只加入三个 `pfs_lag*_delta`，旧特征逐名保留。

- [ ] **Step 1: 先写失败测试**

断言正式列数为 44、原 41 列逐名且顺序不变、末尾只新增三列、模型参数 JSON 哈希与 P3B00 相同。

- [ ] **Step 2: 跑 RED 后实现最小合并**

按 `well_id/fold/row_index` 一对一合并；任何重复、缺行或 shadow 重叠立即失败。

- [ ] **Step 3: 先评价三条路径自身**

输出每个 lag 的 micro/macro/per-fold RMSE 和相对旧 scale8 的差，不训练模型。

- [ ] **Step 4: 只有 smoke 与路径诊断合理时运行 folds0-1**

复用冻结 `train_fold()`；模型为原 1734 棵、seed29 单模 LightGBM。若合并改善不足 0.20 ft 或任一折恶化超过 0.25 ft，按预注册门槛停止 folds2-4。

---

### Task 4: 验证、登记和继续/停止判断

**Files:**
- Create/update: `artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/*`
- Modify: `experiments/registry.jsonl`
- Modify: `experiments/current_state.json`

- [ ] **Step 1: 保存完整产物**

必须有 `config.json/metrics.json/per_well.csv/predictions.parquet/feature_list.json/parameter_list.json/runtime.json/conclusion.md`。

- [ ] **Step 2: 防泄漏与复现验证**

修改隐藏 TVT 后路径逐位不变；相同配置重复运行逐位不变；shadow overlap 为 0；评价前缓存 657/657 且指纹一致。

- [ ] **Step 3: 按门槛作结论**

明确区分事实、推断、猜测、当前实现的否定边界和下一项最便宜实验。正负结果都追加 registry；不得事后改门槛。
