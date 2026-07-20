# UP08 Strict OOF PFS Weight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用现有 outer0 严格嵌套 P3B00 预测训练一个强收缩岭回归，为每井预测 PFS 修正权重，并评价是否超过固定权重 UP03。

**Architecture:** 数值核心负责投影、权重目标、井级合法特征和收缩预测；runner 只负责验证现有 R01a 嵌套缓存、拼接 PFS 缓存、拟合 outer-train 预处理与 Ridge、评价 outer fold0。oracle 与正式预测物理分开。

**Tech Stack:** Python、NumPy、pandas、scikit-learn、Parquet、pytest。

## Global Constraints

- 主 LightGBM、41 列、fold、评价行不变。
- 只跑 outer fold0；fold0 标签只用于最终评分。
- 影子 116 井不得读取。
- 不执行 Git；用户负责提交。

---

### Task 1: 数值核心与严格目标

**Files:**
- Create: `src/p3_up08_strict_oof_pfs_weight.py`
- Test: `tests/test_p3_up08_strict_oof_pfs_weight.py`

**Interfaces:**
- Produces: `optimal_alpha(target, up01, direction) -> tuple[alpha, energy]`
- Produces: `build_well_features(md, base_pred, pfs_paths, runtime) -> dict[str, float]`
- Produces: `shrink_alpha(raw, center=0.25, strength=0.5) -> ndarray`

- [ ] 写失败测试：闭式 alpha、零能量回退、跨井特征、收缩与截断。
- [ ] 运行 `python -m pytest -q --basetemp artifacts/_pytest_up08 tests/test_p3_up08_strict_oof_pfs_weight.py`，确认因模块缺失失败。
- [ ] 实现最小纯函数；`alpha=sum(d*(y-up01))/sum(d*d)`，零能量返回 `0.25`。
- [ ] 重跑专项测试，要求全部通过。

### Task 2: outer0 严格 runner

**Files:**
- Create: `scripts/run_p3_up08_strict_oof_pfs_weight.py`
- Create: `configs/p3_up08_strict_oof_pfs_weight_v1.json`
- Modify: `tests/test_p3_up08_strict_oof_pfs_weight.py`

**Interfaces:**
- Consumes: `artifacts/P3_R01a_nested_linear_residual_v1/base_models/outer_0/**/predictions.parquet`
- Consumes: `artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_cache/*.parquet`
- Produces: `artifacts/P3_UP08_strict_oof_pfs_weight_v1/`

- [ ] 写失败测试：拒绝非 outer0 缓存、拒绝 shadow、训练/验证井不交叉、预处理只拟合 outer-train、shuffle 使用固定 seed 42。
- [ ] 验证五个嵌套预测文件的井集合、行数、fold 和模型指纹；任何不匹配直接退出。
- [ ] 分别从 inner OOF 与 outer prediction 生成 UP01、PFS direction 和合法井级特征。
- [ ] 用 outer-train 中位数与标准差预处理，训练 `Ridge(alpha=100.0)`；预测后执行 50% 收缩与 `[-0.25,0.75]` 截断。
- [ ] 生成固定 alpha0.25、预测 alpha、shuffle-alpha 三条路径；fold0 真值只在三条路径落盘后读取。
- [ ] 保存 `config.json`、`lineage_audit.json`、`train_targets.csv`、`predicted_alpha.csv`、`predictions.parquet`、`metrics.json`、`per_well.csv`、`runtime.json`、`conclusion.md`。
- [ ] 运行专项测试和 `python -m py_compile src/p3_up08_strict_oof_pfs_weight.py scripts/run_p3_up08_strict_oof_pfs_weight.py`。

### Task 3: outer0 正式运行与门槛

**Files:**
- Run: `scripts/run_p3_up08_strict_oof_pfs_weight.py`

- [ ] 运行 3 井 smoke，只验证 shape、有限值和影子隔离。
- [ ] 运行 outer0 正式实验；主要耗时应为读取现有嵌套预测，不重训 P3B00。
- [ ] 核对固定 alpha0.25 基线与同源 UP03 fold0 数值一致。
- [ ] 报告 RMSE 改善、胜率、bootstrap、alpha 相关性、截断比例与 shuffle 对照。
- [ ] 只有改善至少 `0.10 ft`、胜率至少 55%、shuffle 收益消失时，才另行设计完整五折；否则停止当前 Ridge 实现。

