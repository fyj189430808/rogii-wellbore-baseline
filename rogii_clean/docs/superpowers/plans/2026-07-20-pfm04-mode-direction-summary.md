# PFM04 Mode Direction Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成三个合法井级 PF 模式摘要，并在冻结的 41 列 LightGBM 基线上只增加这三列完成 folds 1–2 筛查。

**Architecture:** 独立生成器从现有 PF128 合法缓存确定性重算井级摘要；独立 CV runner 只按自然井号连接摘要与原行级特征。隐藏真值只在全部合法摘要落盘后进入训练和评分阶段。

**Tech Stack:** Python、NumPy、pandas、PyArrow、LightGBM、pytest。

## Global Constraints

- 不修改 P3B00、PFM01、PFM02、PFM03 的代码和产物。
- 新特征固定为 `direction_score`、`p2_position_raw`、`high_minus_low_separation`。
- 模型固定 1734 棵树、seed 29；原 41 列顺序不变，新三列追加在末尾。
- 先测试与 3 井 smoke，再生成 657 井缓存，最后只跑 folds 1–2。
- 不做 Git 操作。

---

### Task 1: 合法井级摘要缓存

**Files:**
- Create: `rogii_clean/scripts/generate_p3_pfm04_mode_direction_cache.py`
- Create: `rogii_clean/configs/p3_pfm04_mode_direction_summary_v1.json`
- Test: `rogii_clean/tests/test_generate_p3_pfm04_mode_direction_cache.py`

**Interfaces:**
- Consumes: PF128 合法 NPZ、P2-P02 路径、`build_ordered_path_modes` 现有核心。
- Produces: `artifacts/P3_PFM04_mode_direction_summary_v1/legal/per_well.parquet`，每井一行且只含自然键、fold、三个摘要和来源指纹。

- [ ] **Step 1: 写失败测试**：验证输出恰有 657 个唯一 `well_id`、三个特征有限、输入列名单不含 `target_tvt/TVT/oracle/rmse/error`，并验证来源指纹变化时拒绝命中。
- [ ] **Step 2: 运行 RED**：`python -m pytest -q rogii_clean/tests/test_generate_p3_pfm04_mode_direction_cache.py`，预期因生成器不存在而失败。
- [ ] **Step 3: 最小实现**：逐井加载 PF128，复用 PFM01 排序逻辑，原子写每井 checkpoint，最后合并为一井一行的 parquet。
- [ ] **Step 4: 运行 GREEN**：同一测试命令预期全部通过；再运行 `python rogii_clean/scripts/generate_p3_pfm04_mode_direction_cache.py --max-wells 3 --workers 1`，预期 `hidden_tvt_read=false`。

### Task 2: 冻结 LightGBM folds 1–2

**Files:**
- Create: `rogii_clean/scripts/run_p3_pfm04_mode_direction_summary_cv.py`
- Test: `rogii_clean/tests/test_run_p3_pfm04_mode_direction_summary_cv.py`

**Interfaces:**
- Consumes: Task 1 的每井摘要、P3B00 原 41 列特征、固定 fold 与模型配置。
- Produces: fold 模型、OOF 预测、逐井/逐折指标和 `metrics_folds12.json`。

- [ ] **Step 1: 写失败测试**：验证准确 44 列、前三个摘要只按 `well_id` 广播、特征列表不含模式路径、模型配置 SHA 与基线一致。
- [ ] **Step 2: 运行 RED**：`python -m pytest -q rogii_clean/tests/test_run_p3_pfm04_mode_direction_summary_cv.py`，预期因 runner 不存在而失败。
- [ ] **Step 3: 最小实现**：复用 PFM03 的折外训练和指标框架，只把路径缓存连接替换为一井一行摘要连接。
- [ ] **Step 4: 运行 GREEN**：运行两份 PFM04 测试和 `py_compile`，预期无失败。
- [ ] **Step 5: 正式运行**：先生成 657 井缓存，再执行 `python rogii_clean/scripts/run_p3_pfm04_mode_direction_summary_cv.py --folds 1,2`；按预注册门槛停止或晋级。

