# R01b Mean Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用固定 21 个井级特征和 Ridge10，严格预测每口井 P2-P02 隐藏段的平均残差，并评价 outer fold 0 路径。

**Architecture:** 新脚本只读取 R01a 已保存的 inner folds 1-4 与 outer0 预测，并复用 R01a 简化版的特征构造函数。训练表是一井一行；系数层训练与基础 LightGBM 完全分离。

**Tech Stack:** Python、pandas、NumPy、scikit-learn、pytest、Parquet。

## Global Constraints

- 不训练或复用任何不满足当前 outer0 严格嵌套边界的基础模型。
- 不读取 116 口影子井目标、旧 global OOF 残差或目标诊断特征。
- 固定 21 个简化特征、Ridge alpha=10、shuffle seed=20260719。
- 不使用斜率、二次项、控制点、收缩搜索或逐行模型。
- 不进行 Git、分支、提交或 worktree 操作。

---

### Task 1: 实现并运行 R01b 系数层

**Files:**
- Create: `rogii_clean/scripts/run_p3_r01b_mean_residual_ridge.py`
- Create: `rogii_clean/tests/test_p3_r01b_mean_residual_ridge.py`
- Produce: `rogii_clean/artifacts/P3_R01b_mean_residual_ridge_v1/`

**Interfaces:**
- Consumes: `run_p3_r01a_nested_linear_residual.load_feature_table`、`build_well_features`、R01a 的五个严格嵌套 prediction parquet。
- Produces: `fit_mean_residual_target(predictions) -> DataFrame[well_id, fold, mean_residual]`，以及 outer0 修正预测与统一指标。

- [ ] **Step 1: 先写失败测试**

测试用两口合成井验证：目标必须等于逐行残差均值；应用预测常数时整条路径每行增加相同数值；目标打乱保持井数但改变井与目标的对应关系。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest rogii_clean/tests/test_p3_r01b_mean_residual_ridge.py -q`

Expected: 因 R01b 函数尚不存在而失败。

- [ ] **Step 3: 写最小实现**

脚本读取四个 inner OOF 预测和一个 outer0 预测，精确核对 526/131 口井、行键覆盖和影子井交集。用 `mean(target_tvt - pred_tvt)` 构造单目标；用固定 21 特征进行 outer-train 中位数填充、标准化和 Ridge10；生成正式预测、shuffle 预测和真实均值 oracle。

- [ ] **Step 4: 运行测试与正式实验**

Run: `python -m pytest rogii_clean/tests/test_p3_r01b_mean_residual_ridge.py -q`

Expected: 全部通过。

Run: `python rogii_clean/scripts/run_p3_r01b_mean_residual_ridge.py`

Expected: 不出现 LightGBM 训练日志；输出 131 口井、651881 行的基础/修正/shuffle/oracle 指标。

- [ ] **Step 5: 复算并登记**

从保存的逐行 Parquet 独立复算基础和修正 RMSE，核对特征数为 21，并将结论写入 artifact 目录与 `experiments/registry.jsonl`。
