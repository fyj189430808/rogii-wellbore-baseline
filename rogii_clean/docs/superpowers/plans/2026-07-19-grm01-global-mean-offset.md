# GRM01 Global Mean Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用当前井完整真实 GR 在严格 P2-P02 路径周围选择一个整井常数偏移，并审计真实平均残差的候选排名和路径 RMSE。

**Architecture:** 数学核心与运行脚本分离。合法阶段只读取无标签 outer0 路径和当前井可见/隐藏合法信息，保存全部候选得分与选择；随后评分阶段才读取 target_tvt，附加 oracle、排名和 RMSE。循环错移只移动有限 GR 值，不改变缺失位置。

**Tech Stack:** Python、NumPy、pandas、scikit-learn HuberRegressor、pytest、Parquet。

## Global Constraints

- 固定候选 `np.arange(-30, 31, 1)`，不得在 outer0 后修改。
- 固定 Huber 前缀校准（至少 30 个配对点）、300 ft 块、共同支持、稳健损失和 0.05 二次回零惩罚。
- 前缀 Huber 残差尺度固定为 `max(1.0, 1.4826 * MAD)`；隐藏段逐点损失固定为 `min(abs(error) / scale, 3)`。
- 前缀配对不足、GR 近似常数或 Huber 数值异常时确定性回退为证据不足和 `0 ft`，不得猜测校准参数。
- 所有候选共享相同行、相同有效块和同一个前缀 a/b。
- 初次读取 outer0 路径时物理排除 target_tvt，选中偏移保存后才读取真值。
- 只使用原始有限隐藏 GR；不填补、不使用影子井、不训练模型。
- 不进行 Git、分支、提交或 worktree 操作。

---

### Task 1: 实现数学核心和 outer0 审计

**Files:**
- Create: `rogii_clean/src/p3_grm01_global_mean_offset.py`
- Create: `rogii_clean/scripts/diagnose_p3_grm01_global_mean_offset.py`
- Create: `rogii_clean/tests/test_p3_grm01_global_mean_offset.py`
- Produce: `rogii_clean/artifacts/P3_GRM01_global_mean_offset_v1/`

**Interfaces:**
- Consumes: strict outer0 prediction parquet、raw horizontal/typewell CSV、开发井/影子井注册表。
- Produces: 每井 61 个真实/错移 GR 合法得分、selected_c、true_m_rank、Top-K、方向率、路径预测和统一指标。

- [ ] **Step 1: 先写失败测试**

测试候选网格与平均并列名次、最近网格的固定平分规则、共同 Typewell 支持、300 ft 块等权损失、MAD 尺度、二次惩罚、证据不足回零、有限 GR 循环错移保持 NaN 位置、已知合成偏移可恢复、合法打分接口不接受 target_tvt。

- [ ] **Step 2: 运行红测试**

Run: `python -m pytest rogii_clean/tests/test_p3_grm01_global_mean_offset.py -q`

Expected: 因核心模块不存在而失败。

- [ ] **Step 3: 实现最小核心与脚本**

复用 `clean_typewell_arrays`；前缀至少 30 个配对点时用 HuberRegressor 拟合一次，并用其残差计算 `max(1.0, 1.4826*MAD)`。隐藏段把候选 TVT 构造成 `[隐藏行,61]`，只保留所有候选都在 Typewell 范围内且原始 GR 有限的共同支持行。逐点计算 `min(abs(error)/scale,3)`，再按 300 ft 有效块等权汇总并加固定惩罚。前缀不足或校准异常时回退为无证据和 `0 ft`。合法输出不得包含 target、true_m、RMSE 或 rank。

- [ ] **Step 4: 测试、小样本和 outer0**

Run: `python -m pytest rogii_clean/tests/test_p3_grm01_global_mean_offset.py -q`

Expected: 全部通过。

Run: `python rogii_clean/scripts/diagnose_p3_grm01_global_mean_offset.py --max-wells 3 --output-dir rogii_clean/artifacts/P3_GRM01_global_mean_offset_v1_smoke`

Expected: 三井完成、合法产物无目标列、错移保持支持量一致。

Run: `python rogii_clean/scripts/diagnose_p3_grm01_global_mean_offset.py`

Expected: 131 井、651881 行，输出真实/错移排名和路径指标。

- [ ] **Step 5: 独立复算并登记**

从逐行预测独立复算基础、真实 GR、错移 GR 和 oracle RMSE；核对合法候选表没有目标列；按实验卡门槛写 conclusion 并登记 registry。
