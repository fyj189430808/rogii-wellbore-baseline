# P2-P03 PF 随机种子分歧执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在冻结 P2-P02 上只加入 `pf128_seed_std`，按预注册门槛完成自动停止或完整五折。

**Architecture:** 新 runner 复用现有数据加载、LightGBM 训练和评分函数，不修改冻结的 P01/P02 文件。P03 自己冻结 P02 OOF 和 42 列合同，按井合并一列 seed 标准差并运行两阶段门槛。

**Tech Stack:** Python、pandas、NumPy、LightGBM、pytest、Parquet。

## Global Constraints

- 主 CV 必须是 `balanced_well_5fold_v1`，验证单位为完整井。
- 正式模型必须是冻结参数的单个 LightGBM，1,734 棵树、seed 29。
- P2-P02 41 列顺序不变，只在末尾增加 `pf128_seed_std`，共 42 列。
- 不读取隐藏 TVT、surface、oracle 或邻井标签构造正式特征。
- 用户负责 Git；本计划不创建 worktree、不提交、不推送。

---

### Task 1：先写冻结合同与缓存合并测试

**Files:**
- Create: `rogii_clean/tests/test_p2_p03_cv_runner.py`
- Create: `rogii_clean/scripts/run_p2_p03_pf_seed_dispersion_cv.py`

**Interfaces:**
- Consumes: P2-P02 41 列清单、P01 legal cache、P02 OOF。
- Produces: `FROZEN_MODEL_FEATURES` 42 列、`validate_frozen_contract()`、`merge_p01_path_and_dispersion_cache()`。

- [ ] 写测试，冻结 42 列顺序、完整 LightGBM 参数、P02 哈希和严格按井五折。
- [ ] 运行测试，确认因 P03 runner 尚不存在而失败。
- [ ] 实现最少代码，使冻结合同测试通过。
- [ ] 写并运行缓存异常测试：缺 std、NaN/Inf、错误指纹、错误锚点、重复/缺失行键和禁止列必须失败。

### Task 2：实现两阶段训练、P02 比较和负对照

**Files:**
- Modify: `rogii_clean/scripts/run_p2_p03_pf_seed_dispersion_cv.py`
- Modify: `rogii_clean/tests/test_p2_p03_cv_runner.py`

**Interfaces:**
- Consumes: 42 列特征表、冻结 P02 OOF、统一训练函数。
- Produces: folds 0～1 比较、负对照、完整五折及标准 artifacts。

- [ ] 先写失败测试，确认门槛比较对象必须是 P02，bootstrap 上界等于 0 时不能通过。
- [ ] 实现 folds 0～1 训练与 `comparison_vs_p02_folds01.json`。
- [ ] 先写失败测试，确认负对照只反转 std 且保持每井值多重集和其余 41 列不变。
- [ ] 实现 fold 0 负对照和 `0.05 ft` 门槛；不通过则自动停止。
- [ ] 实现后三折、完整 OOF、逐井 bootstrap 与 `comparison_vs_p02_full5.json`。

### Task 3：实现独立 oracle、运行实验并归档

**Files:**
- Modify: `rogii_clean/scripts/run_p2_p03_pf_seed_dispersion_cv.py`
- Modify: `rogii_clean/tests/test_p2_p03_cv_runner.py`
- Modify after result: `rogii_clean/experiments/current_state.json`
- Modify after result: `rogii_clean/experiments/registry.jsonl`
- Modify after result: `rogii_clean/experiments/phase2_roadmap.md`

**Interfaces:**
- Consumes: 完整五折 OOF 和仅用于诊断的隐藏真值。
- Produces: `oracle_seed_dispersion_audit.json` 与最终结论。

- [ ] 先写失败测试，确保 oracle 标记为 diagnostic-only 且不返回可合并的正式行级列。
- [ ] 实现固定分位桶和逐井关系诊断。
- [ ] 运行专项测试和相关回归测试。
- [ ] 运行 folds 0～1；按预注册门槛自动停止或继续。
- [ ] 若通过，运行负对照与完整五折；重新从预测文件复算 RMSE、行数、井数和唯一行键。
- [ ] 按事实、推断、未验证、当前只能否定、下一步更新实验登记和路线图。
