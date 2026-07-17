# AGENTS 规则改写与 RF01 稳定性审计实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把项目规则切换为“完成全部特征路线并优化固定单模 LightGBM CV”，同时独立完成 RF01a folds 2–4 稳定性诊断且不改变原 RF01 不晋级结论。

**Architecture:** 根目录 `AGENTS.md` 只负责长期研究规则；`RF01_stability_audit_v1` 使用独立实验卡和配置记录预注册理由。审计脚本直接读取已经生成的 RF01a 特征缓存和 folds 0–1 预测，只训练 folds 2–4，再与 B00 做固定五折比较。

**Tech Stack:** Markdown、Python 3、pandas、NumPy、LightGBM、Parquet、pytest。

## Global Constraints

- 始终固定一个 LightGBM，只研究特征。
- 不改变 LightGBM 参数、1734 棵树、目标、fold、评价行或评分代码。
- 原 RF01 八个版本继续判定“不晋级”，不能事后改判。
- 审计固定使用 RF01a，不允许查看 folds 2–4 后重新选择版本。
- 审计只新训练 folds 2–4，folds 0–1 复用原预测。
- 默认使用中文和普通说法；代码名、文件名、LightGBM、RMSE、CV 可以保留。
- 不执行 Git、worktree、提交、推送或 PR。

---

### Task 1: 精简重写项目协作规则

**Files:**
- Modify: `AGENTS.md`
- Reference: `docs/superpowers/specs/2026-07-16-agents-experiment-rules-design.md`
- Reference: `rogii_clean/experiments/feature_roadmap_v2.md`

**Interfaces:**
- Consumes: 已批准的规则设计和现有特征路线图。
- Produces: 无内部矛盾、可直接指导后续实验的根目录规则文件。

- [ ] **Step 1: 用实验优先规则替换旧的 PF 理解门槛**

  新文件必须按以下顺序包含完整章节：

  ```text
  0. 当前唯一目标
  1. 固定研究边界
  2. feature_roadmap 执行顺序
  3. 固定 CV 与晋级门槛
  4. 实验前后必须记录的内容
  5. 防止数据泄漏
  6. 缓存与复现
  7. 长任务运行规则
  8. 中文表达与代码可读性
  9. RF01 原结论与稳定性审计
  10. Git 与非实验操作
  11. 当前阶段完成标准
  ```

  第 0 节必须明确写出：

  ```text
  当前阶段的唯一目标是完成 rogii_clean/experiments/feature_roadmap_v2.md 中的全部特征实验，在固定单模 LightGBM、固定参数、固定 spatial-pad 五折下，寻找最低且可信的 micro RMSE。
  ```

  第 9 节必须原样表达两层结论：

  ```text
  原 RF01 八个版本继续按既定规则判定为“不晋级”，不得事后改判。
  RF01_stability_audit_v1 只固定检查 RF01a 的 folds 2–4，无论结果如何都不能追认 RF01 晋级。
  ```

- [ ] **Step 2: 删除所有与新目标冲突的旧规则**

  删除以下旧要求，而不是只在文件开头覆盖它们：

  ```text
  当前唯一目标是理解 PF
  PF 10.7 复现前禁止新实验
  八份 PF 导读文档是研究门槛
  每次实验都必须等待用户逐次允许
  当前阶段暂不允许新增特征实验
  ```

- [ ] **Step 3: 检查规则文件没有矛盾和待定项**

  Run:

  ```powershell
  rg -n "TBD|TODO|待定|PF 10\.7 复现.*禁止|不得主动开启新的大实验|当前阶段不是继续.*实验" AGENTS.md
  ```

  Expected: 没有匹配结果。

  Run:

  ```powershell
  rg -n "feature_roadmap_v2|单模 LightGBM|micro RMSE|RF01_stability_audit_v1|不得事后改判|中文" AGENTS.md
  ```

  Expected: 每个关键约束至少出现一次。

---

### Task 2: 登记 RF01 稳定性审计

**Files:**
- Create: `rogii_clean/experiments/RF01_stability_audit_v1_card.md`
- Create: `rogii_clean/configs/rf01_stability_audit_v1.json`
- Reference: `rogii_clean/configs/rf01a_huber_slopes_v1.json`
- Reference: `rogii_clean/artifacts/RF01a_huber_slopes_v1/`

**Interfaces:**
- Consumes: RF01a 五个 Huber slope 特征定义、已有 folds 0–1、固定 B00 和固定 fold 注册表。
- Produces: 审计脚本可读取的预注册配置。

- [ ] **Step 1: 写实验卡并锁死选择理由**

  实验卡必须包含：

  ```text
  实验编号：RF01_stability_audit_v1
  实验类型：诊断，不参与原 RF01 晋级
  固定对象：RF01a_huber_slopes_v1
  新运行 folds：2、3、4
  复用 folds：0、1
  禁止事项：不选择其他 RF01 版本，不调整窗口、特征、参数、fold、目标或评分
  审计问题：fold 1 是唯一异常折，还是 fold 0 才是异常折？
  ```

  选择理由必须固定为：公式最简单、fold1损失最小、不含高派生差值/曲率/无界外推、选择时未使用 folds 2–4。

- [ ] **Step 2: 创建机器可读配置**

  配置必须包含以下确定字段：

  ```json
  {
    "experiment_id": "RF01_stability_audit_v1",
    "experiment_type": "diagnostic_only",
    "source_experiment_id": "RF01a_huber_slopes_v1",
    "source_config": "configs/rf01a_huber_slopes_v1.json",
    "source_artifact_dir": "artifacts/RF01a_huber_slopes_v1",
    "b00_prediction_file": "artifacts/B00_simple_lgbm_v1/predictions.parquet",
    "reused_folds": [0, 1],
    "new_folds": [2, 3, 4],
    "original_rf01_decision": "not_promoted_and_unchanged",
    "allow_reselection": false,
    "allow_parameter_change": false
  }
  ```

- [ ] **Step 3: 校验配置与原 RF01a 完全一致**

  使用测试断言审计配置引用的 RF01a `feature_version` 为 `rf01_huber_slopes_17_v1`，模型配置仍为 `configs/lgbm_feature_baseline_v1.json`，固定注册表 SHA-256 仍为 `0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab`。

---

### Task 3: 实现只补 folds 2–4 的审计脚本

**Files:**
- Create: `rogii_clean/scripts/run_rf01_stability_audit.py`
- Create: `rogii_clean/tests/test_rf01_stability_audit.py`
- Read: `rogii_clean/scripts/run_simple_lgbm_cv.py`
- Read: `rogii_clean/src/metrics.py`

**Interfaces:**
- Consumes: `train_fold(feature_table, registry, fold_id, model_params, artifact_dir, fingerprint, feature_columns)`、RF01a 特征缓存、原 folds 0–1、B00 OOF。
- Produces: `artifacts/RF01_stability_audit_v1/` 下 folds 2–4、完整五折诊断、B00 配对比较和中文结论。

- [ ] **Step 1: 先写失败测试，锁死不能重跑 folds 0–1**

  测试必须断言：

  ```python
  assert REUSED_FOLDS == (0, 1)
  assert NEW_FOLDS == (2, 3, 4)
  assert set(REUSED_FOLDS).isdisjoint(NEW_FOLDS)
  ```

  并测试折稳定性分类：

  ```python
  assert classify_fold_pattern([-2.6, 0.1, -0.2, -0.3, -0.1]) == "fold1是唯一反向折"
  assert classify_fold_pattern([-2.6, 0.1, 0.2, 0.3, 0.1]) == "fold0是唯一正向折"
  assert classify_fold_pattern([-2.6, 0.1, -0.2, 0.3, -0.1]) == "其余折表现混合"
  ```

- [ ] **Step 2: 运行测试并确认缺少实现**

  Run:

  ```powershell
  & 'D:\anaconda\python.exe' -m pytest tests\test_rf01_stability_audit.py -q
  ```

  Expected: 因 `run_rf01_stability_audit` 或函数不存在而失败。

- [ ] **Step 3: 实现输入审计和新折训练**

  脚本必须：

  ```text
  1. 读取 rf01_stability_audit_v1.json。
  2. 读取 RF01a 原配置、固定模型参数和固定 fold 注册表。
  3. 读取已有 RF01a feature_cache.parquet，不重新选择或生成其他版本特征。
  4. 验证 3,783,989 行、773井、17个固定模型特征和五个固定 Huber slope 名称。
  5. 验证原 fold_0/fold_1 predictions.parquet 与 runtime.json 存在。
  6. 只循环 NEW_FOLDS=(2,3,4)，调用 train_fold。
  7. 新折写入 artifacts/RF01_stability_audit_v1/fold_2、fold_3、fold_4。
  8. 不写入或覆盖原 RF01a fold_0/fold_1。
  ```

  审计指纹必须包含：审计配置、RF01a配置、模型配置、fold注册表、RF01a feature cache 文件大小和修改时间、审计脚本内容哈希。

- [ ] **Step 4: 合并原 folds 0–1 与审计 folds 2–4**

  完整预测必须由以下固定来源组成：

  ```text
  fold0：artifacts/RF01a_huber_slopes_v1/fold_0/predictions.parquet
  fold1：artifacts/RF01a_huber_slopes_v1/fold_1/predictions.parquet
  fold2：artifacts/RF01_stability_audit_v1/fold_2/predictions.parquet
  fold3：artifacts/RF01_stability_audit_v1/fold_3/predictions.parquet
  fold4：artifacts/RF01_stability_audit_v1/fold_4/predictions.parquet
  ```

  合并后验证总行数为 3,783,989，`well_id,row_index` 不重复，预测均有限，并保存 `predictions.parquet`。

- [ ] **Step 5: 与 B00 做固定逐折比较**

  按 `well_id,fold,row_index` 对齐 B00，逐位验证 `target_tvt` 相同。保存：

  ```text
  metrics.json
  per_well.csv
  comparison_vs_B00/metrics.json
  comparison_vs_B00/per_well.csv
  comparison_vs_B00/predictions.parquet
  feature_list.json
  parameter_list.json
  runtime.json
  conclusion.md
  ```

  `conclusion.md` 必须先写“原 RF01 仍不晋级”，再回答折模式；不能使用“晋级”“追认通过”等措辞描述审计结果。

- [ ] **Step 6: 运行单元测试**

  Run:

  ```powershell
  & 'D:\anaconda\python.exe' -m pytest tests\test_rf01_stability_audit.py -q
  ```

  Expected: 全部通过。

---

### Task 4: 运行审计并做最终验证

**Files:**
- Create: `rogii_clean/artifacts/RF01_stability_audit_v1/`
- Modify: `rogii_clean/experiments/registry.jsonl`

**Interfaces:**
- Consumes: Task 2 配置、Task 3 脚本、RF01a 已有产物。
- Produces: 独立、可复核且不会改变原决策的完整稳定性审计。

- [ ] **Step 1: 运行 folds 2–4**

  Run:

  ```powershell
  & 'D:\anaconda\envs\fyj\python.exe' scripts\run_rf01_stability_audit.py --config configs\rf01_stability_audit_v1.json
  ```

  Expected: 日志只出现 fold 2、fold 3、fold 4 的训练，不训练 fold 0 或 fold 1。

- [ ] **Step 2: 验证产物和决策文字**

  Run:

  ```powershell
  rg -n "原 RF01 仍不晋级|fold 1 是唯一|fold 0 是唯一|其余折表现混合" artifacts\RF01_stability_audit_v1\conclusion.md
  ```

  Expected: 第一条固定结论存在，三个折模式中恰好一个出现。

  Run:

  ```powershell
  Get-ChildItem artifacts\RF01_stability_audit_v1 -Recurse | Select-Object FullName
  ```

  Expected: 只有新训练的 `fold_2`、`fold_3`、`fold_4` 子目录；folds 0–1 只在配置和运行说明中引用。

- [ ] **Step 3: 登记诊断结果**

  向 `experiments/registry.jsonl` 追加一条 UTF-8 JSON，字段至少包含：

  ```json
  {
    "experiment_id": "RF01_stability_audit_v1",
    "experiment_type": "diagnostic_only",
    "source_experiment_id": "RF01a_huber_slopes_v1",
    "new_folds": [2, 3, 4],
    "original_rf01_decision": "not_promoted_and_unchanged",
    "artifact_dir": "artifacts/RF01_stability_audit_v1"
  }
  ```

- [ ] **Step 4: 运行全套回归测试**

  Run:

  ```powershell
  $tempPath = 'H:\kaggle\716\.pytest_tmp_agents_rf01_audit_20260716_1'
  & 'D:\anaconda\python.exe' -m pytest tests --basetemp $tempPath -q
  ```

  Expected: 当前77项测试加新增审计测试全部通过。

- [ ] **Step 5: 最终报告**

  最终只报告：

  ```text
  AGENTS.md 已改成实验优先规则
  RF01 原结论仍为不晋级
  RF01a 五折逐折结果
  fold0/fold1 异常性判断
  审计产物路径
  全套测试结果
  ```

  不执行任何 Git 操作。
