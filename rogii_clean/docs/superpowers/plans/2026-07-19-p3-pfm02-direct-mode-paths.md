# P3-PFM02 Direct Mode Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将三条合法 PF 模式 delta 路径作为唯一新增特征，运行冻结的 44 列单模 LightGBM CV。

**Architecture:** 新建一个不读取标签的逐井模式缓存生成器，再新建一个复用 P3-PF02 开发集训练骨架的专用 CV runner。旧代码和旧 artifact 全部只读。

**Tech Stack:** Python、NumPy、pandas、PyArrow、SciPy Ward、LightGBM、pytest。

## Global Constraints

- 正式特征严格为 P3B00 原 41 列加 low/middle/high 三列，共 44 列。
- LightGBM 固定 1,734 棵树、seed 29 和原完整参数，不 early stopping、逐行等权。
- 只使用 657 口开发井，116 口影子井目标不得进入 pandas。
- 先 folds 0～1，按实验卡门槛决定是否运行 folds 2～4。
- 不提交、不执行 Git；由用户自行处理 Git。

---

### Task 1: 无标签三模式合法缓存

**Files:**
- Create: `rogii_clean/scripts/generate_p3_pfm02_mode_path_cache.py`
- Test: `rogii_clean/tests/test_generate_p3_pfm02_mode_path_cache.py`

**Interfaces:**
- Consumes: 每井共享 NPZ 的 `seed_delta/final_ll/seed_ids/row_index/last_tvt`。
- Produces: `legal_cache/<well>.parquet`，列严格为井号、fold、自然行号、anchor、三条 delta 和指纹；以及 `legal_runtime/<well>.json`。

- [ ] 写失败测试：三模式只依赖 PF128，禁止 target/oracle 字段，固定 schema/指纹，并拒绝缺井、错行、影子井和非 128 seed。
- [ ] 运行 `python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q`，确认因模块不存在而失败。
- [ ] 最小实现：复用 `src.p3_pfm01_ordered_pf_modes` 的二维 Ward、scale8 加权和位置排序，只保存三条路径。
- [ ] 重跑测试，要求全部通过。

### Task 2: 44 列冻结 LightGBM runner

**Files:**
- Create: `rogii_clean/configs/p3_pfm02_direct_mode_paths_v1.json`
- Create: `rogii_clean/scripts/run_p3_pfm02_direct_mode_paths_cv.py`
- Test: `rogii_clean/tests/test_run_p3_pfm02_direct_mode_paths_cv.py`

**Interfaces:**
- Consumes: 原 B00/F05a/P2-P01 缓存、657 口三模式合法缓存、冻结 fold/影子表和 P3B00 OOF。
- Produces: 44 列 fold 模型、OOF、指标、逐井/逐折结果、重要性和结论。

- [ ] 写失败测试：特征顺序必须为 41+3，任何摘要列/参数变化/影子目标/错键/错 anchor 都失败。
- [ ] 运行专项测试，确认因 runner 不存在而失败。
- [ ] 以 `run_p3_pf02_target_ess_lgbm_cv.py` 为骨架实现最小专用 runner，只替换实验编号、三列 schema 和 44 列合同。
- [ ] 重跑专项测试和被复用的 PF02 测试，要求全部通过。

### Task 3: 缓存补齐与小样本验证

**Files:**
- Runtime output: `rogii_clean/artifacts/P3_shared_pf_seed_paths_v1/`
- Runtime output: `rogii_clean/artifacts/P3_PFM02_mode_paths_v1/`

- [ ] 用 PF03 `--generation-only-fold 1/2/3/4` 补齐共享 PF128；逐井原子保存，可重复同命令续跑。
- [ ] 运行三井模式缓存 smoke，检查三列有限、row_index/anchor 对齐且无标签。
- [ ] 运行全部 657 口模式缓存，并验证 3,211,872 行、统一指纹、零影子井。

### Task 4: 正式 folds 0～1 与停止门槛

**Files:**
- Runtime output: `rogii_clean/artifacts/P3_PFM02_direct_mode_paths_v1/`
- Modify: `rogii_clean/experiments/registry.jsonl`（只追加）

- [ ] 运行 `python scripts/run_p3_pfm02_direct_mode_paths_cv.py --folds 0,1`。
- [ ] 独立复算 paired micro RMSE、每折改善、井级指标和 bootstrap。
- [ ] 若 folds 0～1 未达到 `0.20 ft` 且最差折不低于 `-0.10 ft`，登记停止；否则运行 `--folds all`。
- [ ] 完整五折时按实验卡全部门槛判定，并追加 registry 记录，不事后更改特征、参数或门槛。
