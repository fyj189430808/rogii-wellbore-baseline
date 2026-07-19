# P2-CV00：按井五折重跑 C01 设计说明

## 目标

把一阶段最佳可复现方案 C01（`F05a_deterministic_candidates_v1`）的 36 个特征、目标、LightGBM 参数和预测还原方式全部保持不变，只把验证协议改成确定性的按井平衡五折，得到二阶段唯一基线 `P2_CV00_group5_c01_v1`。

一阶段 `12.930691493610633` 只作为旧协议历史成绩，不与二阶段实验直接比较。

## 唯一变量

- 旧协议：`spatial_pad_1000_v1`，空间连通 pad 不能跨折。
- 新协议：`balanced_well_5fold_v1`，每口完整井是独立分组，按隐藏评价行数做确定性五折平衡。

以下内容不变：

- 773 口井；
- 3,783,989 个 `TVT_input` 为空的评价行；
- 12 个 B00 基础特征；
- 24 个 C01 确定性候选特征；
- 目标 `TVT - last_visible_TVT`；
- 单个 `lightgbm.LGBMRegressor`；
- 1,734 棵树及全部固定参数；
- `last_visible_TVT + predicted_delta` 的还原方式；
- uniform row weight，无额外后处理。

## 按井五折算法

1. 从每个训练井读取 `well_id`、隐藏评价行数和现有井级统计。
2. 令兼容字段 `pad_id = well_id`，表示每口井都是不可拆分的最小验证单位；它不代表空间 pad。
3. 按 `hidden_rows` 从大到小、`well_id` 从小到大排序。
4. 依次把井放入当前 `(隐藏行总数, 井数, fold编号)` 最小的 fold。
5. 不使用随机数，因此同一数据必然产生同一注册表。

预期五折规模：

| fold | 井数 | 隐藏评价行 |
|---:|---:|---:|
| 0 | 155 | 757,738 |
| 1 | 155 | 756,650 |
| 2 | 154 | 756,255 |
| 3 | 155 | 757,101 |
| 4 | 154 | 756,245 |

## 缓存复用与折号重映射

C01 两份数值缓存是逐井、逐行的合法确定性输入，不依赖 outer-fold 训练，因此不重新计算：

- `artifacts/B00_simple_lgbm_v1/feature_cache.parquet`
- `artifacts/F05a_deterministic_candidate_cache_v1/candidate_feature_cache.parquet`

B00 缓存中的旧 `fold` 列不可信。新 runner 必须按 `well_id` 用 `balanced_well_5fold_v1.csv` 覆盖它，并验证：

- 每个缓存井恰好映射到一个新 fold；
- 没有缺失或多余井；
- 每折缓存行数等于注册表中该折 `hidden_rows` 之和；
- 全部 3,783,989 行只出现一次；
- 一阶段 artifact 只读且不被覆盖。

## 文件边界

- 扩展 `rogii_clean/src/fold_split.py`：增加按井分折纯函数，保留旧空间函数原样。
- 新增 `rogii_clean/scripts/make_balanced_well_folds.py`：只生成二阶段注册表和元数据。
- 新增 `rogii_clean/scripts/run_p2_cv00_group5_c01.py`：加载原 C01 缓存、覆盖折号、运行 smoke/单折/完整五折。
- 新增 `rogii_clean/configs/p2_cv00_group5_c01_v1.json`：冻结全部输入和指纹。
- 新增 `rogii_clean/tests/test_p2_cv00_group5_c01.py`：覆盖分折与重映射风险。
- 输出 `rogii_clean/artifacts/P2_CV00_group5_c01_v1/`，不写一阶段目录。

## 完成标准

1. 原空间分折测试仍通过。
2. 新分折对输入顺序不敏感，且 `pad_id == well_id`。
3. 新注册表覆盖 773 井和 3,783,989 行，五折规模与上表一致。
4. smoke 验证缓存键、36 列有限值、折号重映射和每折行数。
5. 完整五折均完成，生成 `predictions.parquet`、`per_well.csv`、`metrics.json`、特征/参数/运行时间和结论文档。
6. 二阶段 `current_state.json` 只在完整五折成功后更新。

