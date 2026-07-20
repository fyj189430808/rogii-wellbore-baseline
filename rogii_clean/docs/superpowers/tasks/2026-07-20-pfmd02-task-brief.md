# P3-PFM-D02 实现任务简表

## 目标

新增 `scripts/run_p3_pfmd02_representation_oracle_ladder.py` 和专项测试，一次流式完成 `docs/ROGII 3阶段实验执行指令.md` 中 P3-PFM-D02 的 A–F 全部只读 oracle。不要训练模型，不要改旧文件，不做 Git。

## 冻结输入

- fold：`artifacts/folds/balanced_well_5fold_v1.csv`
- 影子井：`artifacts/P3_shadow_holdout_v1/shadow_holdout.csv`
- P2 与真值：`artifacts/P2_P02_multiscale_pf_paths_v1/predictions.parquet`
- 三模式：`artifacts/P3_PFM02_mode_paths_v1/legal_cache/<well_id>.parquet`
- 128 seed：`artifacts/P3_shared_pf_seed_paths_v1/<well_id>.npz`
- 正式范围必须恰为 657 井、3,211,872 行、影子井交集为 0。P2 parquet 必须先在 Arrow 层按开发井过滤，再转 pandas。
- 自然键是 `(well_id,row_index)`；逐井要求 P2、mode、seed 的键、fold、行数逐位一致；MD 与 `hidden_md` 在 float32 舍入容差内一致。
- P2 冻结源全量旧分数是 10.30570499；本诊断过滤影子后的 657 井 P2 分数必须复算为 10.272146267501086。

## 固定模式定义

每条 seed 用 `[整段 mean_delta, endpoint_delta]`，不标准化，Ward K=3。每个固定簇内部用 scale=8 的稳定 softmax 权重求完整中心；按中心整段均值、末端、最小 seed id 排为 low/middle/high。必须断言重新计算中心与 PFM02 legal cache 在数值容差内一致。极端跨模式 LL 不得下溢。

所有 delta 加 `last_visible_tvt` 后才能与绝对 TVT 真值比较。所有 oracle 输出只能落到新目录，不得成为 legal feature/cache。

## A：整井硬选

- A4：每井从 `P2, low, middle, high` 选整条 SSE 最小路径，平局按此顺序。
- A3：仅从 `low, middle, high` 选，作为 D/E 的公平基准。

## B：整井连续凸混合

- B0：四候选非负权重、和为 1 的全局最小 SSE。
- 用 15 个非空 active subset 的 KKT 闭式枚举求精确 simplex 最小二乘；不得用 OLS 后 clip。
- B25：强收缩固定为 `q_P2 >= 0.75`。写成 `q=0.75*e_P2+0.25*v`，其中 v 仍在四维 simplex 上精确求解。不可在 B0/B25 间事后取最优。
- 保存每井权重、active 数和 SSE。

## C：分段选择

- 按隐藏首个 MD 起点做不重叠 250/500/1000 ft 段，不按行数分段。
- 每段计算四候选 SSE；保存独立 argmin 路径。
- 另做四状态 DP。切换惩罚固定为“完整窗口每行 1 ft²”：`lambda_L = max(1, round(L / median_positive_md_step))`，不扫描参数。DP 目标中的 penalty 与最终原始 SSE 分开记录。
- 平局优先保持原状态，其次按 `P2,low,middle,high` 顺序。
- 保存逐段四个 SSE、独立/DP state、margin；保存每井各窗口独立 SSE、DP 原始 SSE、penalized objective、switch count。

## D：128 seed 整井硬选

- D128 严格只从 128 条原始 seed 整井路径选一条；平局最小 seed id。
- 另报 D128+P2 辅助值，但不得和 D128 事后取最优或改写 D128。
- 主要公平比较为 D128 对 A3。

## E：固定成员的三种代表

- center3：现有簇内 scale8 LL 加权中心，必须等于 A3。
- rowmedian3：每个固定簇逐行未加权中位路径。
- medoid3：每个固定簇从真实成员 seed 中，选择到簇内所有 seed 的 LL 加权全路径均方距离最小者；平局最小 seed id。这里不得用隐藏真值定义 medoid。
- 三种代表各自只做三代表整井硬选，不允许在代表方法之间再看真值选择。
- 保存三模式 medoid seed id。

## F：逐行包络

- F-mode3：每一行从 low/middle/high 选误差最小值。
- F-seed128：每一行从 128 seed 选误差最小值。
- 明确标为不可部署的 pointwise coverage，不得驱动路线选择。

## 指标和决策

- 每法每井保存 SSE/RMSE；pooled micro=`sqrt(sum SSE / sum rows)`；另报 macro、median、P90、最差井和逐折 micro。
- 固定比较：B0/B25/C 对 A4；D128 对 A3；median3/medoid3 对 center3；F 只展示。
- 连续混合信号：B0 比 A4 改善至少 0.30 ft，且 B25 同方向。
- 分段信号：主判断 C250-DP 比 A4 改善至少 0.30 ft；同时列出 C250-independent 与 500/1000 结果。
- 形状聚类信号：D128 比 A3 改善至少 0.30 ft。
- medoid 只报告实际改善值和 `>0`、`>=0.10` 两个事实，不擅自改变后续路线。

## 最小产物

正式目录：`artifacts/P3_PFM_D02_representation_oracle_ladder_v1/`

- `config.json`, `input_audit.json`, `metrics.json`
- `per_well.csv`, `per_fold.csv`, `per_segment.csv`
- `runtime.json`, `conclusion.md`
- `smoke_3/` 与正式目录物理隔离

逐井处理，不能构造全局 `128 × 321万` 数组。每井或每批打印进度。先允许 `--max-wells 1|2|3`，再正式全量。正式结果必须支持读现有 per-well checkpoint 续跑，避免中断重算。

## 测试要求

至少覆盖：simplex 解已知例子、B25 的 P2 权重下界、MD 分段、DP 惩罚与回溯、medoid 必为原 seed、包络 SSE、micro 聚合、自然键错位拒绝、shadow 拒绝、smoke 与正式目录隔离、极端 LL 稳定中心。先得到失败测试，再实现；运行专项测试和 `py_compile`。
