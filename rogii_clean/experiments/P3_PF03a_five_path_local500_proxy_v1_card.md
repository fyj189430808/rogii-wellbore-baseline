# P3-PF03a 实验卡：五条冻结路径的 500 ft 局部混合代理

- 实验编号：`P3_PF03a_five_path_local500_proxy_v1`
- 实验性质：**零 PF 重跑的方向筛查，不是正式 128-seed PF03。**
- 唯一假设：整井统一路径权重掩盖了沿井深变化的 GR 证据；即使只在冻结的五条聚合路径之间做局部混合，500 ft 动态权重也可能优于整井固定路径。
- 唯一变化：把冻结的 `mean/scale3/scale5/scale8/scale12` 五条绝对路径按隐藏段原始有效 GR 做 500 ft 局部加权。PF 状态转移、五条输入路径、旧 `gr_sigma`、CV 和评价行不变。
- 局部评分：只使用每段自然隐藏行中原始非缺失 GR；候选 TVT 映射到 Typewell GR 后，使用冻结旧 `gr_sigma` 计算高斯残差总分。
- 权重规则：每段固定目标 ESS=`4/5`，目标值在看标签前写死；相邻段中心之间线性插值五个权重。
- 回退规则：一段有效 GR 少于 50 行，或五条路径 GR 分数完全不可区分时，直接回退冻结 `pf128_mean_delta`。
- 合法输入：当前井 MD、原始 GR、`TVT_input` 缺失掩码、Typewell TVT/GR、冻结旧 `gr_sigma`、五条冻结 PF 聚合路径。
- 预测目标：不训练模型，只生成一条 `pf128_local500_delta` 并做路径自身 RMSE 方向筛查。
- 固定 CV：`balanced_well_5fold_v1` 的 657 口开发井；116 口影子井排除且不开标签。
- 正对照：同一评价行上的冻结 `mean/scale3/scale5/scale8/scale12` 五条路径。
- folds 0～1 方向门槛：相对合并后最佳旧固定路径改善至少 `0.20 ft`，任一折恶化不超过 `0.10 ft`。
- 停止条件：未过门槛则不跑 folds 2～4，不进入 LightGBM。
- 预计耗时：不重跑 PF；smoke 数秒，folds 0～1 主要耗时为 263 口井 CSV/Parquet 读取和局部插值。
- 输出目录：`artifacts/P3_PF03a_five_path_local500_proxy_v1/`。
- 失败只能否定：**五条已经聚合的冻结路径 + 500 ft 分段 + ESS=4 + 当前 GR 分数**这一快速代理。
- 失败不能否定：正式 128 条 seed 路径上的分段似然、动态 ESS、不同分段长度或更稳健的 GR 观测模型。
