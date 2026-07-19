# P3-N01m 实验卡：邻井只预测整井平均残差

实验编号：`P3-N01m_neighbor_mean_residual_v1`

实验名称：用严格内层 OOF 邻井残差的加权中位数，修正 P2-P02 的一个整井常数偏差

唯一假设：空间上接近、方向相似且轨迹平行重叠的井，其 P2-P02 整井平均残差更相似；邻井只提供一个低频常数偏移，比迁移完整路径或四个控制点更稳。

为什么值得验证：常数偏移 oracle 可将本次 outer0 基础 RMSE 从 `10.3398` 降到约 `6.75`；旧邻井 control4 路线有微弱正信号，但复杂度更高。当前只预测一个标量，是最便宜的空间连续性审计。

与当前最佳基线唯一不同之处：不修改 P2-P02 模型、41 个特征、LightGBM 参数或逐行基础预测；每口验证井只加一个由邻井估计的常数 `m_hat`。

合法输入：

- outer0 的严格 P2-P02 无标签基础路径；
- outer-train folds 1–4 的四份严格 inner OOF P2-P02 路径和训练标签；
- 水平井隐藏段 X/Y 轨迹；
- Typewell 最后 500 ft 的 TVT/GR 指纹；
- 固定 fold、自然隐藏行和井号。

预测目标：

\[
m_j=\operatorname{mean}(TVT_j^{true}-TVT_j^{innerOOF})
\]

验证井输出：

\[
TVT_i^{N01m}=TVT_i^{P2P02}+\widehat m
\]

固定 fold：outer fold 0；源井为开发 folds 1–4 的 526 口井；验证井为 fold 0 的 131 口井；116 口影子井不参与。

固定评价行：outer0 自然隐藏段 651,881 行。

邻井门槛与权重：沿用旧 N01a 已冻结的无标签几何定义，不在 outer0 搜索。

```text
最大轨迹距离 2500 ft
最大方位差 45°
最小平行重叠 500 ft
最多 8 口邻井
距离衰减 1000 ft
重叠满权重 1000 ft
同 Typewell 尾段指纹权重 ×1.5
```

\[
w_j=\exp(-d_j/1000)\cos^2(\theta_j)
\min(overlap_j/1000,1)q_j
\]

按 `权重降序 → 距离升序 → well_id 升序` 取最多 8 口。先求 `weighted_median(m_j,w_j)`，再固定收缩。

本实验在查看 outer0 结果前冻结最简单的离散加权中位数定义：按残差从小到大排序，累计权重第一次达到或超过总权重 50% 时，直接取该口邻井的真实残差值。它不会在两口邻井之间插值，也不与旧 F07R 的 centered-CDF 插值版本进行事后比较。

\[
\eta=\frac{\sum w_j}{\sum w_j+1},\qquad
\widehat m=\eta\,weighted\_median(m_j)
\]

无邻居时 `m_hat=0`。第一版不加额外 0.5 倍缩放，不做裁剪，不预测斜率，不训练模型。

正对照：严格 outer0 P2-P02 基础路径，RMSE `10.3397936872`。

负对照：保持目标井、邻居编号、几何和权重完全不变；在每个 inner fold 内用固定种子 `20260719` 对源井 `m_j` 做无固定点置换。

全局对照：按源井隐藏行数加权的全局平均残差。

oracle 诊断：最后读取 outer0 真值，报告每井真实平均残差常数路径；oracle 不参与邻居、阈值、权重或收缩选择。

成功门槛：只有以下条件全部满足，才为 outer1 生成新嵌套资产。

1. 至少 50% 的 outer0 井有合法邻居；
2. N01m 路径相对基础路径改善至少 `0.10 ft`；
3. N01m 路径比折内错配标签路径至少好 `0.20 ft`；
4. N01m 对真实平均残差的井级 MAE 比错配标签低至少 10%；
5. N01m 的井级 MAE 低于全局常数；
6. 对 `|m_true|>=2 ft` 的井，方向正确率比错配标签高至少 10 个百分点；
7. 最近邻不超过 1000 ft 的切片中，真实邻井路径优于错配标签路径；
8. 逐井 bootstrap 的候选减基础 RMSE 区间上界小于 0，且收益不能只来自最强 5% 的井。bootstrap 固定按井重采样 2,000 次、种子 42；再按 `base_sse-candidate_sse` 从大到小删除最强 `ceil(131×5%)=7` 口井，剩余井的 pooled micro RMSE 仍必须优于基础路径。

停止条件：任一核心门槛失败即停止当前实现；不根据 outer0 调距离、角度、邻居数、Typewell 倍数、收缩或裁剪。

预计运行时间：复用五份已存在的严格 P2-P02 预测，只做井级几何和加权中位数，预计分钟级。

需要生成的文件：

```text
artifacts/P3_N01m_neighbor_mean_residual_v1/
├── config.json
├── source_mean_residuals.csv
├── source_pair_signal.csv
├── legal_neighbor_predictions.csv
├── metrics.json
├── per_well.csv
├── predictions.parquet
├── feature_list.json
├── parameter_list.json
├── runtime.json
└── conclusion.md
```

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定当前固定 `2500 ft / 45° / 500 ft / top8 / 固定权重 / 加权中位数 / kappa=1` 的平均残差邻井实现，不能宣布邻井或平均残差没有信息。

## 严格嵌套血缘

```text
开发 folds 1–4
    ↓ 四次 inner OOF P2-P02
526 口源井各自得到纯 inner OOF m_j
    ↓ 只用无标签几何和 Typewell 指纹选邻井
outer0 得到 m_hat，并先保存合法结果
    ↓ 最后才读取 outer0 target_tvt
651,881 行路径评分
```

禁止直接使用全局五折 OOF 残差、R01b 的 outer0 oracle 表、旧 N01a oracle 缓存或影子井。
