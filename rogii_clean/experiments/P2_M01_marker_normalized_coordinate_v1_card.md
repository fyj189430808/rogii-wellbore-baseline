# P2-M01 实验卡

实验编号：`P2_M01_marker_normalized_coordinate_v1`

实验名称：外折模板 marker 转移与跨区间别名排名审计

唯一假设：F01b 在绝对 TVT 中选中的错误高峰主要来自其他 marker 区间的重复 GR 形状；用测试期合法获得的六个 marker 标明地层区间后，PF 所在区间能够排除大量错误别名，并让真实 offset 的得分排名显著提前。

为什么值得验证：F01b 的真实 offset 中位排名只有 `18/41`，但真实 GR 明显优于循环打乱 GR，说明信号存在但层位分支错误。《一阶段总结》把 marker 归一化列为从 8～9 继续突破的核心表示。

与当前最佳基线唯一不同之处：本实验不训练模型、不生成新路径，也不修改 F01b 的 GR 得分。只在 F01b 已冻结的 41 个候选上增加由 outer-train Typewell 合法转移的 marker 区间，并审计候选排名。

合法输入：

- 当前 fold 全部 Typewell 的 `TVT/GR`，只用于模板指纹；
- outer-train Typewell 的 `Geology`，只用于 donor marker；
- 验证井冻结 PF 中心、F01b 合法 score/pair count 和 block 对齐；
- 固定按井 fold 注册表。

禁止输入：

- 验证井自己的 `Geology` 进入 marker 构造；
- 验证井 surface；
- 隐藏 TVT 进入模板匹配、marker 转移、候选门控或参数选择；
- 按相同 well_id 从训练副本搬测试 marker。

预测目标：本阶段没有 TVT 路径预测，只输出每个 block 的原始排名、oracle 区间排名、合法 PF 区间排名和负对照排名。

固定 fold：`balanced_well_5fold_v1` 的 fold 0，155 口井、757,738 个隐藏行、15,236 个 F01b 控制块。

固定评价块：F01b fold 0 中 score 缓存与隐藏行键完全一致的全部 block；任何覆盖筛选都必须报告分母和回退后的全体指标。

模板指纹：清洗并按 TVT 排序后，取最后 500 行 `(TVT, GR)` 的 float64 精确字节 SHA-256。该规则在结果出现前已经复现出 54 个模板；不做伸缩、模糊匹配或按 well_id 匹配。

marker 定义：固定六个顺序稳定的标签：

```text
ANCC → ASTNU → ASTNL → EGFDU → EGFDL → BUDA
```

每个验证井只使用 fold 1～4 中同指纹 donor。每个 donor 取 Geology 首次标签 TVT；文件首行开始的左截断 marker 不算真实边界。逐 marker 取 donor 中位数；donor 极差超过 `1 ft` 时记为 ambiguous 并禁用该 marker。

归一化坐标：相邻 marker `m_k <= t < m_(k+1)` 内：

```text
q(t) = k + (t - m_k) / (m_(k+1) - m_k)
t(q) = m_k + (q - k) * (m_(k+1) - m_k)
```

首版只使用 ANCC 到 BUDA 之间有双侧 marker 包围的位置，不向外无界外推。

排名定义：

- `R_abs`：真实 offset 最近网格在全部 41 个候选中的原始 emission 排名；
- `R_oracle_zone`：隐藏真值只用于指出真实 marker 区间，再在同区间候选中排名；
- `R_legal_pf_zone`：只保留 PF 中心所在 marker 区间的候选；若真实候选被排除，排名百分位直接记为 1；
- 主要指标为 `(rank - 1) / (candidate_count - 1)`，单候选且命中记 0，避免候选变少后 top-5 虚高；
- 同时报告 top-1/top-3/top-5、候选数、`q_best-q_true` 和按井聚类 bootstrap。

正对照：

1. 人工裁剪一条训练 Typewell 并删除 Geology，使用同模板异井 donor 恢复 marker，误差不超过 `0.5 ft`；
2. 合法转移 marker 全部落盘后，再用验证井自身 Geology 做 withheld oracle 核对；
3. 程序级 `q → TVT → q` 往返误差不超过 `1e-10`。

负对照：

1. 错 marker 厚度：在 outer-train 的其他模板中选择总厚度最近者，固定 ANCC 锚点，只替换五段厚度；
2. 等候选数随机连续门控：每个 block 用 `SHA256(well_id, block, seed=29)` 固定选择与合法 PF 区间相同数量的连续 raw-offset 候选；真实候选被排除同样记最差。

oracle 诊断：隐藏 TVT 只在合法 marker 表与所有门控定义落盘后读取，用于真实区间、真实 offset 和排名评分；不得写回 legal marker 缓存。

成功门槛：必须全部满足才实现 q-state 连续路径：

- fold 0 验证井合法同模板 donor 覆盖率至少 `90%`；
- 可审计 block 覆盖率至少 `90%`；
- withheld marker 绝对误差 P90 不超过 `0.5 ft`；
- ambiguous 井比例不超过 `5%`；
- 原始错误 top-1 中，跨真实 marker 区间比例至少 `50%`；
- oracle-zone 相对原始平均 normalized rank 至少改善 `0.10`，按井 bootstrap 95% CI 上界小于 0，top-5 至少增加 10 个百分点；
- PF 中心区间与真实区间一致率至少 `80%`；
- legal-PF-zone 相对原始平均 normalized rank 至少改善 `0.05`，bootstrap 95% CI 上界小于 0；
- legal-PF-zone 相对两个负对照的 normalized rank 均至少改善 `0.03`。

停止条件：指纹或行键不一致；marker legal/oracle 阶段混用；程序正对照失败；任一上述门槛失败。

预计运行时间：程序测试与 3 井 smoke 少于 1 分钟；fold 0 只读现有 score，预计 1～3 分钟。

需要生成的文件：

```text
artifacts/P2_M01_marker_normalized_coordinate_v1/
├── config.json
├── lineage.json
├── legal_markers.csv
├── withheld_marker_oracle.csv
├── block_ranks.parquet
├── per_well.csv
├── summary.json
├── runtime.json
└── conclusion.md
```

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：

- 指纹/marker 恢复失败，只否定精确重复模板转移；
- marker 准但 oracle-zone 不改善，只否定六 marker 区间能解决当前 F01b Pearson 别名；
- oracle-zone 改善但 legal-PF-zone 不改善，说明 marker 有信息但 PF 不能合法确定区间；
- legal 排名也改善，才允许实现状态在 q 上移动的连续路径。

