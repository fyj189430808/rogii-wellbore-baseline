# P2-D01：PF 中心多尺度 GR 可辨识性审计

## 要回答的问题

不训练新模型，先验证：以冻结 `pf_ancc` 路径为中心，在 `[-40, 40] ft` 搜索 Typewell GR 时，正确 offset 是否比错误 offset 得分更高；峰差、尺度一致性和 GR 支撑率能否在不知道真值时识别可靠井段。

这一步落实 `docs/一阶段总结.md` 的判断：旧 F03b 失败不能否定 GR 对齐，因为它以恒定 U 为中心；真正需要的是 PF 中心的“井深 × offset”得分面和连续路径。

## 三种方案比较

1. 继续复用旧 F03b 的逐行 10 ft 平滑 + 101 ft NCC：最快，但会重复恒定 U 中心和逐行找峰问题，不采用。
2. PF 中心、固定井段、多尺度得分面：能先把“评分有无信号”和“路径优化好不好”分开，成本可控，采用。
3. 直接实现动态规划/前后向平滑：离最终方案最近，但失败时无法归因，留给后续 P2-F01。

## 冻结设计

- 中心路径：`last_visible_tvt + pf_ancc_delta`。
- offset 网格：`-40` 到 `40 ft`，步长 `2 ft`，共 41 个位置。
- 固定井段宽度：`50 ft` 和 `100 ft`，两者都报告，不事后挑一个改门槛。
- GR 平滑尺度：`5/11/21/51/101 ft`。
- 每个井段每个 offset：比较水平井平滑 GR 与按 `PF中心+offset` 从 Typewell 插值得到的 GR，使用 Pearson NCC。
- 单尺度得分与五尺度等权平均得分全部保存。
- 每段至少 30 个公共有效 GR 点；不足时保留缺失和支撑率，不补造得分。
- 第二峰必须距第一峰至少 `6 ft`。

## 三条证据线

1. 可见前缀正对照：中心使用已知 `TVT_input`，正确 offset 固定为 0，验证评分公式确实能找回已知层位。
2. 隐藏段 oracle：正式评分函数不接收 TVT；评分完成后，独立读取隐藏真实 TVT，计算 `true_offset = TVT - PF中心`、正确 offset 排名和误差。
3. 井内循环平移负对照：把隐藏段水平井 GR 固定循环平移半段后重算，保留自相关和缺失结构但破坏真实层位对应。

## 输出

- `block_diagnostics.parquet`：真实评分的逐井段多尺度结果及 oracle 标签。
- `negative_block_diagnostics.parquet`：循环平移负对照。
- `prefix_positive_control.parquet`：可见前缀零偏移正对照。
- `per_well.csv`：井级覆盖、排名、峰差、尺度离散和当前 P2B00 误差。
- `summary.json`：总体、分 block width、分 fold、分合法置信度的汇总。
- `config.json`、`runtime.json`、`lineage.json`、`conclusion.md`。

## 进入后续连续路径实现的预设证据

满足以下条件，说明 GR scorer 值得进入后续 P2-F01；正式建模顺序仍先执行 `P2-S01`：

- PF 中心真实 offset 落在 `[-40,40]` 的井段比例至少 90%；
- 可见前缀正对照的 offset=0 top-5 命中率至少 50%；
- 隐藏段多尺度平均分的真实 offset top-5 命中率比循环平移负对照高至少 10 个百分点；
- 隐藏段最佳 offset 的绝对误差中位数比负对照至少低 3 ft；
- 测试期合法的高置信块（峰差较大、尺度 offset 较一致）比低置信块有明确更低的 oracle offset 误差。

若未达到，只否定这组固定分块、NCC 和尺度组合，不否定整段 GR 信息。

## 泄漏隔离

正式 scorer 的输入只包含 `MD/GR/TVT_input`、Typewell `TVT/GR` 和冻结 PF cache。隐藏真实 `TVT` 由单独的 oracle 函数在评分完成后附加，任何 threshold、score 或 offset 选择不得读取它。

