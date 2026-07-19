# P2-S03 实验卡

实验编号：`P2_S03_nearest_dense_surface_v1`

实验名称：严格外折最近 dense 轨迹点 surface 路径

唯一假设：不拟合局部平面，每个目标控制点直接使用最近 outer-train 轨迹点的 EGFDU，并用目标可见前缀校准常数，可以保留真实空间 surface 信号同时避免无界斜率灾难。

为什么值得验证：P2-S01 最坏井在改用 source 轨迹局部点后，多口 RMSE 从 65～312 ft 降到约 6～11 ft；P2-S02 的结果提示跨井局部平面可能仍是主要问题，因此最便宜的下一步是完全删除平面。

与当前最佳基线唯一不同之处：不训练模型，只新增一条最近轨迹点确定性路径；复用冻结的 95,995 个 50 ft dense source 点，P2B00 不变。

合法输入：outer-train `X/Y/EGFDU` dense 点；目标 `MD/X/Y/Z/TVT_input`；固定按井折表。

预测目标：自然隐藏段 TVT；合法函数不接收目标 TVT/surface。

固定 fold：`balanced_well_5fold_v1`。

固定评价行：3,783,989 个自然隐藏行。

正对照：目标井自身 EGFDU 公式上限。

负对照：每口 source 井保持位置和 surface 中位值，但反转井内 surface 变化方向，再执行同一最近点查询。

成功门槛：自身 surface 正对照不超过 0.60 ft；主路径相对 carry 至少改善 0.50 ft；至少 4/5 折优于 carry；相对井内梯度反转负对照至少改善 1.00 ft。

停止条件：dense cache 指纹不一致；验证折混入 source；合法函数读取目标 TVT/surface；三井 smoke 行键或公式失败。

预计运行时间：3 井 smoke 少于 1 分钟，全量约 1～3 分钟；单井缓存可续跑。

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定“50 ft dense source、50 ft 目标控制点、逐控制点单一最近 source 点、全可见前缀中位偏移校准”。
