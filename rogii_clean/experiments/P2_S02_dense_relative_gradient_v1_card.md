# P2-S02 实验卡

实验编号：`P2_S02_dense_relative_gradient_v1`

实验名称：严格外折 dense EGFDU 相对梯度路径

唯一假设：保留 outer-train 井内部沿轨迹的 EGFDU 变化，只聚合局部二维梯度并从目标井最后可见结构位置积分，可以避免 P2-S01 的绝对高度和无界斜率灾难，并生成稳定优于 carry 的相对结构路径。

为什么值得验证：P2-S01 最坏井把真实约 100～200 ft 的 surface 变化预测成 500～800 ft，根因是每井一个中位点丢失井内倾角；把 source 改成沿轨迹的局部 surface 点后，多口灾难井到真实空间支撑的距离缩短 2～4 倍。

与当前最佳基线唯一不同之处：本阶段不训练模型，只把 S01 的“一井一个绝对点局部平面”改成“每井多局部点、只积分相对梯度”的确定性路径；P2B00 不变。

合法输入：outer-train 井 `X/Y/EGFDU`；目标井 `MD/X/Y/Z/TVT_input`；固定按井折表。

预测目标：自然隐藏段 TVT；合法路径接口不接收目标 TVT 或 surface。

固定 fold：`balanced_well_5fold_v1`。

固定评价行：`TVT_input.isna()`，3,783,989 行、773 口井。

正对照：目标井自身 EGFDU 公式上限，仅在合法路径完成后评分。

负对照：每口 outer-train 井保持 XY、surface 中位值和变化幅度不变，但把井内 `surface - median(surface)` 乘以 -1，反转倾角方向。

oracle 诊断：隐藏 TVT 只用于路径完成后的 RMSE、逐井与支撑切片。

成功门槛：自身 surface 正对照不超过 0.60 ft；主路径相对 carry 至少改善 0.50 ft；至少 4/5 折优于 carry；主路径相对井内梯度反转负对照至少改善 1.00 ft。

停止条件：source fold 排除失败；outer-train 梯度上限无法计算；合法函数读取目标 TVT/surface；三井 smoke 公式或行键失败。

预计运行时间：首次构建 dense source 约 1～3 分钟；3 井 smoke 少于 1 分钟；全量约 3～10 分钟。逐井缓存支持续跑。

需要生成的文件：配置、dense source 点、每折梯度上限与 source lineage、逐井合法路径、负对照、逐井指标、支撑切片、summary、runtime 和结论。

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定“EGFDU、50 ft source/target 控制点、10 个不同邻井局部梯度、outer-train P99 梯度裁剪、梯形积分”这一实现。

