# N01m 运行速度修复说明卡

修改目标：让源井空间信号审计在几分钟内完成，同时保持井对、距离、方位、重叠、权重和正式 top8 预测数值不变。

当前代码行为：每计算一对井，`compute_trajectory_geometry` 都为同一口 source 井重新建立一次 KDTree；526 口源井全配对时重复建树数万次。首次 smoke 运行 7 分钟仍未完成源井配对，因此已主动中断，未产生任何正式结果。

准备修改的文件：

- `src/p3_n01m_neighbor_mean_residual.py`（如需放纯计算帮助函数）；
- `scripts/run_p3_n01m_neighbor_mean_residual.py`；
- `tests/test_p3_n01m_neighbor_mean_residual.py`。

准备修改的函数：`SourceRecord`、源井记录构造、轨迹几何计算和 source-source 全配对审计。

为什么要修改：重复建立完全相同的空间索引是纯计算浪费，不提供新信息。

修改前的数据流：每个 target-source 配对 → 为 source 全轨迹重新建 KDTree → 查询最小距离。

修改后的数据流：每口 source 井读取后只建一次 KDTree并保存 → 所有配对复用该树；source-source 无向井对只查询一次对称最小距离，再分别计算两个方向的重叠和合格状态。

会影响哪些特征：不影响；仍使用完整隐藏 X/Y、距离、方位差、平行重叠和 Typewell 指纹。

会影响哪些模型：不影响；没有模型训练。

会影响哪些参数：不影响；所有几何门槛、权重、top8、加权中位数和 eta 均不变。

会影响哪些缓存：中断的 `_smoke_max_wells_3` 只有未完成的源井残差表，不属于可复用缓存；重跑会覆盖该 smoke 子目录中的同名文件。正式目录未生成。

预期结果：合成轨迹上新旧几何值逐项一致；源井全配对明显加速；正式预测不变。

可能风险：两个方向的重叠长度若错误共用，会改变配对是否合格。测试必须分别与旧函数的 `A→B`、`B→A` 输出比对。

如何回滚：恢复为逐配对调用 `compute_trajectory_geometry`；本次没有正式结果需要回滚。
