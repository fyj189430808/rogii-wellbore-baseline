# P3-N01a 实验卡：邻井四控制点残差迁移

- 实验编号：`P3_N01a_pf_scale8_neighbor_residual_audit_v1`
- 唯一假设：几何相近的 outer-train 邻井，其 `真实 TVT - (末值 + pf128_scale8_delta)` 四控制点残差曲线可迁移到验证井。
- 为什么值得验证：D00 已证明残差主要是低维平滑曲线；本实验只检查邻井几何能否提供这种曲线，不训练模型。
- 基线：原始 `last_visible_tvt + pf128_scale_8_delta` 整段路径。
- 合法来源：657 口开发井；每个 outer fold 只允许其他 fold 的 source 标签。影子 116 口在读取原始井文件前物理排除。
- 固定几何：完整隐藏段 XY 轨迹，最近距离不超过 2500 ft，无向方位差不超过 45 度，平行投影重叠不少于 500 ft，最多取权重最高的 8 口井。
- 固定权重：`exp(-距离/1000) × cos²(方位差) × min(重叠/1000,1)`；Typewell 尾 500 ft 指纹相同再乘 1.5。
- 固定收缩：`eta = sum_weight / (sum_weight + 1)`。
- 方向处理：source 与 target 方向相反时倒序四个控制点。
- 负对照：固定随机种子打乱 outer-train source 的残差 profile，但保持邻井几何和权重不变。
- 正对照：用验证井自身真值拟合 `control4_unanchored`，仅报告 oracle，不参与晋级。
- 固定筛查：开发集 folds 0～1，共 263 口井；不训练 LightGBM。
- 晋级门槛：
  1. 有效邻井支持覆盖率至少 50%；
  2. 真实邻井修正 pooled RMSE 比固定 seed 打乱 profile 至少好 0.20 ft，fold 0 和 fold 1 同方向；
  3. 真实邻井修正比原 scale8 pooled 至少改善 0.10 ft，任一折不得恶化超过 0.25 ft。
- 停止条件：任一门槛失败即停止当前实现，不跑 folds 2～4，不训练 LightGBM。
- 预计运行时间：smoke 少于 1 分钟；folds 0～1 约 2～8 分钟。逐井缓存支持中断续跑。
- 失败只能否定：当前固定几何、四控制点表示、固定权重和固定收缩的直接迁移实现，不能否定所有邻井信息。

