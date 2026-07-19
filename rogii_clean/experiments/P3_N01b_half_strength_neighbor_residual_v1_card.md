# P3-N01b 实验卡：半强度邻井残差修正

- 实验编号：`P3_N01b_half_strength_neighbor_residual_v1`
- 唯一假设：N01a 的真实邻井残差方向有信号，但固定 `eta` 整段修正过强；统一乘以较弱的固定强度可以改善直接路径。
- 唯一变化：`corrected = scale8_baseline + 0.5 × (N01a_full_corrected - scale8_baseline)`。
- multiplier 选择：只在开发集 fold 0 上使用预设网格 `{0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2}`；最低 pooled RMSE 对应 `0.5`。fold 1 只作原样确认，不重新选择。正式全五折锁死 `0.5`，不再调参。
- 邻井生成：完全复用 N01a 的 outer-train 四控制点 profile、完整隐藏段 XY 几何、筛选、权重、方向倒序和 `eta`；不改变任何邻井参数。
- 数据边界：只使用 657 口开发井；每个 outer fold 的 source 标签来自其他 fold；116 口影子井在读取水平井前物理排除。
- 基线：`last_visible_tvt + pf128_scale_8_delta`。
- 固定 CV：`balanced_well_5fold_v1`，完整井验证，自然隐藏区。
- 负对照证据：沿用 N01a 的固定 seed 打乱 profile；它不参与 multiplier 选择。
- 成功门槛：全 657 口直接路径相对 scale8 pooled RMSE 至少改善 `0.10 ft`，且至少 4/5 折同方向改善。
- 通过后动作：只准备固定单模 LightGBM 接入，不自动训练。
- 停止条件：全五折任一门槛失败，则停止 N01b，不接入 LightGBM。
- 长任务：已有 folds 0～1 的 263 口 N01a 缓存直接复用；仅生成 folds 2～4 剩余 394 口，逐井缓存支持中断续跑。
- 失败只能否定：统一 `0.5` 强度的 N01a 确定性整段修正，不能否定学习型门控或逐位置可靠性权重。

