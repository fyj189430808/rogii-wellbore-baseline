# P2-P02 固定四尺度 PF 路径执行计划

1. 先用测试冻结 41 列正式特征、P2-P01 基线哈希、完整 LightGBM 参数和只允许 `0,1`/`all` 的折合同。
2. 只从 P2-P01 单井合法缓存读取均值路径与四条 scale 路径，按 `well_id + row_index` 一一合并；拒绝 seed0、seed_std、目标、surface 和 oracle 列。
3. 运行 folds 0～1，比较对象固定为 P2-P01，不再使用 P2B00 作为晋级门槛。
4. 若通过，先运行 fold 0 井内逆序负对照，再运行 folds 2～4。
5. 生成完整五折指标、按井 bootstrap、scale oracle 诊断和结论；不根据结果删除某个 scale。
