# R01b 平均残差设计

R01b 复用 R01a 已生成的严格嵌套预测，不重新训练 P2-P02。对 folds 1-4 的每口井计算 `mean(target_tvt - pred_tvt)` 作为一个井级训练目标；使用 R01a 简化版固定的 21 个合法井级特征和 `StandardScaler + Ridge(alpha=10)`，预测 outer fold 0 每口井的平均残差，并把该常数加到整条基础路径。

训练单位是一口井，验证单位也是完整井。缺失值中位数、标准化器和 Ridge 只能在 526 口 outer-train 井上拟合。负对照按井打乱训练目标。outer fold 0 真值只能在预测完成后用于评分、oracle 和相关性诊断。

该实验不选择特征、不搜索 alpha、不调收缩比例、不预测斜率，也不接回逐行 LightGBM。未达到实验卡门槛就停止当前实现。
