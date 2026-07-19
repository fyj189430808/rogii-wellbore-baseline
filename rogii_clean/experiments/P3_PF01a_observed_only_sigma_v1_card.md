# P3-PF01a 实验卡：只用真实可见 GR 估计 sigma

实验编号：`P3_PF01a_observed_only_sigma_v1`

实验名称：observed-only GR sigma 的 128-seed PF

唯一假设：旧 PF 把可见段缺失 GR 填成 0 后估计噪声，导致 sigma 系统性偏大、真实 GR 观测过弱；只用可见且 GR 非缺失的行估计 sigma 能生成更准确的 PF 路径。

为什么值得验证：D01 的 657 口开发井中，旧 sigma 相对 observed-only sigma 的井级中位绝对差为 170.49%，而缺失比例本身与 PF mean 误差相关接近 0。

与当前基线唯一不同之处：`gr_sigma` 的样本集合从“可见 GR 缺失填 0 后的全部可见行”改为“可见且原始 GR 有限的行”。其余 PF 参数、隐藏 GR 插值、四个温度、fold、开发井和评价行不变。

合法输入：水平井 `MD/Z/GR/TVT_input`，Typewell `TVT/GR`，冻结 P2-P01 参数，固定开发井注册表。

预测目标：先生成新 PF 路径；路径生成完成后才读取开发井目标，计算 mean 和 scale 3/5/8/12 的 RMSE。

固定 fold：`balanced_well_5fold_v1`；116 口影子井保持关闭。

正对照：旧 P2-P01 四温度路径和 P2-P02 OOF。

负对照：若某井可见 GR 无缺失，新旧 sigma 必须完全相同；小型确定性 PF 中只改 sigma 不应改变其他准备输入。

oracle 诊断：每井事后最佳温度，只估计路径集合余量，不作正式特征。

成功门槛：folds 0～1 最佳固定温度路径相对旧最佳固定温度至少改善 0.20 ft，且两折不出现超过 0.25 ft 的明显恶化；通过后才跑全开发集并替换 P3B00 的五条 PF 路径特征训练同一 LightGBM。

停止条件：smoke 出现非有限值、影子井进入任务、除 sigma 外的 PF 输入发生改变，或 folds 0～1 所有固定路径均无改善。

预计运行时间：smoke 约 1 分钟；folds 0～1 约 25 分钟；通过后剩余三折约 35～40 分钟。

需要生成的文件：冻结配置、逐井合法路径缓存、逐井 runtime、模式路径指标、结论。

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定“直接用 observed-only residual std 替换旧 sigma，且不同时校正 GR 幅值或目标 ESS”的实现。
