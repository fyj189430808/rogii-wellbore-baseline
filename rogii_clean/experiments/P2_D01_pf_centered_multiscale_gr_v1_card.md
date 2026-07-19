# P2-D01 实验卡

实验编号：`P2_D01_pf_centered_multiscale_gr_v1`

实验名称：PF 中心多尺度 GR 可辨识性审计

唯一假设：冻结 PF 路径附近的多尺度 Typewell GR 得分面包含可由测试期合法量识别的正确 offset 信号。

为什么值得验证：旧 F03b 的恒定 U 中心只有约四分之一真值落入 ±40 ft，而 PF 中心覆盖约 98.5%；必须先换正确中心再评价 GR scorer。

与当前最佳基线唯一不同之处：本实验不训练模型，只新增独立诊断产物，不改变 P2B00。

合法输入：当前井 MD/GR/TVT_input、对应 Typewell TVT/GR、冻结 `pf_ancc_delta`、按井 fold 标签。

预测目标：无模型目标；诊断目标为正确 offset 的覆盖、排名和误差。

固定 fold：`balanced_well_5fold_v1`，只用于分组报告。

固定评价行：自然隐藏段；评分按固定 50/100 ft 井段汇总。

正对照：可见前缀以 `TVT_input` 为中心时 offset=0 的排名。

负对照：隐藏 GR 在井内固定循环平移半段后，用同一 scorer 重算。

oracle 诊断：隐藏真实 TVT 只用于在 legal score 完成后附加正确 offset、排名和误差。

成功门槛：设计文档中五条预设可辨识性证据。

停止条件：PF cache 键不一致、legal scorer 接触隐藏 TVT、smoke 无有效块或正对照完全失效。

预计运行时间：3 井 smoke 约 1～3 分钟；全量 773 井预计 5～20 分钟，主要耗时来自 2 种块宽 × 5 个尺度 × 41 个 offset 的 GR 插值和相关计算。

需要生成的文件：配置、逐块 real/negative/prefix 结果、逐井汇总、summary、runtime、lineage 和 conclusion。

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定固定 PF-ancc 中心、50/100 ft 分块、五个固定平滑尺度和 Pearson NCC 组合。

