# PF128 outer0 缓存补齐修改说明卡

修改目标：让既有 PF03 运行器支持只生成 fold0 合法 seed 路径，不读取标签、不进行 PF03 评分。

当前代码行为：只支持 smoke、fold01 和 all；为了补 85 口 outer0，运行 fold01 会额外生成 129 口无关 fold1 缓存并执行本实验不需要的 PF03 oracle 评分。

准备修改的文件：`scripts/run_p3_pf03_segmented_likelihood.py` 及对应测试。

准备修改的函数：命令行参数、井选择、生成完成后的提前返回和 runtime 输出。

为什么要修改：只补当前 PFM01 必需的 outer0 缓存，减少时间和无关标签接触。

修改前的数据流：选择 fold01 → 生成/复用缓存 → 读取 target → PF03 评分。

修改后的数据流：`--generation-only-fold 0` → 只选开发 fold0 → 生成/复用缓存 → 保存合法逐井清单和 runtime → 立即结束，永不读取 target。

会影响哪些特征、模型、参数：都不影响；PF 核心、128 seeds、配置指纹、共享缓存 schema 和 8 线程保持原值。

会影响哪些缓存：只补 `P3_shared_pf_seed_paths_v1`、PF03 对应合法路径及 runtime；已有文件只有完整指纹和 SHA 全匹配才复用。

预期结果：46 口命中缓存，85 口新生成，最终 outer0 131/131 完整。

可能风险：generation-only 分支误入评分阶段。测试必须让 target 读取函数在该模式下抛错，并确认仍能正常结束。

如何回滚：删除新增命令分支；已生成单井缓存与旧配置/指纹相同，仍是合法可复用资产，无需删除。
