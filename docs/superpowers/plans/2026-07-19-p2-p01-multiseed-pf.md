# P2-P01 实施计划

1. 先用单元测试固定 Notebook 第 37 单元格的状态更新、随机种子和输出字段。
2. 实现只读取 `MD/Z/GR/TVT_input` 与 Typewell `TVT/GR` 的确定性核心函数。
3. 实现逐井原子缓存、配置指纹、8 线程并发和中断续跑。
4. 运行固定首井 `000d7d20` 的 exact-128 smoke，检查重复、并发和隐藏 TVT 不变性。
5. 只生成 fold 0 的 155 口井；合法缓存落盘后再读取真值，比较当前单 PF、同参数 seed 0、128-seed mean、carry 和反转负对照。
6. 只要程序与合法性门槛通过就补齐 773 井，并只给 C01 增加 `pf128_mean_delta`；路径自身 RMSE保留为诊断，不作为提前停止条件。
7. 先跑 folds 0～1；达到模型门槛后完成五折和井级 bootstrap。
8. 保存正负结果，更新 registry、current_state 和 phase2 roadmap。
