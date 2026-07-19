# P3_PF02b_replace_fixed_scales_lgbm_v1 实验卡

实验编号：`P3_PF02b_replace_fixed_scales_lgbm_v1`

实验名称：固定 ESS 路径等维替换固定 scale 路径

唯一假设：上一版把四条 ESS 路径追加到 41 列基线上，形成 45 列，可能因路径高度共线而过拟合；在相同位置一一替换旧路径，可能保留 ESS 信息同时避免增加维度。

为什么值得验证：PF02 的 657 口合法缓存已经完成，不需要重新生成 PF，只需复用缓存训练单模 LightGBM。

与当前最佳基线唯一不同之处：

- `pf128_scale_3_delta` 替换为 `pf128_ess2_delta`；
- `pf128_scale_5_delta` 替换为 `pf128_ess8_delta`；
- `pf128_scale_8_delta` 替换为 `pf128_ess32_delta`；
- `pf128_scale_12_delta` 替换为 `pf128_ess96_delta`。

其余 37 列、特征顺序、单模 LightGBM、1734 棵树、seed 29、fold、自然隐藏评价行和评分实现全部不变；总特征数仍为 41。

合法输入：当前井可见数据、Typewell、冻结 P3B00 特征和只由合法输入生成的 PF02 ESS 路径缓存。

预测目标：自然隐藏区 `TVT - last_visible_tvt`。

固定 fold：`balanced_well_5fold_v1`，日常开发只使用去除 116 口影子井后的 657 口井。

固定评价行：开发集 3,211,872 行自然隐藏区。

正对照：同一开发井和评价行上的 `P3B00_group5_p2p02_v1` OOF。

负对照：无；本实验只做预注册的一一替换，不根据 fold 结果挑选 ESS 子集。

oracle 诊断：无。

folds 0～1 成功门槛：合并 micro RMSE 至少改善 0.20 ft，任一折不得恶化超过 0.10 ft。

完整五折成功门槛：沿用三阶段统一晋级门槛。

停止条件：folds 0～1 未通过就不训练 folds 2～4；不得根据结果改成挑选其中一条或几条 ESS 路径。

预计运行时间：直接复用 657 口路径缓存；folds 0～1 主要耗时为两次 LightGBM 训练。

需要生成的文件：独立 artifact 下的配置、特征清单、参数、泄漏审计、逐折预测、逐井指标、逐折指标、运行时间和结论。

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：固定映射 `scale 3/5/8/12 → ESS 2/8/32/96` 的等维 41 列单模 LightGBM 实现。
