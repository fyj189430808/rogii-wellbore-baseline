# P2-P02 实验卡：固定四尺度 PF 路径

实验编号：`P2_P02_multiscale_pf_paths_v1`

实验名称：在 P2-P01 上加入固定四尺度 PF 整段路径

唯一假设：不同观测尺度适合不同井和井段；让同一个固定 LightGBM 同时看到 scale 3、5、8、12 的四条整段路径，会比只看到 `pf128_mean_delta` 更准确。

为什么值得验证：P2-P01 已证明整段 PF 路径是五折第一重要特征。四条 scale 路径在生成 P2-P01 前已经固定并保存，且全量裸路径均有正确信号。它们可能让模型按当前井段自动选择更合适的观测平滑程度。

研究过程提示：本假设是在看过 P2-P01 五折路径诊断后登记的，因此是下一轮开发证据，不是从未看过的最终留出集。为减少事后挑选，不单独选择裸路径最好的 scale 8，而是把预先固定的四条 scale 路径作为一个整体特征组。

与当前最佳基线唯一不同之处：P2-P01 的 37 个特征全部保留，只新增以下四列，正式共 41 列：

```text
pf128_scale_3_delta
pf128_scale_5_delta
pf128_scale_8_delta
pf128_scale_12_delta
```

合法输入：只复用 P2-P01 已通过删除隐藏 TVT、改写隐藏 TVT、重复和并发检查的单井合法缓存。原始来源仍只有水平井 `MD/Z/GR/TVT_input` 与 Typewell `TVT/GR`。

预测目标：自然隐藏段 `TVT - last_visible_tvt`；评分时还原绝对 TVT。

固定 fold：`balanced_well_5fold_v1`。

固定评价行：773 口井、3,783,989 个 `TVT_input.isna()` 行。

本次使用的模型：唯一一个 LightGBM，完整参数与 P2-P01 相同，1,734 棵树、seed 29。

本次使用的特征：P2-P01 的 37 列加四条固定 scale 路径，共 41 列。

本次使用的参数：复用 `lgbm_feature_baseline_v1.json`；PF 参数和缓存完全不变。

正对照：冻结的 P2-P01 OOF，micro RMSE `10.93453719961266`。

负对照：若正式候选通过 folds 0～1，则在 fold 0 额外把四条新增路径分别按井内自然隐藏行逆序，保持每列分布和井身份但破坏当前位置含义。正式候选应明显优于该逆序版本；负对照不参与特征选择。

oracle 诊断：事后计算每口井四条 scale 裸路径中最优一条的上限，只用于判断候选多样性，不进入模型、门槛或正式结论。

成功门槛：

```text
folds 0～1 pooled micro 相对 P2-P01 改善 >= 0.20 ft
fold 0 和 fold 1 单折均不得恶化超过 0.25 ft
完整五折相对 P2-P01 改善 >= 0.10 ft
至少 4/5 折改善
按井 bootstrap 95% CI 上界 <= 0
P90 不得恶化超过 0.50 ft
```

停止条件：folds 0～1 未通过则不训练 folds 2～4；任何缓存指纹、列、行键、模型参数或基线哈希不一致立即停止。

预计运行时间：缓存合并和检查约 1～3 分钟；folds 0～1 约 5～12 分钟；晋级后剩余三折约 10～20 分钟。

需要生成的文件：

```text
artifacts/P2_P02_multiscale_pf_paths_v1/
├── config.json
├── feature_list.json
├── cache_and_fold_provenance.json
├── fold_0 ... fold_4/
├── predictions.parquet
├── per_well.csv
├── per_fold.csv
├── metrics.json
├── comparison_vs_p01_folds01.json
├── comparison_vs_p01_full5.json
├── negative_control_fold0.json
├── oracle_scale_audit.json
├── runtime.json
└── conclusion.md
```

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定“把这四条固定 scale 路径直接作为四列加入当前 P2-P01 LightGBM”的条件增量；不能否定多尺度观测、PF 可靠性门控或重新设计观测模型。
