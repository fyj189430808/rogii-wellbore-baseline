# P2-P03 实验卡：128 随机种子路径分歧

实验编号：`P2_P03_pf_seed_dispersion_v1`

实验名称：在 P2-P02 上加入逐行 PF 随机种子标准差

唯一假设：128 个固定随机种子给出的 PF 路径在某一井段越分散，该井段的 PF 路径越不可靠；把这个测试期合法的分歧交给同一个 LightGBM，可能帮助模型减少过度修正。

为什么值得验证：P2-P01 和 P2-P02 已证明多随机种子均值路径及多尺度完整路径有稳定增量，但容易井仍可能被过度修正。`pf128_seed_std` 已由同一次合法 PF 生成过程保存，不需要重算 PF，也不增加新的数据来源。

研究过程提示：本假设是在查看 P2-P01、P2-P02 完整五折结果后登记的，属于开发证据，不是从未使用过的最终留出集。

与当前最佳基线唯一不同之处：P2-P02 的 41 个特征全部保留，只在末尾新增一列 `pf128_seed_std`，正式共 42 列。

合法输入：现有 P2-P01 单井合法缓存。原始来源只有当前水平井 `MD/Z/GR/TVT_input` 与对应 Typewell `TVT/GR`；不使用隐藏 TVT、surface、邻井标签或外折拟合量。

特征公式：对同一隐藏行的 128 条固定随机种子 PF 绝对 TVT 路径计算总体标准差：

```text
pf128_seed_std[i] = std(seed_path[0:128, i], ddof=0)
```

单位：ft。因为 128 条路径使用相同起点，对绝对 TVT 或相对 delta 求标准差完全相同。

预测目标：自然隐藏段 `TVT - last_visible_tvt`；评分时还原绝对 TVT。

固定 fold：`balanced_well_5fold_v1`。

固定评价行：773 口井、3,783,989 个 `TVT_input.isna()` 行。

本次使用的模型：唯一一个 LightGBM，完整参数与 P2-P02 相同，1,734 棵树、seed 29，无 early stopping、统一行权重。

本次使用的特征：P2-P02 冻结 41 列加 `pf128_seed_std`，共 42 列。

正对照：冻结的 P2-P02 OOF，micro RMSE `10.305704992073148`。

负对照：folds 0～1 数值门槛通过后，在 fold 0 只把 `pf128_seed_std` 按每口井的自然隐藏行顺序反转。它保留每井分布、缺失情况和数值范围，只破坏分歧与当前位置的对应关系；其余 41 列完全不变。

负对照门槛：fold 0 正式 P03 必须比反转版至少好 `0.05 ft`，否则认为新增列的当前位置含义没有得到支持，停止，不运行 folds 2～4。

oracle 诊断：完整五折完成后才读取真值，单独报告 `seed_std` 分位桶与 PF/P02 绝对误差、P02→P03 逐井增益的关系。只用于理解，不进入特征、模型选择或晋级门槛。

成功门槛：

```text
folds 0～1 pooled micro 相对 P2-P02 改善 >= 0.20 ft
fold 0 和 fold 1 任一折不得恶化超过 0.25 ft
fold 0 正式 P03 相对井内反转负对照改善 >= 0.05 ft
完整五折相对 P2-P02 改善 >= 0.10 ft
至少 4/5 折改善
按井 bootstrap 95% CI 上界 < 0
P90 不得恶化超过 0.50 ft
收益不能只来自极少数井
```

停止条件：folds 0～1 或负对照门槛未通过则不训练 folds 2～4；任何 P2-P02 基线哈希、P01 缓存指纹、行键、锚点、42 列顺序、模型参数或 fold 注册表不一致立即停止。

预计运行时间：缓存合并约 1～3 分钟；folds 0～1 约 5～12 分钟；若晋级，负对照和剩余三折约 15～30 分钟。

需要生成的文件：

```text
artifacts/P2_P03_pf_seed_dispersion_v1/
├── config.json
├── feature_list.json
├── cache_and_fold_provenance.json
├── fold_0 ... fold_4/
├── predictions.parquet
├── per_well.csv
├── per_fold.csv
├── metrics.json
├── comparison_vs_p02_folds01.json
├── comparison_vs_p02_full5.json
├── negative_control_fold0.json
├── oracle_seed_dispersion_audit.json
├── runtime.json
└── conclusion.md
```

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定“把原始逐行 `pf128_seed_std` 直接作为一列加入当前 P2-P02 LightGBM”；不能否定多分位宽度、多峰特征、分歧增长、低维路径残差或重新设计的可靠性表示。
