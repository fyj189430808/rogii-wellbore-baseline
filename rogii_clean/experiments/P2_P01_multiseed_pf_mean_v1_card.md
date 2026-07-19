# P2-P01 实验卡

实验编号：`P2_P01_multiseed_pf_mean_v1`

实验名称：128 个固定随机种子的粒子路径均值特征

唯一假设：当前 C01 只使用一条 600 粒子单次 PF；把旧高分代码中测试期合法、固定 `0～127` 随机种子的 128 条 500 粒子路径取逐行均值，可以显著降低单次 PF 的随机漂移，并为固定单模 LightGBM 提供一条更准确的完整路径。

为什么值得验证：

- 《一阶段总结》明确登记了“当前 C01 不是历史多 seed selector PF”；
- 旧高分模型的 196 个特征中只有 `likpf_mean_d` 没有保存在 7.39 GB 旧训练表里；
- 该列在三个旧 LightGBM 中的重要性分别排第 6、第 1、第 7；
- 它只用本井可见前缀、整井 GR、MD/Z 和配对 Typewell TVT/GR，不用 surface、邻井标签、隐藏 TVT 或训练—测试同井覆盖。

与当前最佳基线唯一不同之处：正式模型只在 P2B00/C01 的 36 个特征后增加一列 `pf128_mean_delta`。LightGBM、树数、seed、target、fold、行权重、评分和输出公式全部不变。

合法输入：

- 当前井 `MD/Z/GR/TVT_input`；
- 当前井对应 Typewell 的 `TVT/GR`；
- 固定随机种子 `0～127` 和本卡冻结的 PF 数值参数。

预测目标：仍为 `TVT - last_visible_tvt`；新特征是 128 条 PF 隐藏路径逐行均值减去 `last_visible_tvt`。

固定 fold：`balanced_well_5fold_v1`。

固定评价行：`TVT_input.isna()`，全量 773 口井、3,783,989 行；fold 0 为 155 口井、757,738 行。

固定 PF 参数：

```text
粒子数                         500
随机种子                       0～127，共 128 个
Typewell 重采样步长            0.2 ft
初始 U 位置标准差              4.5 ft
初始 U 倾角                    最近 30 个可见点的 Δ(TVT_input+Z)/ΔMD 中位数
初始倾角标准差                 0.01
倾角动量                       0.998
倾角转移噪声                   0.002
位置转移噪声                   0.005 ft
Typewell 外允许范围            上下各 100 ft
GR 观测尺度                    可见前缀残差标准差，裁剪到 10～60 API
有效粒子数重采样阈值           0.5 × 粒子数
重采样位置扰动                 0.1 ft
重采样倾角扰动                 0.001
正式特征的跨 seed 汇总         不加权逐行均值
```

正对照：

1. 同一井同一配置连续运行两次，输出逐位一致；
2. 并发与串行运行同一组井，输出逐位一致；
3. 删除整列 `TVT`、删除全部 surface 列，或随机改写水平井隐藏 `TVT` 后，特征逐位不变；生成器入口只读取白名单列；
4. 小型数组上的一 seed Numba 内核与直接循环参考实现一致；
5. 每口井输出行键必须与自然隐藏行完全一致，所有值有限。

匹配对照与负对照：

- 同一组 500 粒子参数的 seed 0 单路径作为匹配对照，用来区分“128-seed 均值”与参数变化；
- fold 0 合法路径生成后，把每口井的 `pf128_mean_delta` 按隐藏行顺序反转；反转路径只用于诊断，不进入模型。

oracle 诊断：

- 真实 TVT 只在合法缓存全部落盘后读取，用于计算 mean 路径、四条 likelihood scale 路径和逐井反转路径的 RMSE；
- best-of-scale、best-of-seed 只允许标成 oracle，不能进入正式特征或版本选择。

特征生成阶段成功门槛：

- 1 井 exact-128 smoke 的全部程序正对照通过；
- 生成器物理上不读取 `TVT` 或 surface，缓存键与自然隐藏行完全一致；
- 重复、并发和删除隐藏真值后的最大差都为 `0`；
- 输出全部有限，缓存配置与代码指纹完全匹配。

fold 0 仍必须报告 `pf128_mean`、同参数 seed 0、当前 `pf_ancc`、carry、四条 likelihood-scale 路径和逐井反转路径的 RMSE、胜井率与 P90，但这些是诊断，不是阻止一条互补特征进入 LightGBM 的必要条件。只要程序和合法性门槛通过，就生成其余 618 口井并训练固定模型。

模型阶段成功门槛：

- folds 0～1 合并相对 P2B00 改善至少 `0.20 ft`；
- fold 0、1 任一折不得恶化超过 `0.25 ft`；
- 通过后自动完成五折；
- 完整五折改善至少 `0.10 ft`，至少 4/5 折同方向；
- 井级配对 bootstrap 95% CI 上界小于 0；
- P90 well RMSE 不恶化超过 `0.5 ft`，收益不能只来自极少数井。

停止条件：程序正对照失败、行键/配置指纹不一致、读取隐藏 TVT/surface 构造特征、输出非有限，或模型阶段门槛失败。fold 0 路径自身分数不作为提前停止条件。

预计运行时间：1 井 exact smoke 约 30～40 秒；8 线程 fold 0 约 15～35 分钟；若晋级，全 773 井总计约 1～2.5 小时。每口井独立落盘，支持中断后继续。

需要生成的文件：

```text
artifacts/P2_P01_multiseed_pf_mean_v1/
├── config.json
├── lineage.json
├── legal_cache/<well_id>.parquet
├── legal_runtime/<well_id>.json
├── path_metrics_fold0.json
├── per_well_path_fold0.csv
├── feature_list.json
├── parameter_list.json
├── predictions.parquet
├── metrics.json
├── runtime.json
└── conclusion.md
```

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定旧高分 Notebook 第 37 单元格这一组固定参数产生的 128-seed 均值路径，或只把这一列加入当前 C01 的实现；不能否定其他粒子参数、likelihood scale 路径、PF–Beam 整段分型或所有多随机种子方法。若 mean 优于同参数 seed 0，才能把其中一部分收益解释为多 seed 降低随机性；否则只能把整条 lik-PF 表示作为整体评价。
