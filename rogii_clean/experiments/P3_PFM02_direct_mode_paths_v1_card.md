# P3-PFM02 实验卡：三条模式路径直接进入 LightGBM

实验编号：`P3_PFM02_direct_mode_paths_v1`

实验名称：在 P3B00 原 41 列后直接增加低、中、高三条 PF 模式路径

唯一假设：三条按 TVT 位置统一命名的完整模式路径，虽然不能被简单井级方向分数稳定硬选，但 LightGBM 可以结合已有逐行几何、GR、PF 和 Beam 特征，学习它们对 TVT 的条件增量。

为什么值得验证：PFM01 的三模式逐井 oracle pooled RMSE 为 `9.2290995`，旧五温度路径 oracle 为 `10.3738187`，证明候选覆盖增加；把三条逐行路径直接交给冻结单模 LightGBM 是成本最低的合法选择器测试。

与当前最佳基线唯一不同之处：P3B00/P2-P02 的 41 列、模型、目标、fold、行权重和评分全部不变，只在末尾追加：

```text
pf_mode_low_delta
pf_mode_middle_delta
pf_mode_high_delta
```

合法输入：原 41 列，以及由当前井 PF128 路径和最终 GR 似然在读取隐藏真值前生成的三条模式 delta 路径。

禁止输入：`direction_score`、三种 `mode_mass`、`seed_count`、`p2_position`、模式 separation，以及 PFM01 `oracle/` 目录中的任何字段。

预测目标：`target_tvt - last_visible_tvt`。

固定 fold：`balanced_well_5fold_v1`；只使用排除 116 口影子井后的 657 口开发井。预筛先运行 folds 0～1，通过后才运行 folds 2～4。

固定评价行：开发集 3,211,872 个自然隐藏点；fold 0/1 分别为 651,881/630,395 行。

本次使用的模型：唯一一个 `lightgbm.LGBMRegressor`。

本次使用的参数：冻结 `configs/lgbm_feature_baseline_v1.json`，1,734 棵树、seed 29、无 early stopping、逐行等权；所有其余参数逐项完全一致。

正对照：冻结 P3B00 开发井配对 OOF，micro RMSE `10.2721462675`。

负对照：第一版不增加新的训练型或井级摘要对照；严格检查新增三列不是重复旧五路径、不是常数、不是缺失填充，并保存逐折特征重要性。若 folds 0～1 晋级，再登记并运行井内同步反转三条路径的 fold 0 负对照。

oracle 诊断：只引用已经独立完成的 PFM01 候选覆盖诊断，不在本实验中用 oracle 选择路径或特征。

成功门槛：沿用现有 P3 单模特征组合同。

- folds 0～1 pooled 改善至少 `0.20 ft`；
- folds 0～1 任一折恶化不超过 `0.10 ft`；
- 通过后才运行 folds 2～4；
- 完整五折改善至少 `0.10 ft`；
- 至少 4/5 折改善，且 folds 2～4 至少 2 折改善；
- 最差折恶化不超过 `0.25 ft`；
- 按井配对 bootstrap 95% 区间上界小于 0；
- 胜井率至少 55%，P90 恶化不超过 `0.20 ft`；
- 最强 5% 井占正收益不超过 60%。

停止条件：folds 0～1 未通过即停止，不训练 folds 2～4；任一来源哈希、657 井/3,211,872 行、自然行键、锚点、44 列顺序、模型参数或缓存指纹不一致立即停止。

预计运行时间：补齐 482 口 PF128 原始路径约 45～55 分钟；统一三模式合法缓存约 3～5 分钟；folds 0～1 LightGBM 约 15～20 分钟。

需要生成的文件：标准 `config.json`、`metrics*.json`、`per_well*.csv`、`predictions*.parquet`、`feature_list.json`、`parameter_list.json`、`runtime.json`、`conclusion.md`，以及逐井合法三模式缓存和运行记录。

失败后能否否定整个方向：不能。

失败后只能否定哪一种实现：只能否定“把固定 mean/end Ward K=3、scale8 加权中心得到的低/中/高三条完整路径，原样追加到冻结 41 列单模 LightGBM”。

## 修改说明卡

修改目标：新增 PFM02 的合法三模式缓存生成器和 44 列冻结 LightGBM runner。

当前代码行为：PFM01 只为 fold 0 保存三模式路径并做诊断，`model_training=False`；现有 P3-PF02 runner 支持 41 列加外部路径，但读取的是 ESS 路径。

准备修改的文件：只新增 PFM02 配置、缓存生成脚本、训练脚本和测试；PFM01、P3B00、P2-P02、PF03 旧代码不改。

准备修改的函数：新增三模式合法缓存校验/生成、44 列清单构造、三模式缓存合并和 PFM02 主运行函数。

为什么要修改：候选模式已经合法生成，但尚未进入主 LightGBM，无法检验条件选择价值。

修改前的数据流：PF128 → PFM01 三模式 → 只读 oracle/方向诊断。

修改后的数据流：PF128 → 独立三模式合法缓存 → 原 41 列按自然行键追加三列 → 冻结 LightGBM → 配对 CV。

会影响哪些特征：只增加三列，原 41 列逐位保留。

会影响哪些模型：只新增一个与基线参数完全相同的单模 LightGBM 实验；不修改旧模型。

会影响哪些参数：不影响，全部冻结。

会影响哪些缓存：新增 `artifacts/P3_PFM02_mode_paths_v1/` 和 `artifacts/P3_PFM02_direct_mode_paths_v1/`；不会覆盖旧缓存。

预期结果：判断 LightGBM 能否利用逐行上下文，从三条候选模式中提取有效条件增量。

可能风险：训练井模式缓存不完整、自然行错位、锚点精度、误读 PFM01 oracle 字段、影子井目标进入训练。

如何回滚：删除两个新的 PFM02 artifact 目录和新增代码；旧 P3B00/PFM01 产物不受影响。
