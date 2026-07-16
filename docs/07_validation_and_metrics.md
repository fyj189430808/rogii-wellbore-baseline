# 固定 CV、评价行和统一指标

## 0. 当前能不能回答“最高代码 CV 是多少”

不能给出一个可信的全栈 CV 数字。

当前证据只能回答各组件的旧 CV，不能回答最终高分 Notebook 的 private-safe CV。

| 对象 | 已有数字 | 能否作为当前最终 CV | 原因 |
|---|---:|---:|---|
| 历史最佳单 LightGBM-3 | 10.4733 | 否 | 普通 well GroupKFold；空间 imputer 未按 outer fold 重建 |
| 五模型 Ridge stack | 10.4197 | 否 | 多模型；stacking 不严格 nested；特征有全局 imputer 风险 |
| Model Package raw OOF | 10.7106 | 否 | 多模型 + TCN |
| Model Package post OOF | 10.6702 | 否 | 多模型 + TCN + 后处理；imputer 未按 outer fold 重建 |
| selector optional report | 没有本次结果 | 否 | `RUN_CV_REPORT=False` |
| cell 54 full-stack ablation | 没有本次结果 | 否 | 开关关闭；且代码不含完整最终栈 |
| 历史高分提交 | 只有 LB/提交档案 | 否 | 受同井 contact 覆盖主导，没有完整 OOF |

因此当前正确表述是：

> 历史高分代码没有可信的整条 private-safe CV；必须在新冻结 fold 上重新计算。

## 1. 为什么旧 GroupKFold 不够

旧模型包使用：

    GroupKFold(n_splits=5)
    groups = well_id

它保证一口井不跨折，但不保证相邻井或同 pad 不跨折。

按每口井整条轨迹的中位 XY 检查：

- 旧 fold 中约 71%～81% 的验证井在 1,000 ft 内存在训练井；
- 约 94%～96% 的验证井在 2,500 ft 内存在训练井；
- formation/KNN imputer 只排除查询井，没有按 outer fold 重建。

这会让依赖邻井、surface 或空间 KNN 的特征明显乐观。

## 2. 新冻结 primary CV

配置：

    rogii_clean/configs/cv_spatial_pad_v1.json

注册表：

    rogii_clean/artifacts/folds/spatial_pad_1000_v1.csv

注册表 SHA-256：

    0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab

名称：

    median-XY 1000-ft connected-pad CV v1

这里的 “ft” 沿用项目对原始 XY 的当前解释。若后续官方字段审计证明单位不同，必须建立 v2，不能静默改写 v1。

## 3. pad 如何生成

### 3.1 一口井的代表坐标

对一口井的全部轨迹行：

    representative_x = median(X)
    representative_y = median(Y)

整条测试轨迹的 X/Y 在推理时可见，因此使用完整轨迹中位坐标不读取隐藏 TVT。

### 3.2 连边

若两口井代表点欧氏距离：

    distance <= 1000

就在两井之间连边。

### 3.3 pad

所有通过单链连通的井属于同一个 pad。一个 pad 不能拆到多个 fold。

`pad_id` 取该连通分量中字典序最小的 `well_id`。

结果：

- 773 口井；
- 290 个 pad；
- 155 个多井 pad；
- 多井 pad 覆盖 638 口井；
- 最大 pad 11 口井；
- 最大 pad 80,773 个隐藏评价行。

## 4. pad 如何分到五折

先按：

1. pad 隐藏行数降序；
2. pad 井数降序；
3. pad_id 升序

排序。

再依次把 pad 放入当前：

    (隐藏行数, 井数, pad数, fold_id)

字典序最小的 fold。

这个算法没有随机数。同一份输入应生成完全相同的 CSV 字节和 SHA-256。

## 5. 五折规模

| fold | wells | pads | hidden rows | 行占比 |
|---:|---:|---:|---:|---:|
| 0 | 147 | 57 | 757,050 | 20.0067% |
| 1 | 155 | 58 | 756,990 | 20.0051% |
| 2 | 157 | 58 | 756,649 | 19.9961% |
| 3 | 155 | 58 | 757,061 | 20.0070% |
| 4 | 159 | 59 | 756,239 | 19.9852% |
| 合计 | 773 | 290 | 3,783,989 | 100% |

按 fold 定义使用的中位代表点，验证井到训练井最近距离：

| fold | 最小距离 | 中位距离 |
|---:|---:|---:|
| 0 | 1,020.2 | 2,406.6 |
| 1 | 1,021.9 | 2,048.8 |
| 2 | 1,027.2 | 1,941.2 |
| 3 | 1,009.5 | 2,204.9 |
| 4 | 1,009.5 | 2,003.1 |

## 6. 这个 CV 不保证什么

它只保证：

> 中位 XY 代表点的 1,000-unit 连通 pad 不跨折。

它不保证：

- 两条完整轨迹的任意位置都相距 1,000 ft；
- PS 点一定相距 1,000 ft；
- 首点一定相距 1,000 ft；
- 相同 Typewell 模板不跨折。

实际诊断显示：

- 按 PS 坐标，仍有约 6.5%～12.9% 的验证井在训练井 1,000 ft 内；
- primary fold 下每折约 93.9%～99.4% 的验证井，在训练集中仍有相同 Typewell 模板。

所以不能把它称为“完整轨迹严格 buffer CV”或“新 Typewell 泛化 CV”。

## 7. Typewell strict 只作为第二诊断

773 个 typewell：

- 整文件 hash 有 752 种；
- 将共同 GR 尾段完全相同、只有浅部裁剪差异的文件归并后，只有 54 个模板；
- 760/773 口井属于多井模板；
- 最大模板有 71 口井。

primary spatial fold 仍有 40/54 个模板跨折。

因此后续应另存：

    typewell-template strict 5-fold

它回答“新 typewell 模板能否泛化”，不能替代 primary spatial fold。主实验排名仍只看固定 primary。

## 8. 固定评价行

对每口训练井：

    visible_mask = horizontal_df["TVT_input"].notna()
    hidden_mask = horizontal_df["TVT_input"].isna()

只在 `hidden_mask` 上评分。

固定要求：

- 3,783,989 个评价行；
- 773 口完整井；
- 同一个实验不能改变 mask；
- 缺失预测、重复 ID 或行数变化直接报错；
- 隐藏 `TVT` 在特征构建完成前应从验证 DataFrame 物理删除。

## 9. private-safe outer-fold 规则

每个 outer fold：

1. 验证井全部行不能进入训练；
2. 验证 pad 的所有井不能进入训练派生特征；
3. formation/KNN/空间面必须只用 outer-train 重建；
4. 标准化、缺失填充值和特征选择只用 outer-train 拟合；
5. 参数和后处理不能看 outer-valid 隐藏 TVT；
6. 同 ID 查询井始终从邻井索引排除；
7. 最终训练测试时，任何与测试 well_id 重合的训练井都从训练样本、KNN、surface 和 Typewell 标签统计中删除；
8. `well_id` 和完整 `id` 不进入 LightGBM 特征。

只删除 cell 48 不足以满足这些规则。预训练模型若曾见过同 ID 井，也不能视为 private-safe。

## 10. 固定单模 LightGBM

配置：

    rogii_clean/configs/lgbm_feature_baseline_v1.json

冻结内容：

- 唯一学习模型：LightGBM；
- target：`TVT - last_known_TVT`；
- 最终：`last_known_TVT + predicted_delta`；
- seed：29；
- 固定树数：1,734；
- 不使用 outer-valid early stopping；
- 不用 stacking、blend、learned trajectory 或 model package；
- 不使用任何其他模型预测/OOF 作为特征。

1,734 来自历史最佳单 LightGBM-3 五折 best iteration：

    1828, 1330, 1734, 429, 3383

其中位数是 1,734。这个选择只做一次，后续特征实验不再调整。

## 11. 固定对照

每次特征实验至少同时报告：

1. carry-forward；
2. 冻结 PF；
3. B0 单模 LightGBM；
4. B0 + 本次唯一新特征组。

PF 可以作为确定性候选/特征生成器，但不能作为第二个学习模型做输出融合。

## 12. 统一指标

代码：

    rogii_clean/src/metrics.py
    rogii_clean/scripts/score_predictions.py

必须报告：

- micro RMSE；
- macro per-well RMSE；
- 每折 micro RMSE；
- median well RMSE；
- P90 well RMSE；
- worst well RMSE；
- 相对固定基线的胜井率；
- 运行时间；
- 按井配对 bootstrap 95% CI。

### 12.1 micro RMSE

    sqrt(sum_all_hidden_rows(error^2) / total_hidden_rows)

这是 Kaggle 主指标。

### 12.2 macro per-well RMSE

先分别计算每口井 RMSE，再对 773 口井等权平均。

### 12.3 paired well bootstrap

以井为抽样单位，有放回抽 773 口井；每次将一口井全部行一起带入，再计算：

    candidate_micro_rmse - baseline_micro_rmse

固定：

- 2,000 次；
- seed 42；
- 负值代表新特征更好。

## 13. 开发与晋级制度

### 阶段 1：fold 0 筛查

只运行固定 fold 0。

进入确认折的门槛：

    micro RMSE 至少改善 0.15 ft

### 阶段 2：确认折

使用固定 fold 1。

要求：

- 与 fold 0 同方向；
- 负对照不能得到相近改善。

### 阶段 3：完整五折

要求：

- 五折 micro 至少改善 0.10 ft；
- 至少 4/5 折同方向；
- 井级 paired bootstrap 95% CI 上界小于 0；
- P90 不恶化超过 0.5 ft。

不满足就记录负结果，不继续调 LightGBM、fold 或后处理挽救。

## 14. 如何重建并验证 fold

在 `rogii_clean/` 下：

    python scripts/make_fixed_folds.py
    python -m pytest tests/test_fold_split.py -q

当前验证结果：

    1 passed

重新生成的 registry SHA-256 必须是：

    0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab

## 15. 当前仍然没有的结果

本轮没有训练 LightGBM，所以还没有：

- 新 spatial-pad fold 的 carry RMSE；
- 新 fold 的 PF RMSE；
- B0 单模 LightGBM RMSE；
- 任何 F01～F07 特征实验结果。

这些数字必须按“小样本 → fold 0 → 确认折 → 五折”顺序重新产生，不能引用旧 GroupKFold 数字代替。

## 16. 结论分级

### 数据直接证明的事实

- 历史最终栈没有可信 private-safe CV。
- 旧 GroupKFold 空间隔离弱。
- 新 fold 注册表有 773 井、290 pad、3,783,989 行。
- 五折隐藏行数接近精确 20%。
- 固定注册表已通过独立重建测试。

### 基于事实的合理推断

- 新 CV 会比旧普通井级 GroupKFold 更严格。
- 空间/KNN 特征在新 CV 上很可能比旧 OOF 更弱。
- 去掉同井标签通道后，历史 LB 分数不能作为新方法预期。

### 仍然没有验证的猜测

- 新 CV 与 private leaderboard 分布有多接近。
- 1,000-unit 半径是不是最接近真实 pad 定义。
- Typewell 重复在 private test 中是否仍大量存在。

### 下一步最便宜的验证

先只复算：

1. carry-forward；
2. 冻结 PF；
3. B0 单模 LightGBM 的 fold 0。

在这三项稳定复现前，不启动 F01 特征实验。
