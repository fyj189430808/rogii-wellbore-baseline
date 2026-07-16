# 从 CSV 到最终 TVT 的完整数据流

## 0. 这条流水线解决什么问题

输入是一口井的可见前缀和隐藏后缀：

    visible_mask = horizontal_df["TVT_input"].notna()
    hidden_mask  = horizontal_df["TVT_input"].isna()

可见前缀给出真实 `TVT_input`，隐藏后缀只给出轨迹和测井信息。输出是隐藏后缀每一行的 `TVT` 预测。

当前 Notebook 同时运行多条候选轨迹，并多次覆盖提交文件。为了不混淆，本文把数据流拆成：

1. 原始数据；
2. 可见/隐藏切分；
3. 第一条残差模型；
4. selector PF/Beam；
5. 同井物理轨迹；
6. anchor 和投影；
7. 第二条 learned trajectory；
8. 同井覆盖代码（历史高分启用，当前 private-safe 默认关闭）；
9. 可见前缀候选选择；
10. 模型包微调；
11. 提交审计和评分。

## 1. 原始数据进入内存

### 1.1 数据定位

第一条轨道使用 Notebook 单元格 10：

    CFG.dataset_path = Path(COMPETITION_DATA_ROOT)

第二条轨道使用单元格 33：

    CFG.DATA = _find_data()

Kaggle 路径优先；本地开发可以通过 `ROGII_DATA` 环境变量指定数据根目录。

但 `ROGII_DATA` 只控制第二版 CFG。第一条轨道仍读取单元格 6 固定的 `COMPETITION_DATA_ROOT`，所以原 Notebook 尚不是本地一键复现入口。

### 1.2 每口井读取

代码位置：

- 单元格 11：`load_well(wid, split)`
- 单元格 33：第二个同名 `load_well(wid, split)`

输入：

    <well_id>__horizontal_well.csv
    <well_id>__typewell.csv

输出：

    hw: 一口水平井 DataFrame
    tw: 一口 typewell DataFrame

### 1.3 当前数据规模

| 数据 | 井数 | 文件数 |
|---|---:|---:|
| 训练 horizontal | 773 | 773 |
| 训练 typewell | 773 | 773 |
| 测试 horizontal | 3 | 3 |
| 测试 typewell | 3 | 3 |
| sample submission | 3 口井 | 14,151 行 |

三个测试井隐藏行：

| 井 | 可见行 | 隐藏/评价行 |
|---|---:|---:|
| `000d7d20` | 1,442 | 3,836 |
| `00bbac68` | 1,545 | 6,014 |
| `00e12e8b` | 2,083 | 4,301 |
| 合计 | 5,070 | 14,151 |

## 2. 可见前缀和隐藏后缀

代码位置：

- 单元格 11 `run_particle_filter()` 第 58～59 行；
- 单元格 14 `build_well()` 第 394 行；
- 单元格 37 `lik_pf()` 第 59 行。

共同写法：

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]

含义：

- `kn` 是 known prefix，可见前缀；
- `ev` 是 evaluation suffix，隐藏后缀。

最后一个可见点提供：

- `last_known_tvt`；
- `last_Z`；
- `last_MD`；
- PF 的初始地层坐标；
- carry-forward 基线；
- 所有增量模型的零点。

carry-forward 基线是：

    prediction_i = last_known_tvt

它没有在当前 Notebook 中被单独保存为一份完整基线报告，但它是多个 fallback 和 carry hold 的中心。

## 3. GR 与 typewell TVT 数据如何使用

### 3.1 typewell

PF 会把 typewell 按 TVT 排序：

    tw_tvt = sorted typewell TVT
    tw_gr  = corresponding typewell GR

它把 typewell 看作一个查询表：

    expected_gr = interpolate(typewell_gr, at=particle_tvt)

### 3.2 水平井 GR

水平井 GR 可能缺失。selector PF 使用：

    hw["GR"].interpolate(limit_direction="both")

若仍为空，再用 typewell GR 均值填充。

### 3.3 GR 观测尺度

在可见前缀中：

    typewell_gr_at_known = interp(TVT_input, tw_tvt, tw_gr)
    residual = horizontal_gr - typewell_gr_at_known
    gr_sigma = clip(std(residual), 10, 60)

这个 `gr_sigma` 控制“GR 差多少才算严重”。它和随机种子融合的 `scale` 不是同一个参数。

## 4. 第一条残差模型轨道

代码位置：

- 单元格 14：`build_well()`、`build_dataset()`
- 单元格 15～25：数据、五模型、Ridge、后处理和 `sub_1`

### 4.1 训练特征

`build_well()` 只保留隐藏段行，并为每行构造：

- PF-ANCC 和 PF-Z 路径；
- 多组 Beam 路径；
- 多尺度 NCC；
- 六个地层面 KNN；
- dense ANCC 邻井特征；
- GR 平滑、差分、滞后和领先；
- MD/X/Y/Z 轨迹变化；
- 预测位置与 typewell GR 的差；
- 可见前缀长度、GR 拟合误差等井级特征。

完整特征字典将在 `04_feature_dictionary.md` 中逐项展开。

### 4.2 训练目标

代码位置：单元格 14，`build_well()` 第 559～561 行。

    target = hidden_true_tvt - last_known_tvt

模型学习的是“相对最后可见 TVT 要移动多少”，不是直接学习绝对 TVT。

### 4.3 当前训练数据来源

单元格 15 优先读取：

    Wellbore Geology Prediction  Artifacts/data/train.csv

只有缓存不存在才对 773 口井重新调用 `build_dataset()`。

当前运行读取缓存。该缓存约 7.39 GB，但 Notebook 没有打印完整配置指纹、创建时间或代码提交哈希。这是复现风险，不能把“成功读取”当成“缓存来源已经确认”。

### 4.4 Fold

第一版 CFG：

    GroupKFold(n_splits=5)

groups 是：

    g = train_df["well"]

所以同一口井的隐藏行不会同时出现在训练折和验证折。

但是当前代码是普通 `GroupKFold`，没有证据表明它就是 AGENTS.md 要求的固定 spatial pad 分组。后续复现前必须找到或重建固定 pad 定义。

### 4.5 五个基础模型和 Ridge

当前实际加载：

- 3 个 LightGBM；
- 2 个 CatBoost。

它们保存了 OOF 预测和五折模型。Notebook 用五组 OOF 预测训练：

    Ridge(alpha=1.660283..., positive=True, fit_intercept=True)

Ridge 也按 `well` 做 5 折。

Notebook 日志中的 Ridge fold RMSE：

| fold | RMSE |
|---:|---:|
| 0 | 9.4973 |
| 1 | 10.8118 |
| 2 | 9.3836 |
| 3 | 11.4277 |
| 4 | 10.8220 |

总体行级 OOF RMSE：

    10.4197

这是 Ridge 增量轨道的 OOF 记录，不是当前最终多层提交的 CV。

### 4.6 PF 增量进入残差轨道

代码位置：单元格 22～24。

    ridge_delta = Ridge 输出
    pf_delta = pf_ancc - last_known_tvt
    mixed_delta = 0.85 * ridge_delta + 0.15 * pf_delta
    warmup = 1 - exp(-md_since / 85)
    final_delta = 1.0 * warmup * mixed_delta
    tvt = last_known_tvt + final_delta

随后单元格 24 对每口井用窗口 17、三次多项式的 Savitzky-Golay 滤波平滑。

输出：

    sub_1[id, tvt]

## 5. selector PF 数据流

代码位置：

- 单元格 11：PF、Beam、selector；
- 单元格 26：逐测试井调用。

### 5.1 粒子初始化

对可见前缀最后 30 行估计：

    initial_rate = median((delta_TVT + delta_Z) / delta_MD)

定义地层坐标：

    U = TVT + Z

最后可见地层坐标：

    U_last = last_known_tvt + last_Z

每个粒子：

    particle_U = U_last + 4.5 * Normal(0, 1)
    particle_rate = initial_rate + 0.01 * Normal(0, 1)

当前一次运行有 500 个粒子。

### 5.2 状态更新

隐藏段每前进一个 MD 步长：

    rate = 0.998 * rate + 0.002 * Normal(0, 1)
    U = U + rate * delta_MD + 0.005 * Normal(0, 1)
    particle_TVT = U - current_Z

TVT 被限制在 typewell TVT 范围外加减 100 ft 内。

### 5.3 观测似然

对每个粒子：

    expected_GR = typewell_GR(particle_TVT)
    normalized_error = (horizontal_GR - expected_GR) / gr_sigma
    likelihood = exp(-0.5 * normalized_error²)

误差平方最大截断到 600，似然最小截断到 `1e-300`，防止数值下溢。

### 5.4 重采样

有效粒子数：

    N_eff = 1 / sum(weight²)

当前阈值：

    N_eff < 0.5 * 500 = 250

触发后：

- 高权重粒子会被复制；
- 低权重粒子会被淘汰；
- 新位置加 0.1 ft 噪声；
- 新速度加 0.001 噪声；
- 所有权重重置为 `1/500`。

### 5.5 单个随机种子输出

每一行输出：

    predicted_TVT = sum(weight * particle_TVT)

同时累加整条路径的：

    log_likelihood

### 5.6 128 个随机种子

`run_pf_lik_ensemble_scales()` 用 seed 0～127 重复完整 PF：

    predictions shape = [128, 整口井行数]
    likelihoods shape = [128]

对每个 `scale`：

    seed_weight_s =
        exp((loglik_s - max_loglik) / scale)
        / sum_all_seeds

    ensemble_prediction =
        sum(seed_weight_s * seed_prediction_s)

当前 scale：

    3, 5, 8, 12

scale 越小，越集中相信最高似然的少数随机种子；scale 越大，128 个随机种子越接近平均。

## 6. Beam、selector 分箱和 carry hold

### 6.1 Beam

`run_beam_ensemble()` 使用 14 组 `(beam_size, move_cost, error_scale, smoothing_radius)`。

Beam 每一步只允许 typewell 索引移动：

    -2, -1, 0, +1, +2

代价由：

- 水平井 GR 与 typewell GR 的平方差；
- 移动惩罚；
- 历史累计代价

组成。14 条路径取平均得到 `tvt_beam`。

### 6.2 按井分箱

`selector_well_code()` 只使用：

- 隐藏行数量 `n_eval`；
- 隐藏段 Z 跨度 `z_span`。

阈值：

    n_eval > 4840
    z_span thresholds = 136.73, 185.5133

由此选择不同 PF scale、Beam 权重和 hold 权重。

当前实际日志：

| 井 | variant |
|---|---|
| `000d7d20` | `pf_scale_5_hold_0.2` |
| `00bbac68` | `pf_scale_5_hold_0.15` |
| `00e12e8b` | `pf_scale_12_beam_0.2_hold_0.15` |

这与 Notebook Markdown 中概括的全局 `pf_scale_8_hold_0.2` 不完全一致。实际执行日志和 `SELECTOR_BIN_VARIANTS` 才是当前运行证据。

### 6.3 PF 与 Beam

    selector_before_hold =
        (1 - beam_weight) * selected_pf
        + beam_weight * tvt_beam

### 6.4 carry hold

    selector_after_hold =
        (1 - hold_weight) * selector_before_hold
        + hold_weight * last_known_tvt

这不是“只在开头 hold 几行”，而是整条隐藏路径都向最后可见 TVT 收缩。

例如：

    selector_before_hold = 120
    last_known_tvt = 100
    hold_weight = 0.20

则：

    selector_after_hold = 0.8 * 120 + 0.2 * 100 = 116

## 7. 同井 physical 路径

代码位置：

- 单元格 11：`tvt_from_contacts()`
- 单元格 26：逐井优先选择 `tvt_phys`

默认参考接触面：

    EGFDU

从训练 typewell 找到 EGFDU 接触的最小 TVT：

    ref_tvt

构造原始轨迹：

    raw_tvt_i = ref_tvt - (Z_i - EGFDU_i)

再用同一训练井完整真实 TVT 计算：

    offset = mean(train_TVT_i - raw_tvt_i)

最终：

    tvt_phys_i = raw_tvt_i + offset

单元格 26 的选择条件：

    if same_named_train_well_exists:
        sub_2 = tvt_phys
    else:
        sub_2 = tvt_selector

历史高分运行的 3 口测试井都有同名训练副本，因此旧 `sub_2` 没有使用 PF/Beam selector 数值。当前源码新增 `RUN_SAME_WELL_CONTACT_OVERRIDE=False`，所以 `tvt_phys` 不再构造，`sub_2` 使用 PF/Beam selector。

## 8. 第一条 anchor 与 `U=TVT+Z` 投影

### 8.1 0.3/0.7 anchor

单元格 28：

    anchor_raw = 0.30 * sub_1 + 0.70 * sub_2

当前 private-safe 源码中，`sub_2` 是 PF/Beam selector；“同井 physical”只描述 Notebook 保存的旧运行。

### 8.2 为什么投影 U

定义：

    U = TVT + Z

如果只对 TVT 平滑，井轨迹 Z 的上下变化和地层面的变化会混在一起。对 U 建模相当于先把当前井的垂向轨迹影响拆出来。

### 8.3 投影步骤

单元格 30：

1. 取最后可见 `U_last=last_TVT+last_Z`；
2. 把隐藏段 MD 归一化到 0～1；
3. 计算 `anchor_raw_TVT + Z - U_last`；
4. 迭代四次稳健加权拟合；
5. 使用四次多项式；
6. 还原 `projected_TVT = U_last + fitted_delta_U - Z`；
7. 最终：

       sp45 = 0.25 * anchor_raw + 0.75 * projected_TVT

输出：

    sp45_projection_submission.csv

## 9. 第二条 learned trajectory

代码位置：单元格 33～43。

### 9.1 重新构造测试特征

对每口测试井计算：

- likelihood PF scale 3/5/8/12；
- PF-ANCC、PF-Z；
- Beam；
- NCC；
- 地层面和邻井；
- GR 和轨迹特征。

### 9.2 预训练模型

单元格 43 实际加载：

    rogii-claude-models-pub/lgb0.pkl
    rogii-claude-models-pub/lgb1.pkl
    rogii-claude-models-pub/lgb2.pkl

三者预测取平均，得到 `model_delta`。

### 9.3 learned 模型与 likelihood PF 的融合

`make_prediction()` 当前参数：

    PP.alpha = 1.0
    PP.tau = 85
    PP.w_pf = 0.0
    PP.w_sub1 = 0.60
    PP.sub2_scale = "scale_5"

先对模型增量做 warm-up：

    model_path = last + warmup * model_delta

然后：

    learned_path =
        last
        + 0.60 * warmup_model_delta
        + 0.40 * (likelihood_pf_scale5 - last)

最后逐井用窗口 61、三次多项式 Savitzky-Golay 平滑。

`PP.w_pf=0` 不代表第二条轨道没有 PF。它只表示不再额外混入单粒子 `pf_ancc`；40% 的 likelihood PF scale5 仍直接进入结果，PF 也已作为模型特征。

## 10. 双轨最终基础路径

代码位置：单元格 45。

    base_dual_track =
        0.55 * sp45_projection
        + 0.45 * learned_path

两个文件按 `id` inner merge。若行数或 ID 不完全匹配，代码报错，不允许按行号静默融合。

## 11. guarded same-well contact 覆盖（当前关闭）

代码位置：单元格 48。

这次不直接相信同名训练井，而是增加可见前缀 guard：

1. 在训练井上构造 EGFDU 物理轨迹；
2. 按 MD 插值到测试井可见前缀；
3. 至少要有 50 行；
4. 前缀 RMSE 必须不超过 1 ft；
5. 只覆盖训练 MD 范围内的隐藏行。

旧运行中 3 口井全部通过并覆盖 14,151 行。当前 `RUN_GUARDED_OVERLAP_OVERRIDE=False`，这一层不会改写 `submission.csv`。

## 12. 可见前缀伪 holdout

代码位置：单元格 50。

对每个 cut：

    cut fractions = 0.50, 0.65, 0.75

做法：

1. 只保留可见前缀的早期部分；
2. 把后半个已知前缀暂时遮住；
3. 用缩短后的前缀重建候选；
4. 在被遮住但真实已知的部分计算 RMSE；
5. 对三个 cut 的 RMSE 取中位数，再加 0.1 倍标准差；
6. 只有 gain、margin 和一致性通过门槛才移动；
7. 移动权重上限 0.40，单点最大裁剪 30 ft。

旧运行中 3 口井最佳都是 `contact_md_lookup`。当前源码只有在 `_GOLD_CONTACT_OVERRIDE=True` 时才把 contact 候选加入候选池；默认 False，因此不会利用同井完整标签。

这一步只用可见前缀 TVT 做伪验证，原则上是合法的 same-well calibration。它与“读取同井训练完整 TVT”必须分开评价。

## 13. 最终模型包

代码位置：单元格 52。

输入：

- 当前 `submission.csv`；
- 原始训练/测试井；
- 480 个特征；
- XGB、CatBoost、HGB、LGB、TCN 全训练模型；
- 包内融合配置。

模型包预测记为 `model_tvt`，当前基础路径记为 `base_tvt`。

分歧门控：

    diff = abs(model_tvt - base_tvt)
    gate = 0.005 / (1 + (diff / 6)²)
    final = (1 - gate) * base_tvt + gate * model_tvt

差异越大，模型包权重越小。

当前日志：

- 平均 gate：0.000949；
- 95% gate：0.002114；
- 最大 gate：0.004836；
- 平均最终移动：0.01067 ft；
- 最大最终移动：0.015 ft。

## 14. 最终 TVT 与提交审计

单元格 53 检查：

    columns == ["id", "tvt"]
    len(submission) == len(sample_submission)
    submission.id order == sample_submission.id order
    all finite(tvt)

当前日志：

| 项目 | 值 |
|---|---:|
| 行数 | 14,151 |
| TVT 最小值 | 11,587.0248 |
| TVT 最大值 | 12,240.0025 |
| TVT 均值 | 11,903.6194 |

## 15. CV 和 RMSE 在哪里

### 15.1 第一条模型轨道

- 代码：单元格 8 fallback Trainer、单元格 10、18、19、21、23；
- Fold：`GroupKFold(5)`；
- group：井 ID；
- 指标：`sklearn.metrics.root_mean_squared_error`；
- overall：把所有 OOF 行放在一起计算 micro RMSE。

公式：

    micro_RMSE = sqrt(sum_all_rows(error²) / total_rows)

### 15.2 selector PF 报告

代码：单元格 12 `_rogii_selector_cv_report()`。

它：

- 遍历训练井；
- 在每口井原有 `TVT_input.isna()` 隐藏段上预测；
- 用训练文件的真实 `TVT` 评分；
- 汇总 pooled RMSE、逐井 P50/P90/P99。

当前 `RUN_CV_REPORT=False`，所以这次提交运行没有重新生成 selector CV。

它也不是 5-fold 训练过程：PF 本身不拟合跨井参数，只是在训练井的自然隐藏段上做回放评分。

### 15.3 第二条 learned trajectory

如果预训练模型不存在，单元格 41 `train_stack()` 才会：

- 按井 GroupKFold(5)；
- 训练 3 个 LightGBM + 2 个 CatBoost；
- 再用 Ridge 做 OOF stacking。

当前运行直接加载模型，因此没有重新训练或重新验证。

### 15.4 模型包

manifest 记录 5 折、3,783,989 OOF 行和 10.6702 后处理 OOF RMSE。

但其 imputer 没有为每个验证折完全重建，后续不能把这个分数直接当作严格 outer-fold spatial CV。

## 16. 合法输入与泄漏审计

| 信息源 | 测试文件直接可得 | 对新井 PF 推理是否合法 | 当前风险 |
|---|---:|---:|---|
| 测试 `MD/X/Y/Z/GR/TVT_input` | 是 | 是 | GR 缺失和分布偏移 |
| 测试 typewell `TVT/GR` | 是 | 是 | 多解和 GR 对齐 |
| 可见前缀内部伪 holdout | 是 | 是 | 前缀不一定代表远端隐藏段 |
| 训练井构建的邻井特征 | 是，训练数据可用 | 只有 fold 内重建才合法 | 当前缓存指纹不完整 |
| 同名训练井完整 `TVT` | 当前恰好存在 | 对通用新井基线不合法 | 与测试井是同一底层井，等价于利用同井标签 |
| 同名训练井六个地层面 | 当前恰好存在 | 对通用新井基线不合法 | 测试文件没有这些列 |
| 测试隐藏 `TVT` | 测试文件没有 | 禁止 | 代码未直接从测试文件读取 |

最重要的区别：

- 代码没有直接读取测试 CSV 中不存在的隐藏 `TVT`；
- 但它读取了完全相同井号训练副本的完整 `TVT` 和地层面；
- 对当前 3 井提交非常有效；
- 对“新井泛化”和严格 PF 基线而言，应标记为同井标签复用，不能混入合法 PF CV。

## 17. 数据直接证明、推断和未知

### 数据直接证明的事实

- 测试 3 口井全部有同名训练副本。
- 共享的 `MD/X/Y/Z/GR/TVT_input` 完全一致。
- Notebook 保存的旧输出中，单元格 26 的 `sub_2` 使用 `tvt_phys`。
- 旧输出中单元格 48 对 14,151 行全部执行了同井覆盖。
- 当前源码的三个同井开关均为 False，且空间 imputer 对查询井自排除。
- private-safe 修改后的分数和提交尚未运行，不能从旧输出推断。

### 基于事实的合理推断

- 历史最高提交主要由 EGFDU 同井接触面重建决定。
- private-safe 版本将更依赖 PF/Beam 和 learned 特征，分数大概率与旧提交不同。
- 历史高分不能直接证明 PF 本身达到同样分数。

### 仍然没有验证的猜测

- `ROGII - 03/9.349.csv` 是否由当前 profile 生成。
- 当前 Notebook 对真正未知、没有训练副本的新测试井会得到什么排行榜分数。
- 所谓“PF 10.7”究竟指 selector PF、第一条残差轨道还是模型包 10.6702。

### 当前代码只能否定或证明的具体实现

当前只读考古不能否定 PF、GR 或邻井方向。它只能证明最终提交不是一个纯 PF 实验。

### 下一步最便宜的验证

不运行大实验，先：

1. 固定一口训练井；
2. 只运行 `run_particle_filter()`；
3. 保存 carry-forward、单 seed PF、24/128 seed PF；
4. 在同一隐藏 mask 上复算 RMSE；
5. 明确禁止 `tvt_from_contacts()` 和同井地层列。

这一步需要先写实验卡，再运行。
