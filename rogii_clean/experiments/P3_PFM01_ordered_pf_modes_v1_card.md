# P3-PFM01 实验卡：PF128 高/中/低有方向模式

实验编号：`P3-PFM01_ordered_pf_modes_v1`

实验名称：把 128 条 PF seed 路径按绝对位置分成 low / middle / high 三个统一方向模式

唯一假设：128 条 PF 路径不是无意义随机抖动，而是包含“TVT 偏低 / 居中 / 偏高”的多分支；真实 GR 似然在高低模式上的质量差可以判断严格 P2-P02 的平均残差方向。

为什么值得验证：R01b 和 N01m 都表明普通井级静态特征或纯空间邻井不能预测平均残差；PF128 路径本身直接表示当前井在 GR 证据下的多个合法解释，按位置命名后才具有跨井一致的方向含义。

与当前基础路径唯一不同之处：不修改 P2-P02、LightGBM 或 PF seed 路径；只重新组织已经合法生成的 128 条路径，做只读方向与候选覆盖审计。

合法输入：

- outer0 严格 dev-only P2-P02 路径，SHA256 `cb0d1b778a2193507204fd6b104efbdb4fd44f20a4a96d3f9eff139759df9bad`；
- 每井 128 条 PF seed 逐行路径、最终 LL、seed_id、row_index、hidden_md、last_tvt；
- 旧五条温度聚合路径 `mean / scale3 / scale5 / scale8 / scale12`；
- 原始水平井中测试时可获得的 GR 观测率，仅用于切片。

固定 fold：outer fold 0 的 131 口开发井、651,881 条自然隐藏行；影子井不参与。

## 前置缓存生成

当前共享 PF128 缓存只覆盖 46/131 口 outer0。先使用原 PF03 冻结配置，以 `generation-only fold0` 模式原子补齐其余 85 口：

- 128 seeds；
- 8 线程；
- 共享缓存指纹 `91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0`；
- 逐井已有缓存严格核验后复用；
- 生成阶段不读取任何隐藏 target TVT，不做 PF03 oracle 评分；
- 中断后重新运行同一命令，已验证的单井缓存继续命中。

## 固定三模式方法

每条 seed 只用两个无标签位置量：

\[
x_s=[mean(\Delta TVT_s),\ \Delta TVT_{s,end}]
\]

对 `128×2` 数组使用确定性的 Ward 层次聚类：`K=3`、`optimal_ordering=False`、不做 z-score、不调参数。

最终 LL 用固定温度 8 转成 seed 权重：

\[
w_s=softmax((LL_s-max(LL))/8)
\]

模式中心为类内 seed 完整路径的 LL 加权均值，模式质量为类内权重之和。

三个中心按以下固定顺序重命名：

```text
中心路径整井平均 TVT 升序
→ 末端 TVT 升序
→ 最小 seed_id 升序
```

依次命名为 `low / middle / high`，绝不按 mass 或 likelihood 名次命名。

合法输出：

```text
pf_mode_low_delta
pf_mode_middle_delta
pf_mode_high_delta
low/middle/high_mode_mass
direction_score = high_mode_mass - low_mode_mass
high_minus_low_separation
high_minus_low_endpoint_separation
p2_position_raw
p2_position_within_mode_envelope
low/middle/high_seed_count
```

若 high-low 平均分离小于 `1e-6 ft`，`p2_position` 固定回退为 `0.5` 并标记退化。

## 负对照

1. 每井保持路径和聚类成员不变，把 128 个 `final_ll` 固定循环移动 64 位，重新计算 mode mass、中心和 `direction_score`；
2. 合法结果全部落盘后，在 oracle 阶段把 131 口真实 `direction_score` 用种子 `20260719` 做无固定点跨井置换。

## 物理隔离

合法阶段只生成 `legal_cache/<well>.parquet`、`legal/per_well.csv` 和逐井 runtime。列名禁止出现 `target / true / residual / error / rmse / oracle / best_path`。131 口合法缓存全部完成后，才允许读取 outer0 `target_tvt`，oracle 结果单独保存到 `oracle/`。

## 方向诊断

outer0 真实平均残差最后定义为：

\[
m=mean(TVT^{true}-TVT^{P2P02,strict})
\]

符号预先冻结：`direction_score>0` 表示更支持高 TVT，应对应 `m>0`。不得看结果后翻转。

报告 Pearson、Spearman；在 `|m|>=2 ft` 上报告 ROC-AUC、阈值 0 的 accuracy 与 balanced accuracy，以及 direction score 五分位和 high-low 分离/GR 观测率/最小模式 seed 数切片。

## 候选覆盖 oracle

比较：

- 三模式 oracle：逐井从 low/middle/high 中选真实 RMSE 最小路径；
- 五温度 oracle：逐井从 mean/scale3/scale5/scale8/scale12 中选真实 RMSE 最小路径。

报告 pooled micro RMSE、三模式优于五温度的井比例、low/middle/high 各自成为最优的比例，以及最强 5% 井的收益占比。oracle 只诊断候选覆盖，不是提交路径。

## 预注册门槛

模式可用性：

1. 131 口 seed 缓存全部齐全且无影子井；
2. 至少 80% 的井中三个模式各至少 3 个 seed；
3. 至少 50% 的井 high-low 平均分离 ≥1 ft。

方向信号：

1. `|m|>=2 ft` 的 ROC-AUC ≥0.60；
2. 阈值 0 的 balanced accuracy ≥0.60；
3. AUC 比 LL 循环错配和跨井错配均高至少 0.10；
4. `direction_score` 与 `m` 的 Spearman 为正。

模式候选覆盖：

1. 三模式 oracle pooled RMSE 比五温度 oracle 改善至少 0.10 ft；
2. 三模式 oracle 至少在 55% 的井上更好；
3. 最强 5% 井贡献不超过全部正收益的 60%。

全部通过后才进入“偏低 / 中性 / 偏高”三分类和保守固定幅度修正。本实验本身不训练分类器、不生成提交路径。

停止条件：任一组核心门槛失败，停止当前实现；不在 outer0 上改聚类输入、K、温度、方向符号或模式排序。

预计运行时间：补齐 85 口 PF128 缓存约 5～10 分钟；三模式合法计算和 oracle 预计 1～3 分钟。

失败后只能否定：当前“整井均值 + 末端位置、Ward K=3、scale8 mass”的三模式实现，不能否定全部 PF 多分支信息。
