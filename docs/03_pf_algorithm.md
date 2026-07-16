# 粒子滤波算法：人话、公式与代码对应

## 0. 先用一句人话解释

粒子滤波就是同时维护很多条“可能的地层轨迹”，每走一步就看哪条轨迹对应的 typewell GR 更像当前水平井 GR；像的轨迹保留，不像的轨迹淘汰，最后对剩下的轨迹加权平均。

它的专业名称是 particle filter，也叫序贯蒙特卡洛方法（Sequential Monte Carlo）。

当前最清楚、最适合学习的实现位于：

    rogii-dual-track-prefix-calibrated-geosteering.ipynb
    单元格 11
    run_particle_filter()

版本状态：2026-07-16 起 private-safe 源码关闭同井 physical/contact，并让训练构建的空间 imputer 始终排除查询井。Notebook 旧输出仍显示修改前覆盖结果，必须与当前算法源码分开阅读。

多随机种子汇总位于：

    单元格 11
    run_pf_lik_ensemble_scales()

PF、Beam 和 carry hold 的组合位于：

    单元格 11
    apply_selector_variant()

## 1. 它解决什么问题

已知：

- 水平井前缀的真实 TVT；
- 整口井的 MD 和 Z；
- 水平井 GR；
- typewell 的 TVT-GR 参考曲线。

未知：

- 水平井隐藏后缀的 TVT。

困难在于：

- GR 会有噪声和缺失；
- 相似的 GR 形状可能在 typewell 多处重复；
- 隐藏段很长，早期一点速度误差会不断累积；
- 井轨迹 Z 变化会直接影响 TVT；
- 只沿着“当前最优点”贪心走，容易跳到错误地层分支。

PF 用很多候选同时保留不确定性，避免过早只押一条路径。

## 2. 输入和输出

### 2.1 `run_particle_filter()` 输入

| 参数 | 类型/shape | 含义 |
|---|---|---|
| `hw` | DataFrame，`[井行数, 水平井列数]` | 一口水平井 |
| `tw` | DataFrame，`[typewell 行数, 2或3]` | 一口参考直井 |
| `n_particles` | 整数，当前 500 | 同时维护的假设数量 |
| `seed` | 整数 | 控制本次随机数 |

PF 实际使用的水平井列：

- `MD`
- `Z`
- `GR`
- `TVT_input`

PF 实际使用的 typewell 列：

- `TVT`
- `GR`

它不需要：

- 测试隐藏 `TVT`；
- 六个训练地层面；
- 其他井的标签。

所以单独的 selector PF 是一个合法测试时算法。

### 2.2 输出

`run_particle_filter()` 返回：

| 输出 | shape | 含义 |
|---|---|---|
| `out_vals` | `[整口水平井行数]` | 可见段保留 `TVT_input`，隐藏段填 PF 预测 |
| `log_lik` | 标量 | 这次随机运行整条隐藏路径的累计对数证据 |

`run_pf_lik_ensemble_scales()` 返回字典：

    pf_scale_3
    pf_scale_5
    pf_scale_8
    pf_scale_12
    pf_mean

每个值都是一条整井长度的 TVT 路径。

## 3. 一个粒子代表什么

一个粒子有三个核心量：

| 代码变量 | shape | 人话 | 专业含义 |
|---|---|---|---|
| `pos[j]` | 标量；所有粒子合起来 `[N]` | 第 j 个猜测的地层位置 | 状态位置 `U=TVT+Z` |
| `rate[j]` | 标量；合起来 `[N]` | 这个地层位置沿 MD 的移动速度 | 状态速度 `dU/dMD` |
| `w[j]` | 标量；合起来 `[N]` | 当前有多相信这个猜测 | 后验权重 |

一个粒子不是一行数据，也不是一口井。

它表示：

> 从最后一个可见点出发，如果地层面当前位置是 U、接下来按 rate 演化，那么隐藏段可能沿着这条路线走。

500 个粒子就是 500 个同时存在的局部假设。

## 4. 为什么状态用 `U=TVT+Z`

代码定义：

    U = TVT + Z

所以：

    TVT = U - Z

人话解释：

水平井自己的 Z 会变化。如果直接预测 TVT，井轨迹运动和地层面运动混在一起。用 U 表示地层面的空间坐标，先更新地层面，再减去当前井的 Z，能把两种运动分开。

一个简单例子：

    last_TVT = 11200
    last_Z = -9300
    U_last = 11200 + (-9300) = 1900

下一行假设粒子 U 移到 1900.02，而井的 Z 变成 -9301：

    TVT = 1900.02 - (-9301) = 11201.02

即使地层面 U 只移动 0.02 ft，井本身向下 1 ft，也会让 TVT 增加约 1.02 ft。

## 5. 算法完整步骤

## 5.1 准备 typewell

代码位置：单元格 11，`run_particle_filter()` 第 54～56 行。

    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"]
    tw_gr = tw_s["GR"]

缺失 typewell GR 用 typewell GR 均值填充。

作用：

后面可以用线性插值回答：

> 如果粒子认为当前 TVT 是 11250 ft，那么 typewell 在 11250 ft 的 GR 应该是多少？

## 5.2 划分可见和隐藏段

代码位置：第 58～59 行。

    kn = hw[TVT_input not null]
    ev = hw[TVT_input is null]

若没有隐藏段，直接返回原 `TVT_input`。

## 5.3 找到最后一个可见锚点

代码位置：第 63～66 行。

取：

    last_tvt
    last_Z
    last_MD

并计算：

    U_last = last_tvt + last_Z

## 5.4 从可见前缀估计 GR 噪声

代码位置：第 68～69 行。

    tw_at_known = interp(known_TVT, typewell_TVT, typewell_GR)
    gr_sigma = std(horizontal_GR - tw_at_known)
    gr_sigma = clip(gr_sigma, 10, 60)

人话解释：

如果这口井在已知区里，水平井 GR 和 typewell GR 本来就差得很大，那么隐藏区不能因为几单位 GR 差就强烈淘汰粒子。

专业名称：

这是观测噪声尺度 observation noise scale。

代码风险：

`kn["GR"].fillna(0)` 会把可见前缀缺失 GR 当 0 参与标准差。若缺失很多，可能把 `gr_sigma` 放大。后续干净实现应明确只用同时有限的 GR 行计算。

## 5.5 从最后 30 行估计初始速度

代码位置：第 71～76 行。

因为：

    U = TVT + Z

所以：

    delta_U = delta_TVT + delta_Z

速度估计：

    initial_rate =
        median((delta_TVT + delta_Z) / delta_MD)

只取最后 30 个可见点，是因为离预测起点最近的趋势通常最有参考价值。

如果有效差分少于 3 个，速度回退到 0。

## 5.6 初始化 500 个粒子

代码位置：第 78～85 行。

    N = 500
    particle_U = U_last + 4.5 * Normal(0, 1)
    particle_rate = initial_rate + 0.01 * Normal(0, 1)
    weight = 1 / 500

含义：

- 地层位置初始标准差约 4.5 ft；
- 速度初始标准差约 0.01 ft/ft；
- 刚开始没有证据偏爱任何粒子，所以等权。

## 5.7 填补水平井 GR 缺口

代码位置：第 89～91 行。

先沿行方向线性插值，再向前/向后填充；若仍缺失，用 typewell GR 均值。

输出 `gr_v` shape：

    [隐藏行数量]

## 5.8 状态转移：粒子如何移动

代码位置：第 98～105 行。

每个隐藏点：

    delta_MD = max(current_MD - previous_MD, 1)

速度更新：

    rate_new =
        0.998 * rate_old
        + 0.002 * Normal(0, 1)

位置更新：

    U_new =
        U_old
        + rate_new * delta_MD
        + 0.005 * Normal(0, 1)

再转换：

    particle_TVT = U_new - current_Z

最后把粒子 TVT 限制在：

    [typewell_min_TVT - 100,
     typewell_max_TVT + 100]

人话解释：

- `0.998` 让速度大体延续；
- `0.002` 允许速度慢慢变化；
- `0.005` 允许位置有小扰动；
- 限制范围防止粒子漂到离 typewell 太远的数值。

专业名称：

这是状态转移模型 state transition model。

## 5.9 观测似然：GR 如何给粒子打分

代码位置：第 106～114 行。

对每个粒子查询 typewell GR：

    expected_GR_j = interp(particle_TVT_j, tw_tvt, tw_gr)

标准化误差：

    d_j = (observed_horizontal_GR - expected_GR_j) / gr_sigma

似然：

    likelihood_j = exp(-0.5 * min(d_j², 600))

权重更新：

    weight_j = weight_j * likelihood_j
    weight = weight / sum(weight)

人话解释：

观测似然就是“当前水平井 GR 对这个粒子有多支持”。

专业名称：

这是 observation likelihood。

当前代码中，它把 GR 误差假设为近似高斯分布。

### 数值例子

假设：

    observed_GR = 80
    gr_sigma = 10

粒子 A：

    expected_GR = 78
    d = (80 - 78) / 10 = 0.2
    likelihood = exp(-0.5 * 0.2²) ≈ 0.980

粒子 B：

    expected_GR = 100
    d = (80 - 100) / 10 = -2
    likelihood = exp(-0.5 * 2²) ≈ 0.135

这一行观测会让粒子 A 的相对权重大约是粒子 B 的 7.2 倍。

## 5.10 路径总似然

代码位置：第 110～111 行。

在权重归一化之前计算：

    average_likelihood =
        sum(old_weight_j * likelihood_j)

累积：

    log_lik += log(average_likelihood)

`log_lik` 用来比较不同随机种子的整条路径，不直接作为某一行 TVT。

使用对数是因为几千个小于 1 的概率连乘会快速下溢为 0。

## 5.11 有效粒子数

代码位置：第 116 行。

    N_eff = 1 / sum(weight²)

若 500 个粒子完全等权：

    N_eff = 500

若几乎所有权重集中在一个粒子：

    N_eff ≈ 1

它衡量当前实际还有多少个不同的有效假设。

## 5.12 重采样

重采样就是淘汰权重很低的粒子，并复制权重高的粒子。

它的专业名称是 particle resampling。

在当前 PF 中，当：

    N_eff < 0.5 * N

也就是 500 粒子时低于 250，触发 systematic resampling。

代码位置：第 117～123 行。

重采样后：

    new_U = selected_U + 0.1 * Normal(0, 1)
    new_rate = selected_rate + 0.001 * Normal(0, 1)
    new_weight = 1 / N

位置和速度加小噪声，是为了避免复制出的粒子完全相同。

### 四粒子例子

假设权重：

    [0.7, 0.1, 0.1, 0.1]

则：

    N_eff = 1 / (0.7² + 0.1² + 0.1² + 0.1²)
          = 1 / 0.52
          ≈ 1.92

四粒子的阈值是：

    0.5 * 4 = 2

因为 `1.92 < 2`，触发重采样。高权重粒子可能被复制 2～3 次。

## 5.13 每一行的 TVT 输出

代码位置：第 125 行。

    predicted_TVT_i =
        sum(weight_j * particle_TVT_j)

这叫后验加权均值。

它比直接选择最高权重粒子更平滑，但在两个相距很远的同等可能分支中，平均值可能落在两条真实分支之间。这是 PF 在二峰问题上的一个失败模式。

## 6. 多随机种子到底产生什么

### 6.1 一个 seed 不是一个粒子

当前主 selector：

    1 个 seed = 500 个粒子跑完整个隐藏段
    128 个 seed = 128 次独立的 500 粒子 PF

所以不是总共 128 个粒子，也不是简单的 `128 × 500` 粒子一次运行。

每个 seed 都生成：

- 一整条 TVT 路径；
- 一个整条路径的 `log_lik`。

### 6.2 为什么需要多个 seed

PF 里面有随机初始化、状态噪声和重采样。

单次运行可能：

- 偶然早早丢失正确分支；
- 偶然在重复 GR 中跳错位置；
- 因重采样随机性产生漂移。

多 seed 让这种随机误差平均掉，并用路径似然多相信表现更好的运行。

## 7. scale 如何加权随机种子

代码位置：单元格 11，`run_pf_lik_ensemble_scales()` 第 157～162 行。

先平移：

    normalized_loglik_s =
        loglik_s - max(loglik)

再计算：

    seed_weight_s =
        exp(normalized_loglik_s / scale)

最后归一化。

### scale 增大

- 权重更平均；
- 不会只押最高似然 seed；
- 路径更稳健，但可能保留较差 seed。

### scale 减小

- 权重集中到少数最高似然 seed；
- 更敢相信最佳路径；
- 若似然评价有偏，容易过度相信错误分支。

### 两个 seed 的例子

假设：

    loglik_A = -100
    loglik_B = -104

scale=3：

    weights ≈ [0.791, 0.209]

scale=12：

    weights ≈ [0.583, 0.417]

scale 变大后，较差的 B 得到更多权重。

当前同时保存 scale 3、5、8、12，是为了让 selector 或下游模型选择不同的“相信最佳 seed 的激进程度”。

## 8. selector 如何加入 Beam 和 carry hold

代码位置：单元格 11，`apply_selector_variant()`。

### 8.1 先选 PF scale

例如：

    pf_scale_5_hold_0.2

表示先取 `pf_scale_5`。

### 8.2 可选 Beam

例如：

    pf_scale_12_beam_0.2_hold_0.15

先做：

    pred =
        0.8 * pf_scale_12
        + 0.2 * beam_path

### 8.3 carry hold

carry hold 就是把整条预测路径向“最后一个已知 TVT 的常数延续”收缩。

它的专业名称可以叫 shrinkage to carry-forward baseline。

代码：

    pred =
        (1 - hold_weight) * pred
        + hold_weight * last_known_tvt

hold 增大：

- TVT 路径变化幅度变小；
- 长隐藏段不容易漂太远；
- 也可能压掉真实地层变化。

hold 减小：

- 更相信 PF/Beam；
- 能跟随真实变化；
- 错误分支的影响也更大。

carry hold 和 warm-up 不同：

- carry hold 在整条隐藏段都收缩；
- warm-up 只让起点附近的增量逐渐放开。

## 9. 24 个随机种子是什么

当前 Notebook 有三套容易混淆的 seed 数：

| 位置 | seed 数 | 粒子数 | 作用 |
|---|---:|---:|---|
| 主 selector | 128 | 500 | 构造主 PF scale 3/5/8/12 |
| selector CV | 24 | 500 | 用更低成本回放训练井 |
| visible-prefix calibration | 24 | 350 | 在三个伪 holdout cut 上比较候选 |
| visible-prefix final rebuild | 48 | 350 | 选定候选后更稳定地重建最终候选 |

所以“24 个随机种子”表示：

> 用 24 次独立的完整 PF 随机运行来估计候选路径，而不是 24 个粒子。

在 visible-prefix calibration 中，每次运行内部仍有 350 个粒子。

24 比 128 快，适合大量候选筛选；48 用于最终重建，在成本和稳定性之间折中。

## 10. PF 后面的平滑在哪里

selector `run_particle_filter()` 内部没有对最终 TVT 做 Savitzky-Golay 平滑。

项目里有多种“看起来像平滑”的操作，必须区分：

| 操作 | 位置 | 是否 PF 内部 |
|---|---|---:|
| 粒子加权平均 | 单元格 11 第 125 行 | 是 |
| 速度动量 `0.998` | 状态更新 | 是 |
| 重采样后粗化噪声 | PF 重采样 | 是 |
| Beam 输入 GR 的 Savitzky-Golay | `beam_search()` | 否，是 Beam |
| 第一条模型轨道窗口 17 平滑 | 单元格 22～24 | 否 |
| `U=TVT+Z` 四次投影 | 单元格 30 | 否 |
| learned trajectory 窗口 61 平滑 | 单元格 42 | 否 |
| carry hold | `apply_selector_variant()` | 否，是后组合收缩 |

因此不能把最终路径很平滑全部归功于 PF。

## 11. 项目里为什么有多份 PF

| 实现 | 代码位置 | 粒子/seed | 主要职责 | 输出范围 |
|---|---|---|---|---|
| `run_particle_filter()` | 单元格 11 | 500 粒子，单 seed | 最清楚的 selector 教学版 | 整井 |
| `run_pf_lik_ensemble_scales()` | 单元格 11 | 500 × 128 seed | selector 多 seed 汇总 | 整井字典 |
| `_pf_ancc()` / `run_pf_ancc()` | 单元格 14 | 默认 600 | 第一条树模型特征 | 隐藏段 |
| `_pf_z()` / `run_pf_z()` | 单元格 14 | 默认 600 | 加入 Z-velocity 的候选特征 | 隐藏段 |
| 单元格 36 的 Numba 版本 | 单元格 36 | 默认 600 | 第二条 learned 轨道特征 | 隐藏段 |
| `_pf_lik_allseeds()` / `lik_pf()` | 单元格 37 | 500 × 128 seed | 第二条轨道的 workhorse PF | 隐藏段字典 |
| model package filter features | `rogii_feature_core.py` | 独立实现 | 480 特征模型包 | 特征列 |

单元格 11 的 Python 版和单元格 37 的 Numba 版数学思想接近，但不是逐行完全相同：

- 单元格 11 返回整井路径；
- 单元格 37 返回隐藏段路径；
- 单元格 11 第一隐藏步使用真实 `last_MD`；
- 单元格 37 把第一隐藏步的前一 MD 设为 `first_hidden_MD - 1`；
- 单元格 37 为速度使用 Numba 循环以加速 128 seed。

复现时必须先选定一个权威实现，不能混着改。

## 12. PF 为什么可能有效

1. **几何连续性**
   地层位置和速度通常不会每一英尺剧烈跳变，状态转移模型提供连续性。

2. **GR 约束**
   typewell 提供 TVT 到 GR 的参考映射，水平井 GR 能不断纠正纯几何外推。

3. **保留多种可能**
   500 个粒子不会在第一行就只选择一个 TVT。

4. **重采样集中算力**
   更多粒子会聚集到和观测一致的地层位置。

5. **多 seed 降低随机性**
   128 次独立运行减少单次重采样运气的影响。

6. **carry hold 防止长程漂移**
   对不稳定井收缩到简单基线。

## 13. PF 会在什么情况下失败

### 13.1 GR 多解

如果 typewell 不同 TVT 位置有相似 GR 形状，粒子可能同时形成两个分支。

失败表现：

- seed 之间差异大；
- 最终加权平均落在两个分支中间；
- 某次重采样错误地消灭正确分支。

### 13.2 水平井与 typewell GR 不同尺度

如果水平井 GR 存在增益和偏移：

    horizontal_GR ≈ alpha * typewell_GR + beta

直接比较会错误惩罚正确 TVT。

Notebook 有 heel calibration 和 bimodal 诊断，但当前 `RUN_BIMODAL_DETECTOR=False`，它们不会改变基础 selector 路径。

### 13.3 GR 大量缺失

线性插值或均值填充会制造并不存在的平滑证据。

### 13.4 初始位置或速度错误

如果最后 30 行不代表后续趋势，粒子可能从错误方向出发。

### 13.5 隐藏段很长

小速度误差会不断累积，状态噪声也会扩大不确定性。

### 13.6 gr_sigma 不合适

- 太小：一两个 GR 异常点就让权重坍缩；
- 太大：GR 几乎不能纠正轨迹。

### 13.7 粒子退化

太少粒子或重采样太晚，会只剩少数有效粒子。

### 13.8 typewell 覆盖范围不足

代码虽然允许超出 typewell TVT 100 ft，但区间外 `np.interp` 实际使用边界 GR，不能提供可靠形状信息。

### 13.9 地质条件不对应

水平井和 typewell 的 GR-地层关系若本身不同，再好的 PF 也只是在匹配错误模板。

## 14. 关键参数的直观影响

完整参数字典将在 `05_parameter_dictionary.md` 中给出。这里先列 PF 核心：

| 参数 | 当前值 | 增大后 | 减小后 |
|---|---:|---|---|
| 粒子数 `N` | 500 | 更稳、更慢、较能保留多分支 | 更快但更易丢分支 |
| seed 数 | 128 | 降低随机性、运行更慢 | 更快但方差更大 |
| 初始位置标准差 | 4.5 ft | 搜索更广、可能更散 | 更相信最后可见位置 |
| 初始速度标准差 | 0.01 | 搜索更多趋势 | 更相信前缀尾部斜率 |
| 动量 `MOM` | 0.998 | 速度更持久、更平滑 | 更快忘记旧速度 |
| 速度噪声 `VN` | 0.002 | 更能转弯，也更抖 | 更接近固定速度 |
| 位置噪声 `PN` | 0.005 ft | 搜索更广、路径更散 | 更确定、更易卡错分支 |
| GR sigma 下/上限 | 10/60 | 上限大时更宽容 GR 差异 | 小时更强依赖 GR |
| 重采样阈值 | 0.5N | 更早、更频繁集中 | 更久保留低权重粒子 |
| 重采样位置粗化 | 0.1 ft | 复制后更多样 | 粒子更容易完全相同 |
| 重采样速度粗化 | 0.001 | 趋势更多样 | 趋势更集中 |
| seed scale | 3/5/8/12 | 大时 seed 权重更平均 | 小时集中最佳 seed |
| hold weight | 0.05～0.20 | 更接近 carry-forward | 更相信 PF/Beam |

## 15. 最简伪代码

    读取水平井和 typewell
    把水平井分成可见前缀 kn 和隐藏后缀 ev
    从可见前缀估计 GR 噪声和 U 速度

    初始化 500 个粒子:
        U 在 U_last 附近随机分布
        rate 在前缀尾部速度附近随机分布
        所有权重相等

    对隐藏段每一行:
        用动量和噪声更新 rate
        用 rate、delta_MD 和噪声更新 U
        用 TVT = U - Z 得到每个粒子的 TVT
        在 typewell 查询每个粒子的预期 GR
        用水平井 GR 计算粒子似然
        更新并归一化粒子权重

        如果有效粒子太少:
            复制高权重粒子
            淘汰低权重粒子
            加少量位置和速度噪声

        当前 TVT = 所有粒子 TVT 的加权平均

    返回一条路径和整条路径 log_likelihood

    重复 128 个随机种子
    用 log_likelihood / scale 计算 seed 权重
    得到 scale 3/5/8/12 四条集成路径
    按井选择 PF scale，并可选混入 Beam
    最后向 carry-forward 做 hold 收缩

## 16. 最应该先读的代码

第一遍只读单元格 11 的这些范围：

1. 第 53～76 行：输入、可见/隐藏、GR sigma、初始速度；
2. 第 78～85 行：粒子初始化和核心参数；
3. 第 98～125 行：状态更新、似然、重采样和输出；
4. 第 132～166 行：多 seed 和 scale；
5. 第 257～278 行：按井分箱和 variant 解析；
6. 第 656～696 行：PF、Beam、hold 的最终组合。

暂时不要先读 bimodal scan、heel calibration 和 104 个 visible-prefix 候选。

## 17. 一个小样本应该如何验证

在真正运行前应写实验卡。最小验证建议：

1. 只选一口训练井；
2. 使用它原本的 `TVT_input.isna()` 作为隐藏段；
3. 禁用同井 contact、地层列和 learned 模型；
4. 保存四条路径：
   - carry-forward；
   - 单 seed PF；
   - 24-seed PF；
   - 128-seed PF；
5. 检查输出行数等于隐藏行数；
6. 检查第一行是否接近最后可见 TVT；
7. 检查没有 NaN/Inf；
8. 用训练 `TVT` 只在最后一步评分；
9. 画出 seed 分歧和 `N_eff`；
10. 确认结果可由固定 seed 重复。

## 18. 学完后的自测

应该能用自己的话回答：

1. 一个粒子代表什么？
2. 为什么粒子状态用 `TVT+Z`？
3. 粒子如何从一行移动到下一行？
4. typewell GR 如何改变粒子权重？
5. `gr_sigma` 和 seed `scale` 有什么区别？
6. 为什么需要重采样？
7. `N_eff` 很小代表什么？
8. 128 个 seed 产生了什么？
9. 24 个 seed 为什么不是 24 个粒子？
10. carry hold 和 warm-up 有什么区别？
11. PF 内部有没有 Savitzky-Golay 平滑？
12. 为什么历史运行的 `sub_2` 没采用 PF，而当前 private-safe 源码会采用 PF/Beam selector？
