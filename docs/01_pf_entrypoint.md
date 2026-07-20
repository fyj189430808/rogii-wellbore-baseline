# PF 与当前高分流程的入口

核对日期：2026-07-20。本文把“历史 Notebook 总入口”“裸 PF 生成入口”“41 列 LightGBM 入口”分别说明。

## 0. 一句话结论

历史高分方案的总入口不是某个 Python 脚本，而是：

    rogii-dual-track-prefix-calibrated-geosteering.ipynb

必须从上到下执行整个 Notebook。第一个真正控制算法的单元格是单元格 6；单元格 43 中虽然有一个名为 `main()` 的函数，但它只负责第二条 learned trajectory，不是全项目入口。

但当前已经整理好的、可做五折复现的入口是两个 Python 脚本：

```text
rogii_clean/scripts/run_p2_p01_multiseed_pf_mean.py
    负责生成 128-seed PF 路径缓存

rogii_clean/scripts/run_p2_p02_multiscale_pf_paths_cv.py
    负责把 PF 路径加入 41 列单模 LightGBM 并完成五折评分
```

这两个入口不能混称：第一个输出裸 PF 路径，第二个输出带 LightGBM 的 `10.3057` OOF。

版本状态：2026-07-16 起 Notebook 源码默认关闭直接同井标签/contact 通道。Notebook 中现存的同井覆盖输出是修改前历史日志，不是当前 private-safe 源码的运行结果。当前本地也没有历史 `10.7321` 对应的完整配置和预测产物，因此不能说它已经被精确复现。

## 1. 输入和输出

### 1.1 总输入

| 输入 | 本地路径 | Kaggle 运行路径 | 作用 |
|---|---|---|---|
| 比赛数据 | `input/data/raw/` | `/kaggle/input/competitions/rogii-wellbore-geology-prediction` | 井数据和提交 ID |
| 第一条轨道模型 | `input/others/data/`、`input/others/models/` | `/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts` | 五个预训练模型和训练特征 |
| 第二条轨道模型 | `input/others/features.json`、`input/others/lgb*.pkl` | `/kaggle/input/datasets/fleongg/rogii-claude-models-pub` | 三个 learned trajectory 模型 |
| 最终模型包 | 本地缺失 | `/kaggle/input/datasets/pilkwang/rogii-model-package` | 最多 0.5% 的门控校正 |
| koolbox wheel | `input/others/koolbox-*.whl` | 一个 `koolbox-offline` 挂载目录 | 反序列化旧 Trainer |

本地路径提醒：第二版 CFG 支持用 `ROGII_DATA` 指定数据目录，但第一条轨道仍直接读取单元格 6 的 `COMPETITION_DATA_ROOT`。此外，本地缺少最终模型包。原 Notebook 因而不能只设置 `ROGII_DATA` 就在本机从头跑通；当前可复现流程应走 `rogii_clean/scripts/`。

### 1.2 总输出

Kaggle 中写到：

    /kaggle/working/submission.csv

本地 fallback 写到当前工作目录：

    ./submission.csv

最终要求：

- 两列严格为 `id`、`tvt`；
- 14,151 行；
- ID 顺序和 `sample_submission.csv` 完全相同；
- `tvt` 全部为有限数值。

## 2. 第一入口：单元格 6

单元格 6 的前 18 行是当前最应该先读的代码：

    SUBMISSION_PROFILE = 'dual_track_prefix_modelpkg_005'

    PROFILE_PRESETS = {
        'dual_track_prefix_balanced': ...,
        'dual_track_prefix_balanced_w054': ...,
        'dual_track_prefix_modelpkg_005': ...,
        'dual_track_prefix_modelpkg_010': ...,
    }

当前激活 profile：

    dual_track_prefix_modelpkg_005

它固定了：

| 控制 | 当前值 | 含义 |
|---|---:|---|
| `RIDGE_PP_ALPHA` | 1.0 | 第一条树模型增量的总体倍率 |
| `RIDGE_PP_TAU` | 85 ft | 增量从 0 逐渐放开的 warm-up 长度 |
| `RIDGE_PP_W_PF` | 0.15 | 第一条轨道中 PF 增量权重 |
| `SP45_RIDGE_MODEL_WEIGHT` | 0.30 | 第一条残差轨道进入初始 anchor 的权重 |
| `SP45_SELECTOR_WEIGHT` | 0.70 | physical/selector 子轨道权重 |
| `SP45_SELECTOR_N_PARTICLES` | 500 | selector 单次 PF 粒子数 |
| `SP45_SELECTOR_N_SEEDS` | 128 | selector PF 随机种子数 |
| `SP45_PROJECTION_DEGREE` | 4 | `U=TVT+Z` 的多项式次数 |
| `SP45_PROJECTION_BLEND_WEIGHT` | 0.75 | 投影路径权重 |
| `SP45_BLEND_WEIGHT` | 0.55 | SP45 投影轨道进入双轨融合的权重 |
| `VISIBLE_PREFIX_CAL_SEEDS` | 24 | 可见前缀伪 holdout 阶段 PF 种子数 |
| `VISIBLE_PREFIX_FINAL_SEEDS` | 48 | 最终候选重建阶段 PF 种子数 |
| `VISIBLE_PREFIX_PARTICLES` | 350 | 可见前缀 PF 粒子数 |
| `MODEL_PACKAGE_GATED_MAX_WEIGHT` | 0.005 | 模型包最大权重 |

当前关闭：

- `RUN_CV_REPORT=False`
- `RUN_FULL_STACK_CV_ABLATION=False`
- `RUN_BIMODAL_DETECTOR=False`
- `RUN_SAME_WELL_CONTACT_OVERRIDE=False`
- `RUN_OVERLAP_DRY_RUN_PROBE=False`
- `RUN_GUARDED_OVERLAP_OVERRIDE=False`

当前打开：

- `RUN_HEEL_CALIBRATION=True`
- `RUN_VISIBLE_PREFIX_CALIBRATION=True`
- `RUN_MODEL_PACKAGE_CORRECTION=True`

## 3. 第二入口问题：Notebook 有两个 `CFG`

### 3.1 第一版 CFG：单元格 10

    class CFG:
        dataset_path = Path(COMPETITION_DATA_ROOT)
        artifacts_path = Path(RIDGE_ARTIFACT_ROOT)
        seed = 42
        n_splits = 5
        cv = GroupKFold(n_splits=n_splits)
        metric = root_mean_squared_error

它控制：

- 单元格 11 的 selector；
- 单元格 14～30 的第一条残差/physical 轨道；
- 五个预训练模型和 Ridge；
- 第一次生成、融合和投影 `submission.csv`。

### 3.2 第二版 CFG：单元格 33

单元格 33 再次执行：

    class CFG:
        DATA = _find_data()
        OUT = ...
        seed = 42
        n_splits = 5
        PF_SEEDS = 128
        PF_PARTICLES = 500
        PF_SCALES = (3., 5., 8., 12.)

Python 会用新的类对象覆盖全局名字 `CFG`。此后：

- 不能再假设 `CFG.dataset_path` 存在；
- 后续代码改用 `CFG.DATA` 和 `CFG.OUT`；
- 单元格 53 特意用 `getattr` 兼容两版 CFG。

这是 Notebook 最容易看错的地方之一。两版 CFG 不是继承关系，也不是同一个配置对象。

## 4. 从上到下的执行链

### 阶段 A：环境准备

| 单元格 | 做什么 | 输入 | 输出/副作用 |
|---:|---|---|---|
| 6 | 选择 profile、路径和全局参数 | 无 | 全局常量 |
| 7 | 将可见前缀粒子数写入环境变量 | 单元格 6 参数 | 环境变量 |
| 8 | 查找/安装 koolbox；失败时建立兼容 Trainer | wheel | 可导入的 `koolbox.Trainer` |
| 9 | 导入 LightGBM、CatBoost、sklearn、Numba 等 | Python 环境 | 模块和类 |
| 10 | 定义第一版 CFG | 单元格 6 路径 | `CFG.dataset_path` 等 |

### 阶段 B：selector PF 和候选工具

| 单元格 | 函数 | 作用 |
|---:|---|---|
| 11 | `tvt_from_contacts()` | 用同井完整训练 TVT 和地层接触面重建轨迹 |
| 11 | `load_well()` | 读取一口井的 horizontal/typewell |
| 11 | `run_particle_filter()` | 运行一个随机种子的 selector PF |
| 11 | `run_pf_lik_ensemble_scales()` | 汇总 128 个随机种子，并产生 scale 3/5/8/12 |
| 11 | `beam_search()` | 用 GR 匹配搜索 typewell 路径 |
| 11 | `run_beam_ensemble()` | 平均 14 组 Beam 参数 |
| 11 | `selector_well_code()` | 根据隐藏长度和 Z 跨度给井分箱 |
| 11 | `apply_selector_variant()` | PF、Beam 和 carry hold 的最终组合 |
| 12 | optional CV/oracle | 只在开关打开时运行；当前关闭 |

### 阶段 C：第一条残差模型轨道

| 单元格 | 做什么 | 当前运行行为 |
|---:|---|---|
| 14 | 定义 PF-ANCC、PF-Z、Beam、NCC、空间地层特征和 `build_well()` | 只定义函数 |
| 15 | 读取 7.39 GB 的 `train.csv`，动态构造 3 口测试井特征 | 实际执行 |
| 16 | 定义 3 组 LightGBM、2 组 CatBoost、Ridge 和后处理参数 | 实际执行 |
| 17 | 创建预测字典 | 实际执行 |
| 18 | 从磁盘加载 3 个 LightGBM Trainer，并预测测试集 | 实际执行，不重新训练 |
| 19 | 从磁盘加载 2 个 CatBoost Trainer，并预测测试集 | 实际执行，不重新训练 |
| 20 | 将五组 OOF/测试预测转成 DataFrame | 实际执行 |
| 21 | 用正约束 Ridge 对五组 OOF 再做 GroupKFold 融合 | 实际执行 |
| 22 | 定义 warm-up 后处理和 Savitzky-Golay 平滑 | 只定义函数 |
| 23 | 在 OOF 上计算 `ridge (pp)` RMSE | 实际执行 |
| 24 | 生成测试绝对 TVT，并逐井平滑 | 实际执行 |
| 25 | 按 sample ID 对齐，得到 `sub_1` | 实际执行 |

`sub_1` 的公式：

    model_delta = Ridge(5 个基础模型输出)
    pf_delta = pf_ancc - last_known_tvt
    warmup = 1 - exp(-md_since / 85)
    delta = warmup * (0.85 * model_delta + 0.15 * pf_delta)
    sub_1_tvt = last_known_tvt + delta

### 阶段 D：physical/selector 子轨道

单元格 26 对每口测试井执行：

1. 计算 128-seed PF 和 14-configuration Beam；
2. 调用 `apply_selector_variant()` 得到 `tvt_selector`；
3. 只有手工打开 `RUN_SAME_WELL_CONTACT_OVERRIDE` 时，才检查同名训练井并生成 `tvt_phys`；
4. 写结果时保留以下受开关保护的兼容分支：

       if tvt_phys is not None:
           tvt_val = tvt_phys[row_index]
       else:
           tvt_val = tvt_selector[row_index]

当前开关为 False，所以即使本地 3 口测试井存在训练副本：

- 不读取同井完整 TVT 来生成 `tvt_phys`；
- `sub_2` 使用 `tvt_selector`；
- PF/Beam selector 会真正进入 0.3/0.7 anchor。

单元格 27 把结果变成 `sub_2`。

### 阶段 E：第一条 anchor 和投影

单元格 28：

    anchor_raw = 0.30 * sub_1 + 0.70 * sub_2

并第一次写入 `submission.csv`。

单元格 30：

1. 读取刚写出的 `submission.csv`；
2. 对每口井计算 `U=TVT+Z`；
3. 减去最后一个可见点的 `U_last`；
4. 以归一化 MD 为自变量做稳健四次多项式；
5. 得到完整投影 TVT；
6. 用 25% 原路径 + 75% 投影路径；
7. 覆盖 `submission.csv`。

注意：单元格 30 注释写“deg-5”，但实际传入 `SP45_PROJECTION_DEGREE=4`。运行值是四次，不是五次。

单元格 32 把此时结果另存为：

    sp45_projection_submission.csv

这是后面双轨融合必须依赖的备份。

### 阶段 F：第二条 learned trajectory

单元格 33 覆盖 CFG 后，单元格 36～40 建立另一套：

- `pf_ancc`
- `pf_z`
- likelihood PF
- Beam
- NCC
- 六个地层面 KNN
- GR 和轨迹特征

单元格 43 中的 `main()`：

1. 找到 773 口训练井和 3 口测试井；
2. 用训练井建立空间 imputer；
3. 动态计算测试特征；
4. 从 `rogii-claude-models-pub` 加载 `features.json` 和 3 个 `lgb*.pkl`；
5. 对 3 个模型预测取平均；
6. 调用 `make_prediction()`；
7. 再次覆盖 `submission.csv`。

此时 `submission.csv` 只代表 learned trajectory，但 `sp45_projection_submission.csv` 仍保留第一条轨道。

### 阶段 G：双轨融合

单元格 45：

    learned = submission.csv
    sp45 = sp45_projection_submission.csv
    final_base = 0.55 * sp45 + 0.45 * learned

它生成 0.52、0.54、0.55、0.56、0.58 五个候选文件，当前选择 0.55，并再次覆盖 `submission.csv`。

### 阶段 H：同井接触面覆盖（当前关闭）

单元格 47 只探测测试井是否有同名训练副本。

单元格 48：

1. 从同名训练井取 `TVT`、`Z`、`EGFDU`；
2. 从训练 typewell 的 `Geology=EGFDU` 找接触 TVT；
3. 构造接触面物理轨迹；
4. 在测试井可见前缀上按 MD 插值；
5. 若至少 50 行且 RMSE 不超过 1 ft，覆盖隐藏段。

Notebook 保存的历史日志：

| 井 | 前缀 RMSE | 覆盖行 |
|---|---:|---:|
| `000d7d20` | 0.0101 | 3,836/3,836 |
| `00bbac68` | 0.0090 | 6,014/6,014 |
| `00e12e8b` | 0.0079 | 4,301/4,301 |

旧运行中 14,151 行全部被这一层覆盖。当前 `RUN_GUARDED_OVERLAP_OVERRIDE=False`，单元格 48 只打印 disabled 并保留双轨融合。

### 阶段 I：可见前缀伪 holdout

单元格 50：

- 在可见前缀内部按 0.50、0.65、0.75 三个位置制造伪隐藏段；
- 对 104 个候选比较后段已知 TVT 的 RMSE；
- 旧运行中 3 口井最佳候选都是 `contact_md_lookup`；
- 当前 private-safe 模式不把 contact 候选加入候选池，也不重放 contact guard；
- 新结果必须重新运行后评估。

日志显示 balanced 的平均移动约 `1.39e-13 ft`，说明上一层已经是同一个接触面结果。

### 阶段 J：模型包微调和审计

单元格 52：

- 构建 480 个模型包特征；
- 运行 XGB、CatBoost、HGB、LGB、TCN；
- 生成模型包轨迹；
- 用分歧门控给最多 0.5% 权重；
- 再次覆盖 `submission.csv`。

当前平均绝对移动：

    0.01067 ft

单元格 53 最后检查：

- 行数；
- 列名；
- ID 顺序；
- TVT 是否有限；
- TVT 范围；
- SHA-256。

## 5. `submission.csv` 覆盖时间线

| 顺序 | 单元格 | 此时含义 |
|---:|---:|---|
| 1 | 28 | 30% 残差轨道 + 70% physical/selector |
| 2 | 30 | 对第 1 步做 `U=TVT+Z` 四次投影 |
| 3 | 32 | 不覆盖；保存为 `sp45_projection_submission.csv` |
| 4 | 43 | learned trajectory，临时覆盖 |
| 5 | 45 | 55% SP45 + 45% learned |
| 6 | 48 | 当前关闭，保留第 5 步结果 |
| 7 | 50 | 可见前缀候选选择；不包含 contact 候选 |
| 8 | 52 | 最大 0.5% 模型包校正 |
| 9 | 53 | 不改变数值，只审计并复制 |

如果只看到单元格 28 或 43 的 `submission.csv`，就不是最终提交。

## 6. PF 真正从哪里开始

### 6.1 历史 Notebook 的 selector PF

如果目标是先独立理解历史 Notebook 的 selector PF，入口是：

    Notebook 单元格 11
      └── run_particle_filter(hw, tw, n_particles=500, seed=42)

它的直接调用者：

    run_pf_lik_ensemble_scales(...)
      └── 循环调用 run_particle_filter 128 次

测试推理调用位置：

    Notebook 单元格 26，第 38 行附近

    pf_by_scale = run_pf_lik_ensemble_scales(
        hw_te,
        tw_ref,
        n_particles=500,
        n_seeds=128,
    )

组合位置：

    Notebook 单元格 11
      └── apply_selector_variant(...)

历史运行曾在单元格 26 选择 `tvt_phys`。当前 private-safe 开关关闭该分支，PF/Beam selector 会进入 `sub_2`；仍需重新执行 Notebook 才能得到对应提交。

### 6.2 当前可复现的 PF 入口

当前用于学习和复算的主入口是：

```text
python rogii_clean/scripts/run_p2_p01_multiseed_pf_mean.py
```

调用链：

```text
scripts/run_p2_p01_multiseed_pf_mean.py
→ 读取每口井的 MD、Z、GR、TVT_input
→ 读取配对 typewell 的 TVT、GR
→ src/p2_p01_multiseed_pf.py::prepare_particle_filter_inputs()
→ particle_filter_all_seeds_numba()
→ build_multiseed_pf_features()
→ 保存 mean、scale 3/5/8/12 路径和逐井运行记录
```

最值得先读的代码：

| 位置 | 作用 |
|---|---|
| `src/p2_p01_multiseed_pf.py:57` | 128 个 seed 共用的粒子滤波内核 |
| `src/p2_p01_multiseed_pf.py:466` | 把一口井整理成 PF 所需数组 |
| `src/p2_p01_multiseed_pf.py:682` | 把 128 条路径汇总为均值和四个 scale |
| `scripts/run_p2_p01_multiseed_pf_mean.py:450` | 只读取推理时合法的原始列 |
| `scripts/run_p2_p01_multiseed_pf_mean.py:959` | 命令行主入口和逐井缓存循环 |

这个脚本产生的是路径，不训练跨井模型。当前完整回放中最好的裸路径是 scale 8，RMSE 为 `10.9543`。

### 6.3 当前 10.3057 的入口

```text
python rogii_clean/scripts/run_p2_p02_multiscale_pf_paths_cv.py
```

它做的事情是：

```text
P2-P01 的五条 PF 相对路径
+ 36 条冻结基础/确定性候选特征
→ 41 列输入
→ balanced_well_5fold_v1
→ 单个 LightGBM
→ 预测 TVT 相对最后可见 TVT 的增量
→ 完整五折 OOF RMSE 10.3057049921
```

因此，`P3B00_group5_p2p02_v1` 只是把这次结果冻结为后续实验的比较基线；它不是另一个 PF 算法，也没有重新训练模型。

### 6.4 三个名字不要混用

| 常见叫法 | 实际内容 | 当前可核验分数 |
|---|---|---:|
| selector PF | Notebook cell 11 的 PF + 可选 Beam + carry hold | 没有保存一份可信的 128-seed 完整五折产物 |
| 裸 PF | P2-P01 复刻的 mean/scale 路径 | 最好 scale 8：10.9543 |
| P2-P02 / P3B00 | 五条 PF 路径进入 41 列单模 LightGBM | 10.3057 |
| 历史“PF 10.7” | 口头名称，血缘可能混合 selector 与后处理 | 本地尚不能精确指认 |

## 7. 当前入口的 fallback

| 失败位置 | fallback |
|---|---|
| koolbox 安装失败 | 动态建立兼容 Trainer |
| 预训练第一轨模型不存在 | 重新做 GroupKFold 训练 |
| PF 失败 | last-known carry-forward |
| Beam 失败 | 使用 PF 路径 |
| 投影失败 | 保留投影前轨迹 |
| learned 模型不存在但有完整 CSV | 使用 ID 完全匹配的预计算 CSV |
| learned 模型和 CSV 都不存在 | 从头训练第二条轨道 |
| 同井前缀 guard 不通过 | 保留双轨融合 |
| 模型包缺失 | 当前 profile 要求存在，应该报错，不允许静默跳过 |

## 8. 第一次阅读时只跟踪这些变量

| 变量 | 含义 |
|---|---|
| `hw` / `hw_te` | 一口水平井 DataFrame |
| `tw` / `tw_ref` | 一口 typewell DataFrame |
| `kn` | `TVT_input` 有值的可见前缀 |
| `ev` | `TVT_input` 为空的隐藏后缀 |
| `last_known_tvt` | 最后一个可见 TVT |
| `sub_1` | 第一条树模型残差轨道 |
| `tvt_pf` | likelihood PF 候选 |
| `tvt_beam` | Beam 候选 |
| `tvt_selector` | PF/Beam/carry hold 组合 |
| `tvt_phys` | 同井接触面重建 |
| `sub_2` | physical 存在时用 physical，否则用 selector |
| `sp45_projection_submission.csv` | 第一条 anchor 投影后的固定备份 |
| `submission.csv` | 会被多次覆盖的当前候选 |

## 9. 一个最小人工追踪例子

对井 `000d7d20`：

1. 训练/测试水平井共有 5,278 行。
2. 前 1,442 行 `TVT_input` 可见。
3. 后 3,836 行进入 sample submission。
4. selector PF 计算 128 个随机路径集合。
5. 当前分箱选择 `pf_scale_5_hold_0.2`。
6. 即使同名训练井存在，private-safe 开关也让 `sub_2` 使用 `tvt_selector`。
7. `sub_1` 和 `sub_2` 做 0.3/0.7 融合并投影。
8. 再与 learned trajectory 做 0.55/0.45 融合。
9. 单元格 48 因开关关闭而保留双轨结果。
10. 单元格 50 只比较不依赖同井完整标签的候选。
11. 单元格 52 再做小权重模型包校正。

## 10. 阅读完成后的自测

读完本文件后，应该能回答：

1. 为什么项目没有 `main.py`？
2. 为什么单元格 43 的 `main()` 不是全项目入口？
3. 两个 `CFG` 有什么区别？
4. `submission.csv` 被覆盖了几次？
5. `sp45_projection_submission.csv` 为什么必须保留？
6. private-safe 模式下 `sub_2` 为什么会使用 PF/Beam selector？
7. 为什么 Notebook 中保存的 14,151 行同井覆盖日志已经是历史输出？
8. 最终模型包的 0.5% 是模型内部权重还是整个包的外部权重？
9. 为什么 P2-P01 的 scale 8 `10.9543` 才是当前可核验的裸 PF，而 P2-P02 的 `10.3057` 不是纯 PF？
10. `P3B00_group5_p2p02_v1` 为什么只是冻结指针，不是一个新模型？
