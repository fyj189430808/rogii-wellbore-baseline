# ROGII 项目文件地图

核对日期：2026-07-20。本文同时记录历史高分 Notebook 和当前可复现 P2 主线；两者会明确分栏，不互相冒充。

## 0. 文档目的和当前结论

这份文档回答六个问题：

1. 项目里有哪些文件和目录？
2. 每一类文件解决什么问题？
3. 哪些文件属于当前高分 Notebook 的实际执行链？
4. 谁读取它、它又依赖谁？
5. 它的输入和输出是什么？
6. 学习 PF 时哪些内容可以暂时忽略？

项目根目录是：

    H:\kaggle\716

当前最重要的事实是：

- 项目没有传统的 `main.py`。
- 总入口是 `rogii-dual-track-prefix-calibrated-geosteering.ipynb`。
- Notebook 有 57 个单元格，包含多个重复或相近的 PF 实现。
- 历史高分运行不是纯 PF，而是“树模型残差 + PF/Beam + 投影 + 第二条学习轨道 + 可选同井覆盖 + 小权重模型包校正”。
- 当前电脑上的依赖已经被整理成 `input/data/raw/` 和扁平的 `input/others/`，不再保留 Notebook 中每个 Kaggle 数据集原来的目录名。
- 当前真正可逐行核对的 PF 核心在 `rogii_clean/src/p2_p01_multiseed_pf.py`；它复刻的是 Notebook 单元格 37 的 128-seed likelihood PF。
- 当前可核验的纯 PF 最好路径是 scale 8，完整 773 井回放 RMSE 为 `10.9543`；不是口头沿用的 `10.7`。
- 当前固定的 `10.3057049921` 是“五条 PF 路径 + 36 条其他合法特征 + 单个 LightGBM”的结果，不能称作纯 PF。
- 历史 `10.7321` 在本地没有对应的完整预测、指标和配置产物，因此现在不能声称已经精确复现它。

因此，本文严格区分三件事：

1. 历史高分 Notebook 里的 selector PF；
2. 当前已经复刻并保存缓存的裸 PF 路径；
3. PF 路径进入 41 列单模 LightGBM 后的 `10.3057` 基线。

## 1. 顶层目录

    H:\kaggle\716
    ├── AGENTS.md
    ├── rogii-dual-track-prefix-calibrated-geosteering.ipynb
    ├── docs/
    │   ├── 00_project_map.md
    │   ├── 01_pf_entrypoint.md
    │   ├── 02_pf_data_flow.md
    │   ├── 03_pf_algorithm.md
    │   └── 07_validation_and_metrics.md
    ├── rogii_clean/                              # 当前可复现代码与历史实验档案
    │   ├── configs/
    │   ├── scripts/
    │   ├── src/
    │   ├── artifacts/folds/
    │   ├── experiments/
    │   └── tests/
    └── input/
        ├── data/raw/                             # 当前原始比赛数据
        └── others/                               # 扁平保存的旧模型、训练特征、wheel 和历史提交

## 2. 顶层文件

| 路径 | 用途 | 属于当前主流程 | 被谁读取 | 它读取谁 | 主要输入 | 主要输出 | 现在能否忽略 |
|---|---|---:|---|---|---|---|---:|
| `AGENTS.md` | 规定学习顺序、防泄漏、文档、实验和代码风格 | 间接属于 | Codex 和项目协作者 | 无 | 项目规则 | 工作方式约束 | 否 |
| `rogii-dual-track-prefix-calibrated-geosteering.ipynb` | 当前高分方案的总入口和全部执行顺序 | 是 | Kaggle/Jupyter 内核 | 原始数据、三个模型包、离线依赖 | CSV、模型、JSON | `submission.csv` 和诊断文件 | 否 |
| `docs/00_project_map.md` | 当前文件地图 | 学习主线 | 人阅读 | 文件系统和 Notebook 考古结果 | 路径、调用证据 | 项目地图 | 否 |
| `docs/01_pf_entrypoint.md` | 解释从哪个单元格开始、每一步如何覆盖输出 | 学习主线 | 人阅读 | Notebook | 单元格、函数 | 入口导读 | 否 |
| `docs/02_pf_data_flow.md` | 从 CSV 到最终 TVT 的完整数据流 | 学习主线 | 人阅读 | Notebook、数据 schema | 数据与代码位置 | 数据流导读 | 否 |
| `docs/03_pf_algorithm.md` | PF 的直观含义、公式、数值例子和代码对应 | 学习主线 | 人阅读 | PF 函数 | 参数和公式 | 算法导读 | 否 |
| `docs/07_validation_and_metrics.md` | 旧 CV 审计、当前按井五折和统一指标 | 验证主线 | 人阅读 | fold 注册表和评分代码 | CV 证据 | 固定验证制度 | 否 |

## 2.1 当前可复现项目和实验档案：`rogii_clean/`

这个目录名字里虽然有 `clean`，但经过一、二、三阶段实验后，它现在已经同时包含“可复现主线”和“大量实验档案”。学习 PF 时先只跟踪下表中的主线文件，其余 `P3_*`、`RF*`、`F*` 实验暂时不要展开。

| 路径 | 用途 | 当前状态 |
|---|---|---|
| `src/p2_p01_multiseed_pf.py` | 当前权威 PF 内核：500 粒子、128 个 seed、四个 scale | 已复刻并运行 773 井 |
| `scripts/run_p2_p01_multiseed_pf_mean.py` | 读取合法列，逐井生成 PF 路径与缓存 | 已运行，支持断点续跑 |
| `configs/p2_p01_multiseed_pf_mean_v1.json` | 冻结 PF 参数、输入边界和缓存血缘 | 已保存 |
| `artifacts/P2_P01_multiseed_pf_mean_v1/` | 裸 PF 路径、逐井运行记录和回放指标 | 已保存 |
| `scripts/run_p2_p02_multiscale_pf_paths_cv.py` | 把 PF 路径与基础特征合并，再运行固定单模 LightGBM | 已完成五折 |
| `configs/p2_p02_multiscale_pf_paths_v1.json` | P2-P02 的 41 列合同和产物指纹 | 已保存 |
| `artifacts/P2_P02_multiscale_pf_paths_v1/` | 10.3057 的配置、41 列列表、OOF、指标与逐井结果 | 已保存 |
| `artifacts/P3B00_group5_p2p02_v1/` | 指向 P2-P02 的冻结基线指针，不是一次新训练 | 已冻结 |
| `artifacts/folds/balanced_well_5fold_v1.csv` | 当前正式按井五折注册表 | 773 井 |
| `src/metrics.py` | micro/macro/P90/逐折/井级 bootstrap | 已测试 |
| `scripts/run_simple_lgbm_cv.py` | 冻结 LightGBM 的实际训练和绝对 TVT 恢复 | 已运行 |
| `P3_*`、`RF*`、`F*` 文件与目录 | 后续特征实验和诊断档案 | PF 初读时全部可忽略 |

## 3. 原始比赛数据：`input/data/raw/`

### 3.1 目录结构

    input/data/raw/
    ├── AI_wellbore_geology_prediction_task_en.pptx
    ├── sample_submission.csv
    ├── train/
    │   ├── <well_id>__horizontal_well.csv      # 773 个
    │   ├── <well_id>__typewell.csv             # 773 个
    │   └── <well_id>.png                       # 773 个
    └── test/
        ├── <well_id>__horizontal_well.csv      # 3 个
        └── <well_id>__typewell.csv             # 3 个

### 3.2 数据文件职责

| 文件或模式 | 用途 | 当前主流程 | 被谁读取 | 它调用谁 | 输入 | 输出 | 能否暂时忽略 |
|---|---|---:|---|---|---|---|---:|
| `AI_wellbore_geology_prediction_task_en.pptx` | 比赛官方任务说明 | 背景资料 | 人阅读 | 无 | 无 | 任务语义 | PF 代码初读时可以，字段含义核对时不可以 |
| `sample_submission.csv` | 规定最终提交的 ID、行数和顺序 | 是 | Notebook 单元格 25、26、43、53 | 无 | `id` | 提交骨架 | 否 |
| `train/<id>__horizontal_well.csv` | 训练井完整轨迹、GR、可见前缀和真实 TVT | 是 | `load_well`、`build_well`、模型包特征构建器、同井覆盖层 | pandas | 原始 CSV | 每井 DataFrame | 否 |
| `train/<id>__typewell.csv` | 训练井对应的 typewell TVT-GR 曲线和 Geology 接触标签 | 是 | PF、Beam、接触面重建 | pandas、`np.interp` | 原始 CSV | 参考 GR 曲线和接触 TVT | 否 |
| `train/<id>.png` | 每口训练井的可视化图 | 否 | 人阅读 | 无 | 图像 | 直观检查 | 是 |
| `test/<id>__horizontal_well.csv` | 测试井轨迹、GR、可见 TVT 前缀；无 `TVT` 和六个地层列 | 是 | 两条轨道、selector、覆盖层、审计层 | pandas | 原始 CSV | 测试井 DataFrame | 否 |
| `test/<id>__typewell.csv` | 测试井 typewell TVT-GR 曲线；无 Geology | 是 | PF 和 Beam；若存在同井训练副本则主流程改用训练 typewell | pandas | 原始 CSV | 参考 GR 曲线 | 否 |
| `rogii_artifacts/` | 预留目录，本地为空 | 否 | 无 | 无 | 无 | 无 | 是 |

### 3.3 水平井字段

训练水平井有 13 列：

| 列 | 含义 | 测试时存在 |
|---|---|---:|
| `MD` | measured depth，沿井眼累计长度，ft | 是 |
| `X`、`Y`、`Z` | 三维轨迹坐标 | 是 |
| `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` | 六个地层界面坐标 | 否 |
| `TVT` | 完整真实目标 | 否 |
| `GR` | 水平井伽马曲线 | 是 |
| `TVT_input` | 可见前缀有值、隐藏后缀为空 | 是 |

typewell 训练文件有 `TVT`、`GR`、`Geology`；测试文件只有 `TVT`、`GR`。

### 3.4 当前本地测试集的特殊关系

当前 3 口测试井：

- `000d7d20`
- `00bbac68`
- `00e12e8b`

它们全部也出现在训练目录中。逐列只读比较得到：

- 测试和训练副本的 `MD/X/Y/Z/GR/TVT_input` 完全相同；
- 测试和训练 typewell 的 `TVT/GR` 完全相同；
- 训练副本额外含完整 `TVT`、六个地层界面和 typewell `Geology`。

这就是历史高分运行中同井接触面覆盖层能够触发的原因。当前 private-safe 源码已经关闭该通道；这段关系只作为旧结果考古证据保留。

## 4. 第一条残差模型轨道：Kaggle 逻辑目录与本地镜像

Notebook 中把这组文件称为 `Wellbore Geology Prediction  Artifacts/`。当前电脑没有同名目录；对应文件被扁平保存到 `input/others/`。

### 4.1 文件清单

| 文件 | 用途 | 当前主流程 | 被谁读取 | 输入 | 输出 | 能否暂时忽略 |
|---|---|---:|---|---|---|---:|
| `input/others/data/train.csv` | 预计算的 773 井隐藏段训练特征，约 7.39 GB | 是 | Notebook 单元格 15 | 特征表 | `train_df` | 理解 PF 时可暂时忽略；理解最终 Notebook 时不可 |
| `input/others/models/lightgbm-1/...pkl` | 第一组 LightGBM 的 GroupKFold Trainer | 是 | 单元格 18 | `X_test` | OOF 和测试增量预测 | 同上 |
| `input/others/models/lightgbm-2/...pkl` | 第二组 LightGBM Trainer | 是 | 单元格 18 | `X_test` | OOF 和测试增量预测 | 同上 |
| `input/others/models/lightgbm-3/...pkl` | 第三组 LightGBM Trainer | 是 | 单元格 18 | `X_test` | OOF 和测试增量预测 | 同上 |
| `input/others/models/catboost-1/...pkl` | 第一组 CatBoost Trainer | 是 | 单元格 19 | `X_test` | OOF 和测试增量预测 | 同上 |
| `input/others/models/catboost-2/...pkl` | 第二组 CatBoost Trainer | 是 | 单元格 19 | `X_test` | OOF 和测试增量预测 | 同上 |

### 4.2 调用关系

    Notebook 单元格 15
      ├── 读取 data/train.csv
      └── 对 3 口测试井调用 build_dataset(...) 动态建立 test_df

    Notebook 单元格 18、19
      ├── joblib.load(...) 读取 5 个 Trainer
      ├── 取出 Trainer.oof_preds
      └── Trainer.predict(X_test)

    Notebook 单元格 21
      └── 用正约束 Ridge 融合 5 组预测

这些模型预测的目标不是绝对 TVT，而是：

    target = hidden_TVT - last_known_TVT

## 5. 第二条 learned trajectory：Kaggle 逻辑目录与本地镜像

Notebook 中的逻辑目录名是 `rogii-claude-models-pub/`；本地对应文件位于 `input/others/` 根目录。

| 文件 | 用途 | 当前主流程 | 被谁读取 | 输入 | 输出 | 能否暂时忽略 |
|---|---|---:|---|---|---|---:|
| `input/others/features.json` | 记录 200 多个 learned trajectory 输入特征的顺序 | 是 | Notebook 单元格 43 `main()` | JSON | 特征名列表 | 理解基础 PF 时可以 |
| `input/others/lgb0.pkl` | 预训练 LightGBM 轨迹模型 1 | 是 | `main()` | `test_df[features]` | TVT 增量 | 理解基础 PF 时可以 |
| `input/others/lgb1.pkl` | 预训练 LightGBM 轨迹模型 2 | 是 | `main()` | 同上 | TVT 增量 | 同上 |
| `input/others/lgb2.pkl` | 预训练 LightGBM 轨迹模型 3 | 是 | `main()` | 同上 | TVT 增量 | 同上 |

Notebook 单元格 33 重新定义 `CFG` 后，单元格 36～43 动态构建 PF、Beam、NCC、空间地层和 GR 特征。单元格 43 对三个模型取简单平均，再由 `make_prediction()` 与 likelihood PF 轨迹混合。

## 6. 最终微小校正：`ROGII Model Package/`

这个目录是 Notebook 在 Kaggle 上依赖的独立多模型包。当前 profile 要求它必须存在，但只允许它在最终轨迹上做最大 0.5% 的门控移动。**当前本地 `input/` 中没有找到 `model_package_manifest.json`，所以原 Notebook 不能在本机完整走到这一层。** 下表记录的是 Notebook 代码和保存日志所要求的逻辑结构，不代表这些文件现在都在本机。

### 6.1 入口和特征构建

| 文件 | 用途 | 当前主流程 | 被谁读取 | 它读取谁 | 输出 |
|---|---|---:|---|---|---|
| `metadata/model_package_manifest.json` | 模型包总清单、模型类型、特征集和相对路径 | 是 | Notebook 单元格 52 | 下面的模型和配置 | 推理计划 |
| `feature_builders/build_features.py` | 模型包公开入口，动态加载核心特征代码 | 是 | 单元格 52 | manifest、core、原始数据 | 推理特征 DataFrame |
| `feature_builders/feature_columns.json` | `drift_ncc_v1` 的 480 列顺序 | 是 | `build_features.py` | JSON | 特征列表 |
| `feature_builders/rogii_feature_core.py` | 3,187 行完整特征工程核心 | 是 | `build_features.py` | 原始井数据 | 480 特征及辅助元数据 |

`rogii_feature_core.py` 主要包含：

- prediction zone 划分；
- typewell 插值和相关性；
- sequence/ancc/z PF 特征；
- Beam 和候选路径；
- 六个地层面的 KNN；
- GR、尾段、漂移和 NCC 特征；
- 模型矩阵、后处理和提交对齐工具。

它不是当前 selector PF 的唯一实现，而是最终模型包自己的特征构建系统。

### 6.2 模型文件

下列每个树模型家族都有 `fold0..fold4` 和 `alltrain`：

| 模式 | 用途 | 当前推理是否直接使用 |
|---|---|---:|
| `models/drift_ncc_xgb_fold0.json` ～ `fold4.json` | XGBoost 五折模型 | 否，供 CV/验证 |
| `models/drift_ncc_xgb_alltrain.json` | 全训练集 XGBoost | 是 |
| `models/drift_ncc_catboost_fold0.cbm` ～ `fold4.cbm` | CatBoost 五折模型 | 否，供 CV/验证 |
| `models/drift_ncc_catboost_alltrain.cbm` | 全训练集 CatBoost | 是 |
| `models/drift_ncc_hgb_fold0.joblib` ～ `fold4.joblib` | HistGradientBoosting 五折模型 | 否，供 CV/验证 |
| `models/drift_ncc_hgb_alltrain.joblib` | 全训练集 HGB | 是 |
| `models/drift_ncc_lgb_fold0.txt` ～ `fold4.txt` | LightGBM 五折模型 | 否，供 CV/验证 |
| `models/drift_ncc_lgb_alltrain.txt` | 全训练集 LightGBM | 是 |
| `models/sequence_tcn_tcn_residual.pt` | TCN 序列残差模型 | 是 |

当前 `stacking/blend_config.json` 中的家族权重为：

| 家族 | 权重 |
|---|---:|
| XGBoost | 约 0 |
| CatBoost | 0.4496275 |
| HGB | 0.1177839 |
| LightGBM | 0.0150923 |
| TCN | 0.4174962 |

这是模型包内部的权重。模型包整体进入最终提交时，还要再乘 Notebook 的最多 0.5% 门控权重。

### 6.3 OOF 文件

| 文件 | 用途 | 当前提交是否直接读取 | 现在能否忽略 |
|---|---|---:|---:|
| `oof/train_gt.parquet` | OOF 行的真实目标和键 | 否 | PF 初读时可以 |
| `oof/xgb_oof.npy` | XGBoost OOF | 否 | 可以 |
| `oof/catboost_oof.npy` | CatBoost OOF | 否 | 可以 |
| `oof/hgb_oof.npy` | HGB OOF | 否 | 可以 |
| `oof/lgb_oof.npy` | LightGBM OOF | 否 | 可以 |
| `oof/sequence_tcn_oof.npy` | TCN OOF | 否 | 可以 |
| `oof/blend_oof.npy` | 原始融合 OOF | 否 | 可以 |
| `oof/blend_oof_postprocessed.npy` | 后处理融合 OOF | 否 | 可以 |

manifest 记录：

- OOF 行数：3,783,989；
- 井数：773；
- 折数：5；
- OOF 完整率：1.0；
- 模型包后处理 OOF RMSE：10.6702109975。

需要注意 manifest 明确承认：地层/KNN imputer 没有在每个验证折内完全重建，而是“全训练数据构建、查询井自排除”。这可能使 OOF 比严格 outer-fold 重建更乐观，后续必须在验证文档中单独审计。

### 6.4 配置和报告

| 文件 | 用途 | 当前推理是否读取 |
|---|---|---:|
| `stacking/blend_config.json` | 模型包融合权重、OOF 分数、后处理参数 | 是 |
| `postprocess/postprocess_config.json` | 旧版或单独保存的 alpha、tau、Savitzky-Golay 参数 | 可能由包内逻辑读取 |
| `reports/alltrain_models.csv` | 全训练模型清单 | 否 |
| `reports/blend_l2_grid.csv` | 树模型融合 L2 网格 | 否 |
| `reports/blend_weights.csv` | 树模型融合权重报告 | 否 |
| `reports/drift_ncc_model_package_build.json` | 原始四树模型包构建记录 | 否 |
| `reports/fold_scores.csv` | 每折分数 | 否 |
| `reports/model_package_verify.json` | 文件存在性和 100 行 smoke test | 否 |
| `reports/model_package_verify_smoke.json` | 额外 smoke test | 否 |
| `reports/model_scores.csv` | 模型级分数 | 否 |
| `reports/postprocess_grid.csv` | 后处理搜索结果 | 否 |
| `reports/tcn_augmented_blend_l2_grid.csv` | 加入 TCN 后的 L2 网格 | 否 |
| `reports/tcn_augmented_blend_weights.csv` | 加入 TCN 后的融合权重 | 否 |
| `reports/tcn_augmented_model_scores.csv` | 加入 TCN 后的模型分数 | 否 |
| `reports/tcn_augmented_postprocess_grid.csv` | TCN 版本后处理搜索 | 否 |
| `reports/tcn_augmented_summary.csv/json` | TCN 模型包总结果 | 否 |
| `reports/well_scores.csv` | 逐井 OOF 分数 | 否 |
| `reports/dataset_manifest.json` | 打包时的完整文件清单 | 否 |

## 7. 离线运行依赖：`input/others/*.whl`

该目录包含 16 个 wheel 和 2 个未完成下载文件。当前真正的直接目的，是让 Notebook 能反序列化保存时类型为 `koolbox.Trainer` 的五个模型。

Notebook 单元格 8：

1. 查找与 Python 版本匹配的 wheel；
2. 尝试离线安装；
3. 导入 `koolbox.Trainer`；
4. 若失败，动态建立一个兼容的 fallback `Trainer` 类。

| 文件类别 | 作用 | PF 初读时能否忽略 |
|---|---|---:|
| `koolbox-0.1.3-*.whl` | 提供原始 Trainer 类型 | 不能完全忽略 |
| `joblib/scikit_learn/scipy/...whl` | 离线依赖 | 可以 |
| `*.crdownload` | 未完成下载，无当前作用 | 可以 |

## 8. 当前 Notebook 未使用或无需先读的本地档案

### 8.1 TabICL 和安装包

`input/others/tabicl-*.whl`、`tabicl-regressor-*.ckpt` 以及大部分通用 Python wheel 不在当前 PF 主调用链中。Notebook 没有调用 TabICL，可以暂时忽略。

### 8.2 历史提交 CSV

`input/others/` 包含：

    9.349.csv
    9.537.csv
    9.571.csv
    9.765.csv
    9.956.csv
    10.142.csv
    11.284.csv
    11.338.csv

这些文件名很像排行榜分数，但当前没有 manifest 证明每个数字的来源，Notebook 也没有读取它们。它们只能称为“历史提交档案”，不能作为当前算法参数或 CV 证据。

## 9. Notebook 单元格地图

| 单元格 | 主要职责 | 当前执行 |
|---:|---|---:|
| 0～5 | 问题、公式、profile 说明 | Markdown |
| 6 | 选择 profile 和全局控制参数 | 是 |
| 7～10 | 环境、依赖、导入、第一版 CFG | 是 |
| 11 | selector PF、Beam、carry hold、可选 bimodal | 是 |
| 12 | 可选 oracle/CV 报告 | 否，`RUN_CV_REPORT=False` |
| 14 | 第一条轨道的 PF/Beam/NCC/地层特征构建 | 是 |
| 15～25 | 读取五个模型、Ridge 融合、生成 `sub_1` | 是 |
| 26～28 | private-safe 模式生成 selector `sub_2`，与 `sub_1` 进行 0.3/0.7 融合；同井 physical 分支默认关闭 | 是 |
| 30 | 对 `U=TVT+Z` 做四次多项式投影 | 是 |
| 31 | 分数图 | 执行但不改变预测 |
| 32 | 保存 `sp45_projection_submission.csv` | 是 |
| 33 | 第二次定义 CFG，覆盖第一版 CFG 名字 | 是 |
| 34～40 | 第二条轨道的 EDA、PF、Beam、NCC、空间特征 | 是 |
| 41～43 | 加载 3 个 learned 模型并写新的 `submission.csv` | 是 |
| 45 | 55% SP45 投影 + 45% learned trajectory | 是 |
| 47 | 同井副本只读探测 | 否，默认关闭 |
| 48 | EGFDU 同井接触面覆盖代码 | 否，默认关闭 |
| 50 | 可见前缀伪 holdout 候选选择；contact 候选默认不加入 | 是 |
| 52 | 模型包最多 0.5% 的门控校正 | 是 |
| 53 | 最终行数、ID、有限值和 SHA-256 审计 | 是 |
| 54 | 全栈 CV ablation | 否 |

## 10. 运行时产生的主要文件

| 输出 | 产生位置 | 含义 |
|---|---|---|
| `submission.csv` | 单元格 28、30、43、45、48、50、52 多次覆盖 | 当前时刻的最终候选 |
| `sp45_projection_submission.csv` | 单元格 32 | 第一条轨道投影后的固定备份 |
| `learned_trajectory_submission.csv` | 单元格 45 | 第二条轨道备份 |
| `sp45_learned_blend_report.csv` | 单元格 45 | 0.52～0.58 权重候选报告 |
| `guarded_overlap_override_report.csv` | 单元格 48 | 仅旧运行或手工重新启用时生成；记录同井覆盖 |
| `gold_prefix_calibration_report.csv` | 单元格 50 | 可见前缀伪 holdout 候选分数 |
| `gold_prefix_moves_balanced.csv` | 单元格 50 | 最终移动幅度 |
| `bimodal_selector_report.csv` | 单元格 26 | selector 分箱和可选二峰诊断 |
| `model_package_correction_report.csv` | 单元格 52 | 0.5%/1% 模型包校正报告 |
| `submission_audit.json` | 单元格 53 | 行数、列、ID 顺序、范围和 hash |

## 11. 当前建议阅读顺序

第一轮先看当前可复现 PF：

1. `rogii_clean/src/p2_p01_multiseed_pf.py` 第 57 行附近的粒子内核。
2. 同文件第 466 行附近的输入准备。
3. 同文件第 682 行附近的 128 seed 与四个 scale 汇总。
4. `rogii_clean/scripts/run_p2_p01_multiseed_pf_mean.py` 的逐井缓存入口。

第二轮再回到历史 Notebook：

1. Notebook 单元格 6：profile 和参数。
2. 单元格 11 的 `run_particle_filter()`。
3. 单元格 11 的 `run_pf_lik_ensemble_scales()`。
4. 单元格 11 的 `apply_selector_variant()`。
5. 单元格 26：PF/physical/selector 如何进入 `sub_2`。
6. 单元格 28、30、32：第一条轨道如何落盘。

第三轮再读 Notebook 里的模型轨道：

1. 单元格 14 的 `build_well()`。
2. 单元格 15～25 的五模型和 Ridge。
3. 单元格 33～45 的 learned trajectory。

第四轮最后读：

1. 单元格 48 的同井接触面覆盖。
2. 单元格 50 的可见前缀候选选择。
3. 单元格 52 的 480 特征模型包。

## 12. 暂时不能下的结论

- 不能说“当前高分完全来自 PF”。
- 不能把模型包的 10.6702 OOF 叫作“纯 PF 10.7”。
- 不能把 P2-P02 的 10.3057 叫作纯 PF；它是 41 列单模 LightGBM。
- 不能把当前可核验的 scale 8 裸路径 10.9543 写成 10.7321。
- 不能根据 `input/others/9.349.csv` 的文件名断定当前 Notebook 就是 9.349 方案。
- 不能把同井训练副本覆盖当成对新井可泛化的合法 PF 基线。
- 在完成严格 fold 和缓存来源核对前，不能声称已完整复现 PF 10.7。
