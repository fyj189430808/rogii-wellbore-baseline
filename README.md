# ROGII Wellbore Geology：双轨研究与最终路径管线

这是 ROGII Wellbore Geology Prediction 比赛的学习型研究仓库。

当前采用双轨合同：

1. 特征研究继续固定单个 LightGBM，保持可比性；
2. 最终竞赛管线允许在唯一学习模型后应用无标签、确定性、参数冻结的整井路径处理；
3. 学习型融合必须采用 outer/inner 严格嵌套 OOF。

## 当前状态

仓库同时保留两部分：

- `rogii-dual-track-prefix-calibrated-geosteering.ipynb`：历史高分 Notebook，用于代码考古；
- `rogii_clean/`：以后实验使用的 private-safe 干净基线。

历史 Notebook 中保存的输出来自修改前的 overlap-enabled 运行，其中 14,151 行曾被同井 EGFDU contact 覆盖。源码现在默认关闭：

- cell 26 的 `tvt_phys`；
- cell 47 overlap probe；
- cell 48 guarded contact override；
- cell 50 的 `contact_md_lookup`；
- 空间 imputer 的同 ID 查询井命中。

Notebook 内的旧输出没有被伪装成新结果。重新运行之前，不能把旧 hash 或旧日志当作 private-safe 分数。

## 当前可信结果

| 层级 | 实验 | 范围 | micro RMSE | 状态 |
|---|---|---:|---:|---|
| 原始 LightGBM | `P3B00_group5_p2p02_v1` | 773 井 | `10.305705` | 冻结基线 |
| 确定性路径 | `P3_UP01_robust_u_projection_v1` | 657 开发井 | `9.947068` | 五折均改善 |
| 确定性最终候选 | `P3_UP03_up01_plus_pfs_correction_v1` | 657 开发井 | `9.728976` | folds 3–4 独立确认 |
| 影子验证 | `P4_FINAL00_UP03_v1` | 116 影子井 | 尚未评分 | 候选生成中 |

`9.728976` 是开发集确定性管线成绩，不是 773 井完整 CV；原始模型基线 `10.305705` 不被覆盖。

## 固定 CV

主 CV：`balanced_well_5fold_v1`。

- 773 口完整井；
- 自然隐藏区固定为 `TVT_input.isna()`；
- 共 3,783,989 个评价行；
- 固定 fold 注册表：`rogii_clean/artifacts/folds/balanced_well_5fold_v1.csv`；
- 注册表 SHA-256：`a70bc21e8b91a0ba93e9c94954868adbe00fdd097ea21f7def6dcb749f7a241c`。

详细限制见 [验证与指标文档](docs/07_validation_and_metrics.md)。

## 双轨合同

特征研究轨道只允许一个 LightGBM：

- target：`TVT - last_known_TVT`；
- seed：29；
- 固定 1,734 棵树；
- 不使用 outer-valid early stopping；
- 不调整模型、target 或 fold；
- 不做后处理、stacking 或输出融合；
- 每次只新增一个命名特征组。

最终路径轨道仍然只允许这一个学习模型，但可在其输出后使用冻结的确定性整井算子，包括稳健 U 投影、固定滞后 PFS 和固定比例连续修正。当前唯一候选为：

```text
P4_FINAL00_UP03_v1
= P2-P02
→ U 二次稳健投影（0.50）
→ lag1000 PFS 修正（0.25）
```

任何每井权重、Ridge、第二个 LightGBM 或候选选择器仍须走严格嵌套融合合同。

完整配置见：

    rogii_clean/configs/lgbm_feature_baseline_v1.json

## 目录

    .
    ├── AGENTS.md
    ├── README.md
    ├── requirements.txt
    ├── docs/
    ├── rogii-dual-track-prefix-calibrated-geosteering.ipynb
    └── rogii_clean/
        ├── configs/
        ├── scripts/
        ├── src/
        ├── artifacts/folds/
        ├── experiments/
        └── tests/

原始数据应放在：

    input/data/raw/

`input/` 被 Git 忽略，不会上传到 GitHub。

## 安装

建议使用独立 Python 环境：

    pip install -r requirements.txt

## 生成并核对固定 fold

    cd rogii_clean
    python scripts/make_fixed_folds.py
    python -m pytest tests -q

固定输出：

    rogii_clean/artifacts/folds/balanced_well_5fold_v1.csv

## 统一评分

预测文件需要至少包含：

    well_id
    fold
    target_tvt
    pred_tvt
    carry_tvt

评分：

    cd rogii_clean
    python scripts/score_predictions.py --predictions <OOF预测.csv或parquet>

统一报告：

- micro RMSE；
- macro well RMSE；
- fold RMSE；
- median/P90/worst well RMSE；
- 胜井率；
- 井级 paired bootstrap。

## 当前执行项

唯一候选 `P4_FINAL00_UP03_v1` 已在影子目标打开前冻结。标准影子 OOF 评分和严格排除全部影子井的评分已经同时预登记，只比较 P3B00 与冻结 UP03，不在影子井上搜索其他候选或比例。

## 学习文档

- [项目文件地图](docs/00_project_map.md)
- [PF 入口](docs/01_pf_entrypoint.md)
- [完整数据流](docs/02_pf_data_flow.md)
- [PF 算法](docs/03_pf_algorithm.md)
- [固定 CV 与指标](docs/07_validation_and_metrics.md)
- [全部实验复盘与突破口地图](docs/08_all_experiments_and_breakthroughs.md)

## 研究纪律

所有协作规则以 [AGENTS.md](AGENTS.md) 为准。尤其禁止：

- 随机拆分数据行；
- 用验证井隐藏 TVT 构造特征；
- 在 outer fold 外复用空间/KNN 拟合；
- 偷换 fold、模型、target 或评分行；
- 用一次失败否定整个信息源；
- 未记录就混入模型或后处理；
- 用影子井或榜单重新选择确定性管线参数；
- 使用全局 OOF 预测直接训练外层融合器。

## 2026-07-20 Phase 4 影子确认结果

冻结候选 `P4_FINAL00_UP03_v1` 已完成唯一一次影子验证，并同时通过预注册的标准与严格两套门槛：

| 层级 | 基线 RMSE | 管线 RMSE | 改善 | 结论 |
|---|---:|---:|---:|---|
| 标准影子，116 井 | 10.492112 | 10.097720 | 0.394391 ft | 5/5 折改善，通过 |
| 严格影子，116 井 | 10.956707 | 10.353233 | 0.603474 ft | 5/5 折改善，通过 |

`P4_FINAL00_UP03_v1` 现正式登记为 `shadow_confirmed_pipeline`，对应提交管线编号为 `P4_FINAL_PIPELINE_UP03_v1`。影子集已经消耗，今后不得用于选择参数、特征、比例或研究路线。开发集 `9.728976`、两项影子成绩和 Kaggle 榜单成绩继续分开报告，不能混称为同一个 CV。
