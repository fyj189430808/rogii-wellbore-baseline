# ROGII Wellbore Geology：private-safe baseline

这是 ROGII Wellbore Geology Prediction 比赛的学习型研究仓库。

当前目标不是继续堆模型，而是：

1. 看懂并复现现有 PF 思路；
2. 去除直接同井标签/contact 通道；
3. 冻结可信的 spatial-pad CV；
4. 固定一个 LightGBM；
5. 每次只研究一个新数据特征组。

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

## 当前没有可信的最终 CV

历史结果中：

- 最佳单 LightGBM OOF 约为 10.4733；
- Ridge stack OOF 约为 10.4197；
- Model Package 后处理 OOF 约为 10.6702。

这些结果使用普通井级 GroupKFold，且空间/KNN 特征没有在每个 outer fold 内完全重建；最终高分链也没有完整 OOF。因此它们只作为历史参考，不是新基线成绩。

新的固定 CV 已建立，但尚未训练 LightGBM，所以仓库目前没有宣称新的模型 CV。

## 固定 CV

主 CV：

    median-XY 1000-unit connected-pad CV v1

定义：

1. 每口井取完整轨迹的中位 X/Y；
2. 代表点距离不超过 1000 个原始坐标单位时连边；
3. 连通分量作为不可拆分的 pad；
4. 290 个 pad 按隐藏行数确定性平衡到五折。

| fold | wells | pads | hidden rows |
|---:|---:|---:|---:|
| 0 | 147 | 57 | 757,050 |
| 1 | 155 | 58 | 756,990 |
| 2 | 157 | 58 | 756,649 |
| 3 | 155 | 58 | 757,061 |
| 4 | 159 | 59 | 756,239 |

固定 fold 注册表 SHA-256：

    0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab

详细限制见 [验证与指标文档](docs/07_validation_and_metrics.md)。

## 固定单模 LightGBM

后续特征实验只允许一个 LightGBM：

- target：`TVT - last_known_TVT`；
- seed：29；
- 固定 1,734 棵树；
- 不使用 outer-valid early stopping；
- 不调整模型、target、fold 或后处理；
- 不使用 CatBoost、TCN、TabICL、stacking 或输出融合；
- 每次只新增一个命名特征组。

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

    rogii_clean/artifacts/folds/spatial_pad_1000_v1.csv
    rogii_clean/artifacts/folds/spatial_pad_1000_v1.meta.json

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

## 特征实验顺序

完整实验卡见 [feature_roadmap.md](rogii_clean/experiments/feature_roadmap.md)。

顺序固定为：

1. 前缀 `U=TVT_input+Z` 多窗口倾角；
2. GR 缺失与插值可靠性；
3. 多尺度 horizontal/typewell GR 对齐；
4. 当前井可见前缀 self-template；
5. 冻结 PF 的不确定性；
6. 合法 prefix-cut 可靠性统计；
7. fold-safe 空间 KNN/局部平面。

在 carry、冻结 PF 和 B0 单模 LightGBM 稳定复现前，不开始 F01。

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
- 未记录就混入模型或后处理。
