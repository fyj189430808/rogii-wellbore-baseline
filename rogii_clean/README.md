# ROGII clean：双轨实验目录

本目录同时保存：

1. `feature_ablation_cv`：冻结单个 LightGBM 的特征消融；
2. `deterministic_pipeline_development`：唯一学习模型之后的无标签确定性整井路径；
3. `nested_fusion_cv`：必须 outer/inner 严格嵌套的学习型融合；
4. `shadow_confirmed_pipeline`：一次性影子确认后的最终管线。

当前原始单模基线是 `P3B00` 的 `10.305705`；657 口开发井上冻结 UP03 管线为 `9.728976`，不能冒充 773 井 CV。唯一影子候选为 `P4_FINAL00_UP03_v1`。

生成固定 fold：

    artifacts/folds/balanced_well_5fold_v1.csv

检查预测文件：

    python scripts/score_predictions.py --predictions <预测文件>

研究硬约束：

- 验证单位是完整井；
- fold 固定为 `balanced_well_5fold_v1`；
- 验证井不能参与空间/KNN/标准化拟合；
- 测试同 ID 训练井必须从最终训练和邻井索引中排除；
- 模型固定为一个 LightGBM；
- target 固定为 `TVT - last_known_TVT`；
- 特征消融每次只改变一个特征组且禁止后处理；
- 最终路径允许冻结的无标签确定性整井算子；
- 学习型权重、stacking 或第二模型必须严格嵌套 OOF；
- 影子井只评价预先冻结的 P3B00 与 `P4_FINAL00_UP03_v1`。

## Phase 4 当前确认结果

`P4_FINAL00_UP03_v1` 已同时通过标准和严格影子门槛：

- 标准影子：`10.492112 → 10.097720`，改善 `0.394391 ft`，5/5 折改善；
- 严格影子：`10.956707 → 10.353233`，改善 `0.603474 ft`，5/5 折改善；
- 最终类别：`shadow_confirmed_pipeline`；
- 提交管线：`P4_FINAL_PIPELINE_UP03_v1`；
- 影子集状态：已打开并永久锁定，不再参与任何选择。

当前可提交管线仍只含一个学习模型：P2-P02 LightGBM；之后固定执行 U 二次稳健投影 0.50 和 lag1000 PFS 修正 0.25。原始预测与最终预测必须分别保存。
