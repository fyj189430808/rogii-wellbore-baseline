# ROGII clean：private-safe 单模特征研究

这个目录只保存已经明确理解、可以独立解释的基础设施。

当前阶段只完成：

1. 固定 spatial-pad 五折；
2. 固定统一评分；
3. 固定单模 LightGBM 合同；
4. 禁止直接同井标签/contact 通道；
5. 规划单因素特征实验。

当前没有训练新模型，也没有宣称复现 PF 10.7。

生成固定 fold：

    python scripts/make_fixed_folds.py

检查预测文件：

    python scripts/score_predictions.py --predictions <预测文件>

研究硬约束：

- 验证单位是完整井；
- 同一 spatial pad 不能跨 outer fold；
- 验证井及其 pad 不能参与空间/KNN/标准化拟合；
- 测试同 ID 训练井必须从最终训练和邻井索引中排除；
- 模型固定为一个 LightGBM；
- target 固定为 `TVT - last_known_TVT`；
- 每次只改变一个特征组；
- 不使用 learned trajectory、model package、stacking 或输出融合。
