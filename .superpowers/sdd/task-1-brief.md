# Task 1：精简重写项目协作规则

## 任务位置

这是整个实施计划的第一步。只修改根目录 `AGENTS.md`，不要修改其他项目文件，不运行实验，不使用 Git。

## 必读文件

1. `H:\kaggle\716\docs\superpowers\specs\2026-07-16-agents-experiment-rules-design.md`
2. `H:\kaggle\716\docs\superpowers\plans\2026-07-16-agents-and-rf01-stability-audit.md` 中 Task 1
3. `H:\kaggle\716\rogii_clean\experiments\feature_roadmap_v2.md`
4. 当前 `H:\kaggle\716\AGENTS.md`

## 必须实现

- 把 `AGENTS.md` 从约759行的“PF理解优先”规则，完整精简重写为“特征实验优先”规则。
- 唯一目标：完成 `rogii_clean/experiments/feature_roadmap_v2.md` 全部实验，在固定单模 LightGBM、固定参数、固定 spatial-pad 五折下寻找最低且可信的 micro RMSE。
- 按 v2 的两阶段执行：先完成 S 系列边际价值，再按完整五折晋级结果逐步完成 C 系列条件价值；不得一次合并全部特征。
- 始终只优化特征，不换模型、不融合、不调 LightGBM 参数或树数。
- 默认连续推进：数据检查或正对照 → 1～3井检查 → fold0 → fold1 → 完整五折。
- 保留固定CV、统一门槛、防泄漏、缓存指纹、实验留档和长任务进度规则。
- 尽量使用中文和普通说法；无法替代的代码名、文件名、LightGBM、RMSE、CV可以保留。
- 删除 PF 10.7 复现和八份导读文档作为新实验门槛的规定。
- 不主动执行 Git、worktree、提交、推送或PR。
- 明确原 RF01 八版本继续“不晋级”，不得事后改判。
- 明确 `RF01_stability_audit_v1` 固定使用 RF01a，只补 folds 2–4，无论结果如何不能追认晋级。

## 固定章节

按以下顺序组织：

1. 当前唯一目标
2. 固定研究边界
3. feature_roadmap 执行顺序
4. 固定 CV 与晋级门槛
5. 实验前后记录
6. 防止数据泄漏
7. 缓存与复现
8. 长任务运行
9. 中文表达与代码可读性
10. RF01 原结论与稳定性审计
11. Git 与非实验操作
12. 完成标准

## 验证

运行冲突扫描，确认旧禁令已经删除；再搜索 `feature_roadmap_v2`、`单模 LightGBM`、`micro RMSE`、`RF01_stability_audit_v1`、`不得事后改判` 和 `中文`，确认关键规则存在。

## 报告

把完整报告写到：

`H:\kaggle\716\.superpowers\sdd\task-1-report.md`

报告必须包含：修改摘要、验证命令与结果、自查发现、是否存在顾虑。最终只返回 `DONE` 或 `DONE_WITH_CONCERNS` 和一句测试摘要。
