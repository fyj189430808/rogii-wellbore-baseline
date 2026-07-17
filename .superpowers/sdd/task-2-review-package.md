# Task 2 复核材料

## 需求与规则

- `H:\kaggle\716\.superpowers\sdd\task-2-brief.md`
- `H:\kaggle\716\AGENTS.md` 第9章
- `H:\kaggle\716\docs\superpowers\plans\2026-07-16-agents-and-rf01-stability-audit.md` Task 2

## 实施产物

- `H:\kaggle\716\rogii_clean\experiments\RF01_stability_audit_v1_card.md`
- `H:\kaggle\716\rogii_clean\configs\rf01_stability_audit_v1.json`
- `H:\kaggle\716\.superpowers\sdd\task-2-report.md`

## 复核重点

1. 规格符合性：实验身份、固定RF01a、复用0/1、只跑2/3/4、四条理由、原结论不变是否一致。
2. 配置一致性：路径、fold hash、井数、行数、模型配置是否与真实RF01a一致。
3. 是否存在事后选择、调参、重跑0/1或创建artifact的暗示。
4. 顾虑是否被正确限定为Task3运行前的row hash检查，而不是擅自制造hash。
5. 文档质量：中文是否清楚，字段是否足够供审计脚本使用。

只读复核，不使用Git，不修改项目文件。输出规格结论、质量结论和按严重程度的问题清单到 `.superpowers/sdd/task-2-review.md`。
