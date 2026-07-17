# RF01_stability_audit_v1 实验卡：RF01a 跨折稳定性审计

## 预注册身份

- 实验编号：`RF01_stability_audit_v1`
- 实验名称：RF01a folds 2–4 稳定性审计
- 实验类型：独立诊断（`diagnostic_only`），不参与原 RF01 晋级。
- 固定对象：`RF01a_huber_slopes_v1`。
- 原 RF01 决策：八个版本继续“不晋级”，本审计无论结果如何都不得事后改判或追认晋级。

## 唯一问题与选择理由

- 唯一假设：补齐预先固定的 RF01a folds 2–4 后，逐折方向能够回答 fold 1 是唯一异常折，还是 fold 0 才是异常折。
- 为什么值得验证：这项独立诊断只用于理解 RF01a 的跨折波动，并为未来新实验提供设计依据；它不重新打开原 RF01 的候选筛选或晋级判断。
- 审计问题：fold 1 是唯一异常折，还是 fold 0 才是异常折？
- 预先锁死的选择理由：
  1. 公式最简单。
  2. fold 1 损失最小。
  3. 不含派生程度更高的差值、曲率和无界外推。
  4. 选择依据不使用 folds 2–4。

## 唯一变化与禁止事项

- 与已完成 RF01a 唯一不同之处：直接复用 folds 0–1 的既有预测，只新训练同一 RF01a 的 folds 2–4，以形成固定五折诊断。
- 复用 folds：`[0, 1]`。
- 新运行 folds：`[2, 3, 4]`。
- 禁止事项：不选择其他 RF01 版本；不调整窗口、特征、LightGBM 参数、1,734 棵树、seed、fold、目标、评价行、评分代码或后处理；不重跑或覆盖 RF01a folds 0–1。

## 冻结模型、特征与参数

- 本次使用的模型：固定单模 `lightgbm.LGBMRegressor`，配置为 `configs/lgbm_feature_baseline_v1.json`。
- 本次使用的参数：完全继承冻结模型配置；`n_estimators=1734`，`random_state=29`，不使用 early stopping，不调参。
- 特征版本：`rf01_huber_slopes_17_v1`。
- 本次使用的 17 个特征：`last_visible_tvt`、`md_since_visible_end`、`hidden_fraction`、`x_current`、`y_current`、`z_current`、`dx_from_visible_end`、`dy_from_visible_end`、`dz_from_visible_end`、`dxy_from_visible_end`、`gr_raw`、`gr_missing`、`u_huber_slope_50`、`u_huber_slope_100`、`u_huber_slope_200`、`u_huber_slope_500`、`u_huber_slope_1000`。
- 特征选择：固定使用上述五个 Huber slope 和十二个基线特征，不重新生成、筛选或替换其他 RF01 特征版本。

## 合法输入与数据血缘

- 源配置：`configs/rf01a_huber_slopes_v1.json`。
- 源产物目录：`artifacts/RF01a_huber_slopes_v1`。
- 源特征缓存：`artifacts/RF01a_huber_slopes_v1/feature_cache.parquet`。
- 比较基线预测：`artifacts/B00_simple_lgbm_v1/predictions.parquet`。
- 合法性边界：只读取固定 RF01a 特征缓存、既有 folds 0–1 预测、固定 fold 注册表和 B00 预测；不读取其他 RF01 版本结果来选择对象，不使用隐藏 TVT 构造或选择特征。
- 预测目标：`target_delta`，即固定的 `TVT - last_known_TVT`（源 RF01a 列名为 `last_visible_tvt`）；最终预测为最后可见 TVT 加 LightGBM 预测增量。

## 固定 CV、评价行与比较

- 固定 fold：`spatial_pad_1000_v1`。
- fold 注册表：`artifacts/folds/spatial_pad_1000_v1.csv`。
- fold hash：`0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab`。
- 固定评价行：与 RF01a 和 B00 相同的自然隐藏行；预期 773 口井、3,783,989 行。执行审计时必须按 `well_id, fold, row_index` 与 B00 对齐并验证目标一致，不得改变评价行集合。
- row hash：源 RF01a 配置未单独保存此字段；本登记不生成替代值。后续审计运行必须从固定行键计算并与 B00 核对，若不一致即按合同异常停止。
- 比较基线及 baseline_id：固定 B00，预测文件为 `artifacts/B00_simple_lgbm_v1/predictions.parquet`。

## 对照、覆盖与停止边界

- 正对照：不新增特征正对照；仅做 RF01a 与固定 B00 的逐折配对诊断。
- 等维或同结构负对照：不适用。本审计不提出新特征组，也不参与晋级；不得借审计结果替代原 RF01 的既有对照结论。
- oracle 诊断：不使用，不读取隐藏真实 TVT 来选择版本、窗口、特征或参数。
- 覆盖门槛：合并 folds 0–4 后必须恰为 773 口井、3,783,989 个固定评价行，行键唯一、目标一致且预测有限。
- 成功门槛：完成固定 RF01a 五折诊断并回答预注册问题；没有晋级门槛，结果不能改变原 RF01“不晋级”结论。
- 停止条件：源配置、模型配置、fold hash、评价行/目标、源缓存或 folds 0–1 产物任一不匹配时，按合同异常停止；不得训练替代版本或自行修补输入。
- 预计运行时间：本 Task 2 只登记文件且不训练；后续主要耗时仅来自 folds 2–4 的三次固定 LightGBM 训练。
- 本任务需要生成的文件：`experiments/RF01_stability_audit_v1_card.md` 和 `configs/rf01_stability_audit_v1.json`；本任务不得创建 artifact。
- 失败后能否否定整个方向：不能。
- 失败后只能否定哪一种实现：只能说明这次固定 RF01a 跨折稳定性诊断无法在合同内完成，或其余折不支持某一种异常折解释；不能否定前缀斜率信息源，也不能改判原 RF01。

## 结果解释边界

- 数据直接证明的事实：在审计实际运行前，没有新增折结果。
- 基于事实的合理推断：在固定合同下补齐 folds 2–4 可以区分跨折方向模式。
- 仍然没有验证的猜测：fold 1 或 fold 0 哪一个是异常折。
- 当前实验只能否定的具体实现：固定 `RF01a_huber_slopes_v1` 的跨折稳定性解释。
- 下一步最便宜的验证：后续审计 runner 复用 folds 0–1，只训练 folds 2–4，并与 B00 做固定行级配对比较。
