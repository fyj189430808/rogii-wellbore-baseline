# Task 3：实现只补 folds 2–4 的 RF01 稳定性审计脚本

## 任务位置

创建审计脚本和测试，不实际训练 folds 2–4；训练留给 Task 4。不得修改通用 LGBM runner、RF01a 源特征代码或已有 folds 0–1，不使用 Git。

## 必读文件

1. `H:\kaggle\716\docs\superpowers\plans\2026-07-16-agents-and-rf01-stability-audit.md` Task 3
2. `H:\kaggle\716\rogii_clean\configs\rf01_stability_audit_v1.json`
3. `H:\kaggle\716\rogii_clean\experiments\RF01_stability_audit_v1_card.md`
4. `H:\kaggle\716\rogii_clean\scripts\run_simple_lgbm_cv.py`
5. `H:\kaggle\716\rogii_clean\src\metrics.py`
6. `H:\kaggle\716\.superpowers\sdd\task-2-review.md`

## 创建文件

- `H:\kaggle\716\rogii_clean\scripts\run_rf01_stability_audit.py`
- `H:\kaggle\716\rogii_clean\tests\test_rf01_stability_audit.py`

## 固定接口与常量

脚本必须公开：

```python
REUSED_FOLDS = (0, 1)
NEW_FOLDS = (2, 3, 4)

def classify_fold_pattern(fold_deltas: list[float]) -> str: ...
def compute_row_key_hash(frame: pd.DataFrame) -> str: ...
def run_new_folds(..., train_callable=train_fold) -> list[dict]: ...
```

分类固定为：

```python
[-2.6, 0.1, -0.2, -0.3, -0.1] -> "fold1是唯一反向折"
[-2.6, 0.1, 0.2, 0.3, 0.1] -> "fold0是唯一正向折"
[-2.6, 0.1, -0.2, 0.3, -0.1] -> "其余折表现混合"
```

负数表示 RF01a RMSE 低于 B00，即改善。

## 训练前合同检查

脚本必须在调用任何训练前完成：

1. 读取审计配置、RF01a源配置、固定模型配置和fold注册表。
2. 直接读取 `artifacts/RF01a_huber_slopes_v1/feature_cache.parquet`，禁止重新生成其他RF01特征。
3. 验证 feature version=`rf01_huber_slopes_17_v1`，17列列表、773井、3,783,989行、固定fold和允许NaN列。
4. 验证源 fold0/1 predictions、runtime、importance、model 文件存在。
5. 验证B00完整预测存在且也是773井、3,783,989行。
6. 对特征缓存与B00按 `well_id,fold,row_index` 排序，计算确定性行键SHA-256；两个hash必须完全一致。
7. 验证两侧 `target_tvt` 逐位相同，fold分配逐位相同，行键无重复。
8. 将计算得到的 `eval_row_hash` 保存到审计运行清单；不得猜测旧canonical hash。
9. 任一检查失败必须在训练前报错，不能记作负实验。

## 只训练新折

- `run_new_folds` 只能循环配置中的 `[2,3,4]`。
- 调用通用 runner 已有的 `train_fold`，保持固定模型参数、特征列和fingerprint。
- 输出目录为 `artifacts/RF01_stability_audit_v1/fold_2`、`fold_3`、`fold_4`。
- 不创建审计目录下的 `fold_0` 或 `fold_1`。
- 支持中断后通过相同fingerprint复用已完成新折。

## 完整五折汇总

固定来源：

```text
fold0/1：artifacts/RF01a_huber_slopes_v1/fold_0|1/predictions.parquet
fold2/3/4：artifacts/RF01_stability_audit_v1/fold_2|3|4/predictions.parquet
```

合并后必须验证行数、井数、行键唯一、目标一致、预测有限，并保存完整 `predictions.parquet`。

使用项目统一指标生成：

```text
metrics.json
per_well.csv
feature_list.json
parameter_list.json
runtime.json
config.json
feature_importance.csv
comparison_vs_B00/predictions.parquet
comparison_vs_B00/per_well.csv
comparison_vs_B00/metrics.json
conclusion.md
```

与B00比较时，fold delta定义为 `RF01a_RMSE - B00_RMSE`。结论文件第一条必须是：

```text
原 RF01 仍不晋级，本审计不改变原决定。
```

随后只能写三个固定模式之一，不能使用审计结果追认晋级。

## 测试要求（先失败后实现）

至少测试：

1. `REUSED_FOLDS` 与 `NEW_FOLDS` 固定且不相交。
2. 三种折模式分类。
3. 行键hash不受输入行顺序影响，改变任一行键后hash变化。
4. `run_new_folds` 使用假训练函数时只收到2、3、4，绝不收到0、1。
5. 审计配置禁止重新选择和调参，source RF01a版本/列/模型/fold hash一致。
6. 训练前检查能发现B00行键或target不一致。

Focused test command：

```powershell
& 'D:\anaconda\python.exe' -m pytest tests\test_rf01_stability_audit.py -q
```

## 报告

写入 `H:\kaggle\716\.superpowers\sdd\task-3-report.md`，包含RED测试、实现摘要、GREEN测试、脚本未运行训练的证据、自查和顾虑。最终回复 DONE 或 DONE_WITH_CONCERNS 与测试摘要。
