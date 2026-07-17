# Task 3 实施报告：RF01 稳定性审计 runner

## 结论

状态：`DONE`

已实现只复用 RF01a folds 0–1、只补 folds 2–4 的独立审计脚本及聚焦测试。实现会在任何 `train_fold` 调用前完成配置、源缓存、源折、B00、评价行键、目标、逐井 fold 和已有新折复用缓存检查；完整五折汇总使用固定混合来源，并显式以 B00 预测作为比较 baseline。

本 Task 3 没有运行审计主入口，没有训练 folds 2–4，没有创建 `artifacts/RF01_stability_audit_v1`，没有修改通用 runner、统一 metrics、RF01a 特征代码、原 folds 0–1、路线图、AGENTS.md 或 registry，也没有执行 Git。

## 本任务写入的文件

1. `H:\kaggle\716\rogii_clean\scripts\run_rf01_stability_audit.py`
2. `H:\kaggle\716\rogii_clean\tests\test_rf01_stability_audit.py`
3. `H:\kaggle\716\.superpowers\sdd\task-3-report.md`

脚本和测试在实施前均不存在；没有覆盖历史实验产物。

## RED：先失败的测试证据

所有生产行为均先由失败测试锁定，再做最小实现。聚焦命令始终为：

```powershell
& 'D:\anaconda\python.exe' -m pytest tests\test_rf01_stability_audit.py -q
```

主要 RED 阶段如下：

1. 首次测试因 `scripts.run_rf01_stability_audit` 不存在而 collection error，证明固定折常量和三分类接口是新增行为。
2. 依次新增 row-key SHA-256、`run_new_folds`、配置冻结、B00 行键/target 对齐测试；每轮都先因对应函数无法导入而失败。
3. 新增四条预注册选择理由与空/重复行键测试后，得到预期 `2 failed, 32 passed`；随后只补这两项校验。
4. 新增历史 source fingerprint 测试后，先因 `read_source_runtime_contract` 不存在而失败；实现后固定使用 cache metadata 与源 folds 0–1 runtime 三方一致的历史指纹。
5. 新增 matching-fingerprint 新折复用测试后，先因 `validate_existing_new_fold_reuse` 不存在而失败；实现后，任何会被通用 runner 复用的新折都在第一次训练调用前完整验证。
6. 独立复核构造出 `carry_tvt` 漂移和 runtime 缺 `seconds` 的一行缓存；扩展测试后先出现预期失败，再补逐位 carry 校验、target/carry/pred 有限性与非负有限 `seconds` 校验。

测试期间曾遇到一次与产品代码无关的 Windows 系统临时目录 ACL `PermissionError`。测试随即改为纯内存路径记录；需要真实小文件的复用测试使用 `tests` 下自动清理的 `TemporaryDirectory`，没有放宽任何断言。

## 实现摘要

### 固定公开接口

- `REUSED_FOLDS = (0, 1)`
- `NEW_FOLDS = (2, 3, 4)`
- `classify_fold_pattern(...)`
- `compute_row_key_hash(...)`
- `run_new_folds(..., train_callable=train_fold)`

分类严格拒绝非五个数及 NaN/Inf；零归入“其余折表现混合”。行键哈希先规范化 `well_id,fold,row_index`、稳定排序，再按固定文本编码计算 SHA-256；乱序不变，任一键变化必变，空表、空键、非整数和重复键均拒绝。

### 训练前预检

`preflight_audit` 的数据流为：

```text
审计配置 + RF01a 源配置 + 冻结模型配置 + fold 注册表
→ 直接读取 RF01a feature_cache.parquet
→ 验证 17 列、允许 NaN、773 井、3,783,989 行和逐井 fold
→ 验证源 folds 0–1 predictions/runtime/importance/model
→ 验证 B00 完整预测的行键/fold/target
→ 计算真实 eval_row_hash 与 audit fingerprint
→ 完整验证所有会被复用的已有新折
→ 才允许写运行清单和调用 train_fold
```

没有导入或调用任何 RF01 row builder、`load_or_build_feature_cache`、`--rebuild-cache` 或其他 RF01 版本选择逻辑。

### 历史 source fingerprint 的根因与处理

只读复算发现：当前通用 runner 的 `build_experiment_fingerprint` 得到 `7e5a...`，而 RF01a feature cache 与 folds 0–1 runtime 一致保存的是 `311049...`。证据显示 RF01a cache 创建后，通用 runner 又加入了 RF02a/RF02b 等与 RF01a 无关的模块，并把这些模块纳入过宽的通用 source hash；因此不能用后来扩大的源文件集合倒推历史 RF01a 指纹。

实现按 brief 使用正确血缘锚点：`feature_cache.meta.json` 与源 fold 0/1 runtime 的 fingerprint 必须三方一致。新 folds 2–4 仍使用当前确定性的 audit fingerprint；该指纹包含审计/RF01a/模型配置、fold hash、源 cache 大小与 `mtime_ns`、历史 source fingerprint、审计脚本、通用 runner、统一 metrics 和 cache metadata 内容哈希。该策略也会写入训练前 `config.json` 运行清单。

### 新折训练与恢复

`run_new_folds` 只接受配置恰为 `[2,3,4]`，按该顺序复用通用 `train_fold`，没有任何 CLI 参数可以改折、特征或模型参数。相同 audit fingerprint 可恢复已完成新折。

通用 runner 原本只凭 predictions/model/runtime 与 fingerprint 决定复用。审计预检额外验证所有 matching-fingerprint 新折的 importance、model、fold 值、行数、井数、行键、target、carry、有限预测、树数、特征数和 runtime；坏缓存会在任何其他折训练前报错。

### 固定五折汇总

脚本没有调用不适合混合来源的 `finalize_complete_cv`。来源固定为：

- folds 0–1：`artifacts/RF01a_huber_slopes_v1/fold_0|1`
- folds 2–4：`artifacts/RF01_stability_audit_v1/fold_2|3|4`

汇总再次逐折验证 predictions/runtime/importance/model，合并后验证 773 井、3,783,989 行、唯一行键、target 一致和预测有限。`comparison_vs_B00` 显式调用：

```python
build_per_well_metrics(..., baseline_column="b00_pred_tvt")
```

因此不会误用默认 `carry_tvt`。逐折 delta 固定为 `RF01a_RMSE - B00_RMSE`，bootstrap 继续使用统一的井级配对 micro RMSE 算法。五折 feature importance 固定汇总源 0–1 和新 2–4。

`conclusion.md` 只写两行：第一行精确保持原 RF01 不晋级决定，第二行只允许三个预注册模式之一。

## 运行入口与代码阅读指引

Task 4 的运行入口是：

```powershell
Set-Location H:\kaggle\716\rogii_clean
& 'D:\anaconda\python.exe' scripts\run_rf01_stability_audit.py
```

本 Task 3 没有执行该命令。主要输入为审计 JSON、RF01a feature cache、源 folds 0–1 四类文件、B00 OOF 和固定 fold 注册表。模型仍是固定单模 LightGBM，17 列、1,734 棵树、seed 29、无 early stopping。输出位置固定为 `artifacts/RF01_stability_audit_v1`。

建议先读的约 20 行从 `run_new_folds`（脚本约第 1208 行）开始；它最直接展示“只传 folds 2–4 给通用 train_fold”。随后读 `preflight_audit`（约第 849 行）和 `finalize_audit`（约第 1302 行）。核心表 shape：registry 为 773 井一井一行；feature cache 与 B00 都是 3,783,989 个自然隐藏评价行；模型输入是 `[训练行数, 17]`，每折输出是该折完整验证行的绝对 TVT。

易错点是：不得运行 `--rebuild-cache`，不得把当前通用 runner 的扩大 source hash 当作历史 RF01a 指纹，不得把源 0–1 复制进审计目录，不得让 B00 比较使用 metrics 默认 carry baseline。

1–3 井的小样本验证已由纯函数/一行临时 fold bundle 单测覆盖；真实 773 井全量预检和 folds 2–4 训练按任务边界留给 Task 4，不能用小样本替代固定覆盖合同。

## GREEN 与回归证据

最终聚焦测试：

```text
....................................                                     [100%]
36 passed
```

相邻通用 runner 与统一 metrics 回归：

```text
..........................                                               [100%]
26 passed
```

`py_compile` 通过。独立只读代码复核确认当前版本无剩余合同问题，并给出以下 SHA-256：

```text
run_rf01_stability_audit.py  8BB9903B1FFE56EAC273F650D64FF344FCB54BAA481AC263F000D0DDD69E4A92
test_rf01_stability_audit.py 8899A1B2DEF889F1B90A2168268FC4F860C851BFEC1A04DAB03217EB3B5D4F89
```

## 未运行训练与副作用证据

- `Test-Path artifacts\RF01_stability_audit_v1` 返回 `False`。
- 测试只传入 fake train callable；真实 `train_fold` 没有被调用。
- 没有执行 `scripts\run_rf01_stability_audit.py` 主入口。
- 静态扫描没有发现 `load_or_build_feature_cache`、RF01 builder、`--rebuild-cache` 或对 `finalize_complete_cv` 的调用；脚本中仅有一处注释明确说明不调用后者。
- 没有修改或生成 fold 0/1、registry、实验路线图或历史结果。
- 没有使用 Git。

## 简化自查与已知顾虑

脚本体量约 63 KB，主要来自项目要求的中文解释、固定常量和逐层安全错误信息；没有类继承、工厂、注册器、装饰器、依赖注入或元编程。实际主调用链只有：

```text
preflight_audit → run_new_folds → finalize_audit
```

保留的重复验证均有独立目的：训练前阻断坏输入、训练前阻断坏复用 cache、汇总前防止中断后文件被替换。没有为抽象而抽象，也没有重构通用 runner。

已知边界不是实现缺陷：Task 3 明确禁止真实全量运行，因此本报告不声称 773 井预检或 folds 2–4 已实际完成。Task 4 必须先执行脚本内全量预检；任一 fold、row hash、target、cache 或血缘不一致都应按合同异常停止，而不能记成负实验。
