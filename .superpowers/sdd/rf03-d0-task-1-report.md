# RF03-D0 Task 1 实施报告

## 任务边界

本次只完成 `rf03-d0-task-1-brief.md` 规定的冻结配置、实验卡、运行合同和聚焦单测。没有实现 Task 2 的完整 artifact builder、bootstrap 或原子落盘，也没有调用 `main()`、三井 smoke 或 773 井正式诊断。

## 修改文件

1. 创建 `H:\kaggle\716\rogii_clean\configs\rf03_d0_prefix_alignment_v1.json`。
2. 修改 `H:\kaggle\716\rogii_clean\experiments\RF03_D0_prefix_alignment_card.md`。
3. 修改 `H:\kaggle\716\rogii_clean\scripts\diagnose_rf03_prefix_alignment.py`。
4. 修改 `H:\kaggle\716\rogii_clean\tests\test_rf03_prefix_alignment.py`。
5. 创建本报告 `H:\kaggle\716\.superpowers\sdd\rf03-d0-task-1-report.md`。

没有修改 `src/rf03_prefix_alignment.py`、B0、LightGBM runner、模型参数、目标、fold、历史配置、历史实验或 `experiments/registry.jsonl`。

## RED 证据

先只修改测试文件，生产脚本尚未增加新接口时运行：

```powershell
Set-Location H:\kaggle\716\rogii_clean
& 'D:\anaconda\python.exe' -m pytest tests\test_rf03_prefix_alignment.py -q
```

实际结果为退出码 1，测试收集阶段按预期失败：

```text
ImportError: cannot import name 'load_and_validate_diagnostic_config'
from 'scripts.diagnose_rf03_prefix_alignment'
1 error in 3.49s
```

失败原因是 Task 1 要求的配置加载接口尚不存在，不是测试拼写或环境误报；确认 RED 后才创建冻结 JSON 并修改生产脚本。

第一次 GREEN 尝试暴露了本机默认 pytest TEMP 目录的 `WinError 5` 权限问题（12 passed、30 setup errors）。新增测试随后改用工作区内自动清理的临时目录。下一轮只剩一个测试 spy 误拦截合法 registry 读取（41 passed、1 failed）；修正 spy，使其只禁止全量预检完成前读取逐井 CSV。没有为了通过测试放松生产合同。

## 实现摘要

### 冻结配置

- 冻结 `experiment_id = RF03_D0_prefix_alignment_v1`、`experiment_type = diagnostic_only` 和 `baseline_id = B00_simple_lgbm_v1`。
- 冻结 `spatial_pad_1000_v1` 路径、SHA-256、773 井、五个 offset、两个 scope、最低 50 点、bootstrap 次数/seed 和 `hidden_tvt_loaded = false`。
- 新增固定 `train_dir`、smoke 输出目录和 full 输出目录。
- 配置加载逐层比较字段、列表顺序、值和精确 JSON 原生类型；拒绝缺失键、未知键、`bool` 冒充整数和 `50.0` 冒充 `50`。

### 路径、CLI 和运行前合同

- 相对配置路径和配置内路径固定锚定 `rogii_clean`/项目根，不读取当前工作目录。
- CLI 只接受 `--config` 与 `--mode smoke|full`；旧的输入路径、输出路径、offset、scope 和最低点数覆盖口均被移除并由测试确认拒绝。
- smoke/full 都先验证完整 registry 文件、固定 hash、完整 773 井及既有 pad/fold 合同；smoke 之后才按字符串 `well_id` 排序取前三井。
- 对所有选定井的 horizontal/typewell 文件先做全量存在性检查，随后才允许读取第一口井。
- registry/hash/井数/输入文件或计算失败时不会创建 smoke/full 输出目录；`main()` 中目录创建位于全部计算成功之后。

### 纯运行接口

实现并公开：

```python
load_and_validate_diagnostic_config(path: Path) -> dict
run_alignment_for_registry(
    registry: pd.DataFrame,
    train_dir: Path,
    minimum_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]
```

保留既有 `build_summary(offset_scores, margins)` 接口和语义。

`run_alignment_for_registry` 对每井精确使用：

```text
horizontal: MD / GR / TVT_input
typewell:   TVT / GR
```

临时三井测试的水平井 CSV 不含隐藏真值 `TVT`，仍返回 30 行 offset scores（3 井 × 2 scopes × 5 offsets）和 6 行 margins（3 井 × 2 scopes）。测试同时包装 `pd.read_csv`，逐次确认精确 `usecols`，不是只依赖缺列偶然通过。

### 实验卡

实验卡现在明确记录：

- `baseline_id = B00_simple_lgbm_v1 = B0`；
- `diagnostic_only`，不训练模型；
- 不读取水平井隐藏 TVT；
- 精确预注册成功门槛：raw NCC 或 affine-MAE 至少一个指标，必须在 `all_visible`、`tail_1000ft` 两个 scope 各自的 overall 与五个 fold、共 12 组中，让 0 ft 最佳率全部严格 `> 50%`；
- 精确停止条件：raw NCC 与 affine-MAE 两个指标均未满足上述完整 12 组条件；
- 通过只允许进入循环平移负对照/F03a；
- 通过不代表 CV、micro RMSE 或特征晋级已经改善。

## GREEN 输出

聚焦测试的稳定结果：

```text
..........................................                               [100%]
42 passed in 3.03s
```

脚本编译检查退出码为 0：

```powershell
& 'D:\anaconda\python.exe' -m py_compile scripts\diagnose_rf03_prefix_alignment.py
```

独立只读代码审查结论：Critical 0、Important 0、Ready。唯一 Minor 是正式目录建立后若发生写入 I/O 故障可能留下部分文件；完整原子落盘已经明确属于 Task 2，本 Task 不越界实现。

## 未运行正式实验的证据

本次运行过的 Python 命令只有聚焦 pytest 和 `py_compile`。没有执行诊断脚本 CLI，没有调用 `main()`，没有读取真实三井或 773 井逐井数据，也没有训练模型或运行 CV。

正式目录检查结果：

```text
H:\kaggle\716\rogii_clean\artifacts\RF03_D0_prefix_alignment_v1        False
H:\kaggle\716\rogii_clean\artifacts\_smoke\RF03_D0_prefix_alignment_v1 False
```

源码扫描未发现 `build_diagnostic_artifacts`、`bootstrap_alignment_margins`、`bootstrap_replicates` 或 `predictions.parquet` 实现；这些均留给 Task 2。

## 自查

- [x] 先看到缺少新接口导致的 RED，再写生产实现。
- [x] 冻结配置字段、顺序、值、类型和未知键均有测试。
- [x] 相对路径从任意 CWD 解析到同一固定位置。
- [x] CLI 没有路径或诊断参数覆盖入口。
- [x] smoke 先验证完整 registry/hash/773 井，再固定选字符串排序前三井。
- [x] full 对井数不等于 773 的注册表立即拒绝。
- [x] 全部选定井输入文件在逐井读取前完成预检。
- [x] horizontal/typewell 的 `usecols` 精确符合防泄漏白名单。
- [x] 水平井无隐藏 TVT 列仍能运行。
- [x] `build_summary` 保留。
- [x] 实验卡成功/停止门槛与 `build_summary` 的 2 scopes ×（overall + 5 folds）共 12 组、严格 `> 50%` 和 NCC/MAE 逻辑或判据一致。
- [x] 没有创建正式 artifact，没有运行 main/正式数据，没有实现 Task 2。

## 顾虑与后续边界

1. 当前机器系统默认 pytest 临时根目录存在权限异常；本文件的新增测试使用工作区内、测试结束自动清理的临时目录，因此 brief 指定的原始 pytest 命令可以稳定运行。该问题不影响生产 runner。
2. Task 1 有意不提供完整 artifact、bootstrap 和原子落盘。正式 smoke/full 运行在 Task 2 完成并重新验证前仍应保持禁止；不能把当前接口测试通过解释为正式诊断或 CV 已完成。
3. Task 2 应解决只读审查指出的原子落盘 Minor 风险，而不是在本 Task 提前扩大实现范围。
