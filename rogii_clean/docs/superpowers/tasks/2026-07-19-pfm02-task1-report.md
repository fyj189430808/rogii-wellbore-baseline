# P3-PFM02 Task 1 实现报告：无标签三模式合法缓存

## 范围与限制

- 新增生成器：`rogii_clean/scripts/generate_p3_pfm02_mode_path_cache.py`
- 新增专项测试：`rogii_clean/tests/test_generate_p3_pfm02_mode_path_cache.py`
- 未修改 PFM01、PF03、P3B00 或其他既有代码；未进行 Git 操作。
- 未运行真实 657 井数据。全部验证使用 pytest 临时目录内的合成 2 口开发井 + 1 口影子井、每井 5 行、128 seed NPZ。

## TDD 记录

### RED

先只新增测试文件，未创建生成器。命令：

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q
```

核心错误：

```text
ImportError: cannot import name 'generate_p3_pfm02_mode_path_cache' from 'scripts'
```

这证明测试因待实现的新模块不存在而失败，而不是测试本身已覆盖既有行为。

### GREEN（首轮历史结果，已被后续审查修复取代）

实现最小生成器后，第一次以工作区 pytest 临时目录运行时，10 项通过、1 项失败：127 个 `seed_ids` 的错误信息没有明确写出 `128`。随后将该分支拆为“数量必须是 128”和“编号必须是 0..127”两个明确校验，再次运行：

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q --basetemp H:\kaggle\716\.pytest-pfm02
python -m py_compile scripts\generate_p3_pfm02_mode_path_cache.py
```

当时结果（历史记录，不是最终验证）：

```text
...........                                                              [100%]
11 passed in 8.72s
```

`py_compile` 退出码为 0，未输出错误。

默认 pytest 临时目录因 Windows 用户临时目录权限被拒绝，首次绿色运行未进入测试主体；显式 `--basetemp` 后已完成真实的专项测试执行。

## 已实现的合同

- 读取折表与影子清单，仅选择非影子的开发井；正式调用锁定 657 井和 3,211,872 行，运行摘要锁定 0 影子井。
- 正式调用使用固定输入位置，默认输出 `artifacts/P3_PFM02_mode_paths_v1/`；`--max-wells 1..3` 输出到物理隔离的 `smoke_<n>/`。
- 当前最终合同：先检查所选 shared NPZ 文件齐全；worker 内逐井验证 `seed_delta[128,N]`、`final_ll[128]`、`seed_ids=0..127`、严格递增且无重复的任意起点 `row_index[N]`、`last_tvt[1]` 与有限值。
- 模式路径直接复用 PFM01 核心：`[mean_delta, end_delta]` 无标准化 Ward K=3，scale-8 LL 簇内加权中心，按中心均值、末端值、最小 seed id 排成 low/middle/high。
- 每井 parquet 严格写为要求的八列及类型；无标签读取路径，运行时记录 `hidden_tvt_read=false`。
- 每井 parquet 与 JSON 均原子写入。cache hit 在复用前验证 schema、井号、fold、行数、row_index、fingerprint、数值有限性和 runtime 必填字段；不合格缓存不会被复用。
- 总产物写入 `config.json`、`feature_list.json`、`parameter_list.json`、`legal/per_well.csv`、`runtime.json`；每井写入 `legal_cache/` 与 `legal_runtime/`。

## 专项测试覆盖

1. 128 条合成路径生成的三条路径与 PFM01 核心逐位一致。
2. parquet 列名、列顺序与 Arrow 类型严格匹配，且没有任何禁止字段词。
3. 缺 NPZ、127 seed、错误 seed id、重复 row index、非有限值均拒绝。
4. 错 fold、错行数、影子井缓存、错 fingerprint 与不完整 runtime 均拒绝。
5. smoke 与正式输出目录物理隔离。
6. 将 `pandas.read_parquet` 替换为立即抛错的“目标读取”桩后，合法生成仍成功；生成器实际仅读取折表、影子清单和 shared NPZ，并用 PyArrow 校验/读取自身缓存。

## 自审

| 检查项 | 结论 |
|---|---|
| 真实标签、TVT 真值、P2-P02、PFM01 oracle 是否读取 | 否 |
| 方向、mass、position、separation、seed_count 是否进入 cache schema | 否 |
| 影子井是否进入选择、缓存或汇总 | 否 |
| 是否可在不满足严格合同的旧缓存上静默命中 | 否 |
| 是否运行真实 657 井数据 | 否，按任务要求未运行 |
| 是否修改允许清单以外的源码/测试/报告文件 | 否 |

## 结论

首轮历史事实（已被后续 14 项、18 项验证替代）：当时合成输入下 11 项专项测试通过，模式值逐位等于 PFM01 核心，编译通过。

基于事实的合理推断：生成器按冻结的无标签输入和模式语义写出符合合同的缓存，并会拒绝测试覆盖的损坏或不匹配缓存。

仍然没有验证的猜测：尚未对真实的 657 口井、3,211,872 隐藏行 shared NPZ 完整运行，无法在本任务内证明真实输入全部满足任意起点、严格递增且唯一的 `row_index[N]` 与正式数量锁定。

当前实验只能否定的具体实现：它未评估 P3-PFM02 的路径质量、ESS 加权、模型 RMSE 或任何影子集指标；这些不属于 Task 1。

下一步最便宜的验证：由主流程在正式环境先运行 `--max-wells 1` smoke（输出将隔离到 `smoke_1/`），检查进度、runtime 和 schema 后再决定是否执行完整 657 井缓存生成。

## 主流程真实集成 smoke

主代理随后从仓库父目录执行：

```powershell
python rogii_clean/scripts/generate_p3_pfm02_mode_path_cache.py --max-wells 3
```

入口修复后命令退出码为 0，真实生成 `000d7d20`、`00bbac68`、`00e12e8b` 三口井，输出物理隔离在 `artifacts/P3_PFM02_mode_paths_v1/smoke_3/`。因此前文“未运行真实数据”只描述实现代理交付时的状态，不再代表主流程当前验证状态。

---

## 审查修复（Task 1 第二轮）

### 修正的事实

先前报告中“`row_index=0..N-1`”的说法不正确，现已被以下冻结语义取代：

- PF seed 的行数严格对应 fold 表的 `hidden_rows`，不对应 `total_rows`。
- `row_index` 必须长度为 `N`、严格递增、唯一；保留 shared NPZ 给出的任意自然起点。真实井 `000d7d20` 的预期形式为 `1442..5277`。
- cache hit 不再只检查行数或人为构造的行号：它与当前 shared NPZ 的 `row_index` 逐位比较，并要求 runtime 的 `shared_npz_sha256` 等于当前 NPZ SHA。

### 本轮 TDD：RED

先扩展专项测试，再修改生成器。命令：

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q --basetemp H:\kaggle\716\.pytest-pfm02-r2
```

结果：`14 failed`。核心失败分别证明：

- 合成 fold 的 `hidden_rows=5`、`total_rows=9` 时，旧实现把 `total_rows` 当作 PF 行数，报 `seed_delta must have exactly 128 seeds and expected rows`；
- 旧正式调用仍暴露 `formal_expected_wells/formal_expected_rows` 绕过参数；
- 非正式测试在不传 `max_wells` 时被旧的可绕过正式检查阻断；
- 新 cache-hit 调用所需的 NPZ 行号与 SHA 合同尚未实现。

总耗时字段同样遵循一次独立 RED/GREEN：删除该字段后新增断言，命令：

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py::test_changed_shared_npz_sha_prevents_cache_hit -q --basetemp H:\kaggle\716\.pytest-pfm02-r2b
```

核心错误：`KeyError: 'total_elapsed_seconds'`。

### 实现修复

- fold 选择器改为要求和使用 `hidden_rows`；正式总行数也按隐藏行求和。
- shared NPZ 检查接受任意起点的严格递增 `row_index`，并在 cache hit 时逐位比较当前 NPZ 的值。
- `shared_npz_sha256` 纳入 runtime 命中条件；改变有效 NPZ 内容后旧缓存会重建，不能静默命中。
- 移除了 `formal_expected_wells` 与 `formal_expected_rows` 参数。`max_wells=None` 现在强制检查：fold 表 773 口、影子集 116 口且均在 fold 表、开发/影子零交集、657 开发井、3,211,872 隐藏行。
- `--max-wells 1..3` 保持 smoke 行为和物理隔离。
- 预检仅检查全部所选 shared NPZ 文件是否存在；每个 worker 才加载、校验、生成并在函数返回后释放单井数组，不再预加载全部 657 口的约 3.3GB `seed_delta`。
- `runtime.json` 增加 `total_elapsed_seconds`；移除了未使用的 `MODE_NAMES` 导入。

### 本轮 GREEN（第二轮历史结果，已被最终 18 项验证取代）

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q --basetemp H:\kaggle\716\.pytest-pfm02-r2
python -m py_compile scripts\generate_p3_pfm02_mode_path_cache.py
```

当时结果（历史记录，不是最终验证）：

```text
..............                                                           [100%]
14 passed in 11.40s
```

`py_compile` 退出码为 0。

新增覆盖包括：非零自然起点 `1442..1446`、`hidden_rows != total_rows`、修改 shared NPZ 后 SHA 不同且 cache-hit 为 false、正式调用无绕过参数并拒绝非 773 口的正式合同，以及总耗时字段。

### 更新后的结论

第二轮历史事实（已被最终 18 项验证替代）：14 项合成专项测试通过；缓存路径逐位复用 PFM01 核心；隐藏行数、自然 row index、共享 SHA 和正式合同均受测试覆盖；编译通过。

基于事实的合理推断：在真实 shared NPZ 满足冻结字段合同的前提下，正式调用不会把全井 `total_rows` 错当隐藏路径行数，也不会复用已被修改的 shared NPZ 对应的旧缓存。

仍然没有验证的猜测：按任务要求未运行真实 657 井，因此尚未以真实输入证明所有 NPZ 的自然 `row_index` 和 SHA 都满足合同。

当前实验只能否定的具体实现：该任务仍不评估路径质量、ESS、残差模型、RMSE 或影子集成绩。

下一步最便宜的验证：主流程仅运行一次真实 `--max-wells 1` smoke，检查该井的 `hidden_rows`、cache `row_index`、runtime SHA 和总 runtime；通过后才考虑完整 657 井运行。

---

## 最终边界修复：原始整数向量验证

### TDD：RED

新增 4 个参数化测试，分别向 shared NPZ 写入：非整数浮点 `seed_ids`、NaN `seed_ids`、非整数浮点 `row_index`、NaN `row_index`。先运行：

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py::test_raw_seed_ids_and_row_index_reject_fractional_or_nonfinite_values -q --basetemp H:\kaggle\716\.pytest-pfm02-r3
```

结果：`4 failed`。核心证据：`0.5` 的 seed id 和 `1442.5` 的 row index 在旧实现中被静默截断并成功生成；NaN 先触发 int64 转换警告，之后才以不精确的语义错误失败。

### 修复

新增原始整数向量验证，严格依序执行：

1. 原数组必须一维；
2. dtype 必须是整数或浮点数值型；
3. 原数值必须全为有限值；
4. 必须处在 int64 范围且每个值为整数值；
5. 仅通过以上检查后才转换为 `int64`。

因此非整数浮点和 NaN 不会再被截断或在转换时产生警告。该规则同时应用于 `seed_ids` 与 `row_index`。

### GREEN（第三轮历史结果，已被入口回归后的 19 项验证取代）

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q --basetemp H:\kaggle\716\.pytest-pfm02-r3
python -m py_compile scripts\generate_p3_pfm02_mode_path_cache.py
```

结果：

```text
..................                                                       [100%]
18 passed in 12.51s
```

`py_compile` 退出码为 0。

### 第三轮历史结论（已被入口回归后的最终结论取代）

数据直接证明的事实：最终 18 项合成专项测试通过，涵盖三模式逐位一致、隐藏行数、任意自然 row index、共享 SHA 失效、正式合同、总耗时，以及原始整数向量的非整数/NaN 拒绝；编译通过。

基于事实的合理推断：生成器不会把浮点整数向量静默截断为合法 `seed_ids` 或 `row_index`，并仍只使用无标签的 fold、shadow 和 shared PF NPZ 输入。

仍然没有验证的猜测：未运行真实 657 井数据；真实输入对最终合同的逐井满足性仍须由后续 smoke 验证。

当前实验只能否定的具体实现：不涉及 Arrow 损坏文件的额外韧性，也不评价路径质量、ESS、残差模型、RMSE 或影子集成绩。

下一步最便宜的验证：主流程按既定边界运行一次真实 `--max-wells 1` smoke，核对该井的 hidden 行数、自然 row index 和 runtime SHA。

---

## 入口修复：从仓库父目录直接执行（第四轮历史，已被 20 项验证取代）

### TDD：RED

新增 subprocess 回归测试：以工作区父目录 `H:\kaggle\716` 为 cwd，直接运行：

```powershell
python H:\kaggle\716\rogii_clean\scripts\generate_p3_pfm02_mode_path_cache.py --help
```

测试先失败，核心错误为：

```text
ModuleNotFoundError: No module named 'src'
```

错误发生在生成器第 20 行的 `from src...`，未读取真实数据。

### 唯一修复

在 `from src...` 之前定义：

```python
CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))
```

并让既有 `PROJECT_ROOT` 复用 `CLEAN_ROOT`。除此以外没有改变缓存、验证、CLI 参数或数据读取逻辑。

### GREEN（第四轮历史结果，已被 20 项验证取代）

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q --basetemp H:\kaggle\716\.pytest-pfm02-r4
python -m py_compile scripts\generate_p3_pfm02_mode_path_cache.py
```

结果：

```text
...................                                                      [100%]
19 passed in 20.87s
```

`py_compile` 退出码为 0。新增入口测试从仓库父目录执行 `--help` 成功，且只验证 Python 导入与 argparse 帮助，不运行真实 smoke 或读取真实数据。

### 第四轮历史结论（已被 20 项验证取代）

数据直接证明的事实：最终 19 项合成/入口专项测试通过，涵盖此前全部无标签缓存合同以及从仓库父目录的 CLI 导入；编译通过。

基于事实的合理推断：用户报告的 `python rogii_clean/scripts/generate_p3_pfm02_mode_path_cache.py --max-wells 3` 不再会因 `src` 模块路径缺失而在导入阶段失败。

仍然没有验证的猜测：按任务要求，未运行真实 3 井或 657 井数据，真实输入的共享 NPZ 完整性仍待后续 smoke 检查。

当前实验只能否定的具体实现：只修复入口模块路径；不改变 Arrow 损坏韧性、路径质量、ESS、残差模型、RMSE 或影子集成绩。

下一步最便宜的验证：主流程可按既定边界运行一次真实 `--max-wells 1` smoke。

---

## 最终数值稳定性修复：极端跨模式 LL

### TDD：RED

新增极端合成 shared NPZ：128 条路径仍按原有三组低/中/高轨迹分布，但三组 `final_ll` 分别约为 `-6861.414`、`-3000`、`-693.582`。先运行：

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py::test_extreme_cross_mode_likelihoods_produce_finite_three_mode_centers -q --basetemp H:\kaggle\716\.pytest-pfm02-r5
```

结果：失败于旧 PFM01 全局 softmax 后的 `RuntimeError: 模式质量必须是有限正数`。这复现了 mode 间 LL 差足够大时，低 LL Ward 簇总权重下溢为零的边界。

### 修复

未修改 `src.p3_pfm01_ordered_pf_modes`。仅 PFM02 generator 新增内部中心计算：

- Ward K=3 仍直接复用 PFM01 的 `cluster_seed_descriptors`；
- 描述仍是整段均值与末端值、无标准化；
- 对每个已固定成员簇，计算 `exp((ll - max_ll_in_that_mode) / 8)` 后在该簇内归一化；
- 仍按中心整段均值、末端值、最小 seed id 排为 low/middle/high；
- 只返回三条中心路径，未增加质量或其他 cache 列。

该簇内归一化与全局 scale-8 权重再除以簇质量在数学上等价；不同模式之间的公共归一化常数会抵消。其区别仅是避免极端模式质量在浮点数中下溢为零。

既有正常 LL 回归测试改用严格 `allclose(rtol=1e-12, atol=1e-12)` 与旧 `summarize_ordered_modes` 比较，以允许数学等价实现的末位浮点舍入差异。

### 最终 GREEN

```powershell
python -m pytest tests/test_generate_p3_pfm02_mode_path_cache.py -q --basetemp H:\kaggle\716\.pytest-pfm02-r5
python -m py_compile scripts\generate_p3_pfm02_mode_path_cache.py
```

结果：

```text
....................                                                     [100%]
20 passed in 16.43s
```

`py_compile` 退出码为 0。

### 最终结论

数据直接证明的事实：最终 20 项合成/入口专项测试通过；正常路径中心与旧 PFM01 核心在 `1e-12` 容差内一致，极端跨模式 LL 差下低/中/高三条中心均可生成且为有限值；编译通过。

基于事实的合理推断：对发现的 `4a8ecc0b` 这类跨模式 LL 范围，PFM02 cache generator 不会因无关模式的全局 softmax 下溢而拒绝一个有效 Ward 簇。

仍然没有验证的猜测：未重新运行真实 smoke 或正式 657 井，故尚未在真实 `4a8ecc0b` 文件上复核其 cache 产物；按任务边界没有读取该井数据。

当前实验只能否定的具体实现：修复只覆盖 PFM02 三模式中心的数值稳定性，不改变 PFM01 本体，也不评价路径质量、ESS、残差模型、RMSE 或影子集成绩。

下一步最便宜的验证：主流程可从中断点重试正式 cache 生成，或先运行真实 `--max-wells 1` smoke；生成器会在该模式中心计算处使用簇内稳定权重。
