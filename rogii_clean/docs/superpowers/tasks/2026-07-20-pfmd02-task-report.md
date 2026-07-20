# P3-PFM-D02 实现报告

## 结论

已新增可运行的 P3-PFM-D02 A–F 只读 oracle runner、纯数值核心和两份专项测试。实现没有训练模型，没有改旧代码，没有执行 Git。主代理随后已完成真实 3 井 smoke 和 657 井正式运行；正式运行耗时 101.23 秒，正式数值与独立预览一致且没有变化。

新增文件只有：

- `src/p3_pfmd02_representation_oracle.py`
- `scripts/run_p3_pfmd02_representation_oracle_ladder.py`
- `tests/test_p3_pfmd02_representation_oracle.py`
- `tests/test_run_p3_pfmd02_representation_oracle_ladder.py`
- 本报告

## RED / GREEN 证据

### RED 1：纯数值核心尚不存在

命令：

```powershell
cd H:\kaggle\716\rogii_clean
python -m pytest -q tests/test_p3_pfmd02_representation_oracle.py
```

结果：测试收集失败，`ModuleNotFoundError: No module named 'src.p3_pfmd02_representation_oracle'`。失败原因正是新核心尚未实现。

### GREEN 1：11 类核心合同

同一命令首次实现后得到 13 项通过、1 项测试环境错误。错误来自系统临时目录 `pytest-of-冯禹杰` 无访问权限，业务测试未失败。把纯路径测试改成不访问临时目录后重跑：

```text
14 passed in 6.55s
```

### RED 2：端到端逐井 runner 尚不存在

命令：

```powershell
cd H:\kaggle\716\rogii_clean
python -m pytest -q tests/test_run_p3_pfmd02_representation_oracle_ladder.py
```

结果：测试收集失败，`ModuleNotFoundError: No module named 'scripts.run_p3_pfmd02_representation_oracle_ladder'`。失败原因正是 runner 尚未实现。

### GREEN 2：合成井 A–F 与指标汇总

命令：

```powershell
cd H:\kaggle\716\rogii_clean
python -m pytest -q tests/test_p3_pfmd02_representation_oracle.py tests/test_run_p3_pfmd02_representation_oracle_ladder.py
```

结果：

```text
16 passed in 6.39s
```

### RED / GREEN 3：checkpoint 输入源指纹

自审发现旧 checkpoint 若只校验实验指纹和自然键，无法拒绝同路径下被替换的逐井 mode/seed 文件。因此先增加拒绝源文件指纹变化的测试；RED 为 `cannot import name 'checkpoint_matches'`。实现后从仓库父目录重跑：

```powershell
cd H:\kaggle\716
python -m pytest -q rogii_clean/tests/test_p3_pfmd02_representation_oracle.py rogii_clean/tests/test_run_p3_pfmd02_representation_oracle_ladder.py --basetemp H:\kaggle\716\.pytest_tmp_pfmd02_green
```

结果：

```text
17 passed in 6.55s
```

### 父目录导入回归

第一次从 `H:\kaggle\716` 收集时，复现了顶层 `src` 找不到和环境中同名 `scripts` 抢占的问题。只修改两份新增测试，在导入前把 `rogii_clean` 绝对路径放入 `sys.path[0]`；随后父目录命令通过。runner 自身也在入口显式加入 `rogii_clean`，从父目录执行 `--help` 返回码为 0。

### 编译检查

命令：

```powershell
cd H:\kaggle\716
python -m py_compile rogii_clean/src/p3_pfmd02_representation_oracle.py rogii_clean/scripts/run_p3_pfmd02_representation_oracle_ladder.py rogii_clean/tests/test_p3_pfmd02_representation_oracle.py rogii_clean/tests/test_run_p3_pfmd02_representation_oracle_ladder.py
```

结果：退出码 0，无输出。

## Reviewer checkpoint 与来源指纹修复（2026-07-20）

### 根因

1. `pd.read_csv(segment_checkpoint_path)` 未固定 `well_id` 类型。当一个逐井 checkpoint 只含全数字井号（例如 `00012345`）时，pandas 会推断为整数 `12345`，前导 0 永久丢失；含十六进制字母的井号只是在当前数据上碰巧保留 object。
2. 实验指纹只含 runner 和本诊断核心，没有包含实际参与 Ward 聚类的 `src/p3_pfm01_ordered_pf_modes.py`。
3. seed loader 没有校验 NPZ 的 `_cache_fingerprint` 与 `_format_version`；mode parquet 的 `_cache_fingerprint` 也没有绑定到已验证的 PFM02 `config.json`。
4. checkpoint 的逐井来源签名只有 path/size/mtime，无法发现同大小、同时间戳但内容已变化的文件。

### 修复 RED

- 前导 0 roundtrip/resume 测试先失败：`AttributeError: ... has no attribute 'read_segment_checkpoint'`。
- 缺内容 SHA 的 checkpoint 测试先得到错误的 `True`，证明旧逻辑会接受不完整来源身份。
- 两个 seed 指纹/格式测试均为 `DID NOT RAISE`。
- mode cache 与 mode config 指纹错配测试、PFM01 依赖 hash 测试、mode config 内容 hash 测试均因所需校验入口不存在而失败。

### 最小修复

- 所有 checkpoint CSV 统一从 `read_segment_checkpoint()` 读取，并显式指定 `dtype={"well_id": str}`。审计确认 runner 中只有这一处 checkpoint CSV 读点；另外两个 `pd.read_csv` 是 folds/shadow 冻结输入注册表，不是 checkpoint。
- `_input_signature()` 增加文件内容 SHA256；`checkpoint_matches()` 要求新旧两侧都存在合法 SHA256，来源 hash 变化即 cache miss。
- `load_seed_npz()` 强制每井 NPZ 的 shared fingerprint 为 `91053aa...6468f0`、format version 为 1。
- `load_mode_cache()` 强制每井 parquet 只有一个统一 fingerprint，且必须等于已验证 mode config 的 experiment fingerprint `0e37e94c...e2a8d4`。
- `load_and_validate_mode_config()` 同时校验 config 内容 SHA256、正式井/行数、shadow/hidden-target 合同、shared fingerprint 和记录的 PFM01 core hash。
- 实验/ checkpoint fingerprint 的代码合同现包含 runner、oracle core 和 `p3_pfm01_ordered_pf_modes.py` 三个内容 hash。

### 修复 GREEN

定向来源合同测试：

```text
7 passed, 3 deselected in 19.45s
```

最终从仓库父目录运行两份完整专项测试：

```powershell
cd H:\kaggle\716
python -m pytest -q rogii_clean/tests/test_p3_pfmd02_representation_oracle.py rogii_clean/tests/test_run_p3_pfmd02_representation_oracle_ladder.py --basetemp H:\kaggle\716\.pytest_tmp_pfmd02_reviewer_green
```

结果：

```text
24 passed in 19.08s
```

同轮 `py_compile` 退出码 0。

旧正式 checkpoint 只有 path/size/mtime，不满足新安全合同，因此不会迁移或伪装成 cache hit。主代理将先安全重建一次 657 井 checkpoint，再运行第二次验证 657/657 全命中；这一过程只重写新 D02 目录的 checkpoint/audit/runtime，不改变已完成的正式 metrics 数值定义。

## 实现摘要

### 纯数值核心

- B0 枚举四候选的 15 个非空 active subset；每个 subset 内消去“权重和为 1”约束后解最小二乘，只接收非负可行解，没有 OLS 后 clip。
- B25 严格实现 `q=0.75*e_P2+0.25*v`，`v` 仍由四维 simplex 精确求解。
- MD 分段以首个隐藏 MD 为起点，窗口为 250/500/1000 ft。
- DP 使用固定 `max(1, round(L/median_positive_md_step))`，分别返回原始 SSE、含惩罚目标、切换次数和回溯状态；平局优先保持状态，再按 P2/low/middle/high。
- 固定二维 `[整段 mean_delta, endpoint_delta]`、不标准化、Ward K=3；簇内 scale8 softmax 先减成员最大 LL，避免极端跨模式 LL 下溢。
- rowmedian 与 medoid 都沿用同一固定成员；medoid 只从真实成员 seed 中选，平局取最小 seed id，未使用隐藏真值定义。
- 自然键、fold、行数和 float32 MD 对齐均为硬校验；shadow 交集为硬拒绝。

### runner

- P2 parquet 使用 Arrow filter 按开发井过滤后才转 pandas。
- 每次只持有一口井的 `128 × hidden_rows` seed 数组，不构造全开发集 seed 矩阵。
- A4、A3、B0、B25、三个窗口的 independent/DP、D128、D128+P2、三种固定成员代表和两类逐行包络一次逐井完成。
- 所有 delta 都先加 `last_visible_tvt` 才与绝对 TVT 比较；重算 center 必须与 PFM02 legal cache 在 `rtol=1e-7, atol=1e-6` 内一致。
- 每井 checkpoint JSON 和对应 segment CSV 都原子写；恢复时同时验证实验指纹、自然键、行数以及该井 mode/seed 文件签名。
- 正式运行硬校验 657 井、3,211,872 行、shadow 交集 0、开发集 P2 RMSE `10.272146267501086`。
- 正式汇总还会用主代理独立快速预览的 A3/A4/D128/F-mode3/F-seed128 数值做 `1e-5 ft` 回归检查。
- 最小产物为 `config.json`、`input_audit.json`、`metrics.json`、`per_well.csv`、`per_fold.csv`、`per_segment.csv`、`runtime.json`、`conclusion.md`；smoke 写入 `smoke_1/2/3`，与正式根目录物理隔离。

## 自审

逐项核对任务简表：

- A–F 全部实现，D128 没有被 D128+P2 改写，F 明确标记为不可部署 coverage。
- 固定比较只在预注册参照之间计算；没有在 B0/B25、代表方法或窗口之间事后取最优。
- 每法逐井 SSE/RMSE、pooled micro、macro、median、P90、最差井和逐折 micro 均有输出。
- B0/B25 权重、active 数、D128 seed id、三模式 medoid seed id、逐段四 SSE/状态/margin、DP 原始 SSE/penalized objective/switch count 均有保存位置。
- `conclusion.md` 固定区分“数据直接证明的事实、合理推断、未验证猜测、只能否定的实现、下一步最便宜验证”。
- oracle 输出只进入新诊断目录；配置明确写 `formal_feature_or_cache_output=false`。
- 未修改 `current_state.json`、`registry.jsonl`、旧 runner、旧 src 或任何旧 artifact。

## 已知风险与正式复验检查点

1. 主代理已完成真实 3 井 smoke 和 657 井正式运行（101.23 秒），真实 parquet/NPZ 值域与 Windows 长时 I/O 已通过；本次安全合同修复不改变任何 oracle 公式。
2. 正式 P2 先经 Arrow 过滤再整体转 pandas，约 321 万行；128 seed 始终逐井，但 P2 pandas 表本身仍会占用一段稳定内存。
3. 正式 A3/A4/D128/F 回归参考来自独立快速预览的 6 位小数，runner 使用 `1e-5 ft` 绝对容差；若主代理保存了更高精度参考，可在新 runner 内收紧常量，不应改旧产物。
4. 当前 MD 分段要求每井 MD 单调不减且存在正步长。已完成的真实 smoke/正式运行没有触发空窗口异常；若未来来源版本出现大于一个完整窗口的无行空洞，需先写合成回归测试再改变 DP 语义。
5. 新来源身份合同会安全拒绝旧 checkpoint。首次修复后正式运行用于重建，第二次必须验证 657/657 cache hit；不得为省一次运行而迁移缺 SHA 的旧 checkpoint。

## 主代理复验命令

真实 3 井 smoke 已完成；修复后可用同一命令复核来源合同：

```powershell
cd H:\kaggle\716
python rogii_clean/scripts/run_p3_pfmd02_representation_oracle_ladder.py --max-wells 3
```

正式全量已完成且数值不变。修复后同一命令需连续执行两次：第一次安全重建 checkpoint，第二次验证全命中。

```powershell
cd H:\kaggle\716
python rogii_clean/scripts/run_p3_pfmd02_representation_oracle_ladder.py
```
