# P3-D01 Task 4A 完成报告

## 本任务边界

本任务只实现了内存表格上的路径评分与井级统计纯函数：

- 没有读取真实比赛数据；
- 没有修改 D01 runner；
- 没有训练模型；
- 没有生成可供 LightGBM 使用的正式特征；
- 没有执行 Git 提交、推送或 worktree 操作。

## 新建文件

- `rogii_clean/src/p3_d01_diagnostic_analysis.py`
- `rogii_clean/tests/test_p3_d01_diagnostic_analysis.py`
- `docs/superpowers/tasks/2026-07-19-p3-d01-task4a-report.md`

## TDD 的 RED 记录

先只新建测试文件，然后运行：

```powershell
python -m pytest rogii_clean/tests/test_p3_d01_diagnostic_analysis.py -q --basetemp .pytest_tmp_d01_task4a_red
```

实际结果：

```text
ModuleNotFoundError: No module named 'src.p3_d01_diagnostic_analysis'
1 error during collection
```

失败原因符合预期：生产模块尚未创建，测试确实能阻止功能缺失时误判为完成。

## 实现内容

### 1. 单井路径评分

`score_well_paths` 会先审计：

- 真值只属于一口井和一个 fold；
- 两侧 `row_index` 都唯一；
- 两侧键集合完全相同；
- 评分相关数值全部有限。

它按 `row_index` 严格一对一连接，不依赖输入行顺序。四条温度路径严格按旧 P2-P01 公式计算：

```text
float64(last_visible_tvt) + float64(scale_delta)
```

每条路径输出 SSE 和 RMSE。四种温度平手时固定按 `3、5、8、12` 选择第一个。oracle 字段只出现在评分结果里。

`aggregate_path_metrics` 以总 SSE 除以总评价行数计算 pooled RMSE，同时输出 macro、中位井、P90 和最差井 RMSE。`oracle_scale` 被明确标记为事后上限。

### 2. Spearman 与置换比例

`spearman_report` 使用平均秩计算 Spearman，并只在井之间打乱 y。双侧置换比例严格使用：

```text
(1 + 超过或等于观测绝对相关的次数) / (permutations + 1)
```

不足 3 对或任一变量为常数时，汇总字典返回 `valid=False`，并用 `None` 保存 rho 和 p，确保 `json.dumps(..., allow_nan=False)` 可以直接写出。

`correlation_tables` 真正分别计算 overall 与各 fold；随机种子由基础 seed、变量对顺序和 fold 通过 SHA-256 稳定派生，不使用 Python 随机 hash。

### 3. 诊断列、稳定分箱和路线判断

`add_diagnostic_columns` 按实验卡原公式生成：

- GR 缺失率；
- 全部 GR sigma 相对仅观测 GR sigma 的绝对差；
- 四个温度下 ESS 的严重/中等塌缩标记；
- scale 3 在简单井中过尖锐的标记。

它会拒绝非有限或不在 `[0, 1]` 内的 `observed_gr_fraction`，并要求四个温度的 ESS 都位于 128 粒子理论允许的 `[1, 128]` 区间。

`build_binned_metrics` 用“变量值 + 井号”稳定排序后平衡分成四组，因此输入换行顺序和重复变量值不会随机改变分组。每组输出 pooled RMSE、四温度 ESS 中位数和塌缩比例。

`decide_pf_routes` 逐项执行实验卡中的 PF01/PF02 阈值，并返回每个子条件的实际数值、是否通过和总判断。返回值明确限定为 `whole_well_only`，不把整井累计似然误写成沿 MD 的位置证据。

## GREEN 与验证记录

加入实现和输入边界测试后，重新运行：

```powershell
python -m pytest rogii_clean/tests/test_p3_d01_diagnostic_analysis.py -q --basetemp .pytest_tmp_d01_task4a_green2
```

首轮实际结果：

```text
25 passed in 3.55s
```

静态复核又补出了两个边界：ESS 的理论区间，以及 `path_metrics.path_name` 不能重复。先加测试后运行：

```powershell
python -m pytest rogii_clean/tests/test_p3_d01_diagnostic_analysis.py -q --basetemp .pytest_tmp_d01_task4a_red_boundaries
```

实际观察到 3 个预期失败：`ESS=0.99`、`ESS=128.01` 未拒绝，以及无关路径名重复未拒绝。加入最小边界检查后运行：

```powershell
python -m pytest rogii_clean/tests/test_p3_d01_diagnostic_analysis.py -q --basetemp .pytest_tmp_d01_task4a_green_boundaries
```

实际结果：

```text
30 passed in 3.00s
```

语法编译验证：

```powershell
python -m py_compile rogii_clean/src/p3_d01_diagnostic_analysis.py rogii_clean/tests/test_p3_d01_diagnostic_analysis.py
```

实际结果：退出码 0，无输出。

测试覆盖了：乱序键手算、重复/缺失/多余键、非有限输入、温度平手、pooled SSE、可复现置换、折方向、无效相关、固定派生公式、非法观测率、ESS 理论边界、稳定四分组、重复路径名、决策阈值和 JSON 安全。

## 独立审查后的第二轮 RED/GREEN

独立审查指出两个边界仍需直接锁定：重复的 DataFrame index 不能被当作行身份，以及相关表中的非有限数不能流入决策 JSON。先补测试后运行：

```powershell
python -m pytest rogii_clean/tests/test_p3_d01_diagnostic_analysis.py -q --basetemp .pytest_tmp_d01_task4a_review_red
```

实际观察到：

```text
10 failed, 32 passed
```

其中 9 个失败来自有效相关行没有拒绝非法 rho、p 或方向计数，1 个失败来自空 `per_well` 没有显式拒绝。随后把重复 index 测试加强为“同一 index 标签跨越四个分组”，旧实现也按预期失败，只剩第 4 组。

最小修复包括：

- 分箱前内部 `reset_index(drop=True)`，只按行位置回填分组；
- 所有调用方用位置掩码选行，不再让重复 index 参与对齐；
- 有效相关行的 rho 必须是 `[-1,1]` 内有限数，p 必须是 `[0,1]` 内有限数；
- `same_direction_folds` 和 `valid_folds` 必须是 `[0,5]` 内整数，且前者不能超过后者；
- 无效相关行把 rho/p 安全转换为 `None`；
- 空逐井表显式报错，并补齐逐井 SSE、行数、缺失率及路径指标边界检查。

修复后运行：

```powershell
python -m pytest rogii_clean/tests/test_p3_d01_diagnostic_analysis.py -q --basetemp .pytest_tmp_d01_task4a_review_green
```

实际结果：

```text
42 passed in 3.19s
```
