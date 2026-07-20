# P3-PFM02 Task 2 实现报告

## 状态

DONE

本任务只新增 brief 允许的 config、runner、专项测试和本报告；未修改旧 runner、训练器、PFM01、PF03 或 P3B00，未运行正式训练或真实全量缓存生成，未执行 Git 操作。

## RED 记录

### 第一轮：runner 缺失

命令：

```powershell
python -m pytest rogii_clean/tests/test_run_p3_pfm02_direct_mode_paths_cv.py -q
```

预期失败：

```text
ImportError: cannot import name 'run_p3_pfm02_direct_mode_paths_cv' from 'scripts'
1 error during collection
```

确认失败原因是 brief 指定的新 runner 尚不存在。随后只实现首条测试所需的最小 41+3 特征合同，专项测试变为 `1 passed`。

### 第二轮：其余合同缺失

补齐冻结配置、模型、cache/runtime、自然键合并、禁止特征和自动停止测试后，再次运行同一专项命令：

```text
1 passed, 32 failed
```

失败原因是新 config 尚不存在，以及 `validate_frozen_contract`、`merge_pfm02_legal_cache`、`parse_fold_spec`、`remaining_folds_after_screen` 等合同函数尚未实现，符合预期 RED。

## GREEN 与验证

专项测试：

```powershell
python -m pytest rogii_clean/tests/test_run_p3_pfm02_direct_mode_paths_cv.py -q
```

结果：

```text
33 passed in 15.49s
```

专项、旧 PF02 target/paths 及 Task1 cache 回归：

```powershell
python -m pytest rogii_clean/tests/test_run_p3_pfm02_direct_mode_paths_cv.py rogii_clean/tests/test_p3_pf02_target_ess_lgbm.py rogii_clean/tests/test_p3_pf02_target_ess_paths.py rogii_clean/tests/test_generate_p3_pfm02_mode_path_cache.py -q --basetemp rogii_clean/artifacts/_pytest_tmp/pfm02_task2_basetemp
```

结果：

```text
62 passed in 22.74s
```

首次回归未指定 `--basetemp` 时，旧 Task1 测试因 Windows 用户临时目录权限问题出现 18 个 setup error；切换到工作区临时目录后全部通过，未为此改代码。

静态编译：

```powershell
python -m py_compile rogii_clean/scripts/run_p3_pfm02_direct_mode_paths_cv.py rogii_clean/tests/test_run_p3_pfm02_direct_mode_paths_cv.py
```

结果：退出码 0，无输出。

## 新增文件

- `rogii_clean/configs/p3_pfm02_direct_mode_paths_v1.json`
- `rogii_clean/scripts/run_p3_pfm02_direct_mode_paths_cv.py`
- `rogii_clean/tests/test_run_p3_pfm02_direct_mode_paths_cv.py`
- `rogii_clean/docs/superpowers/tasks/2026-07-19-pfm02-task2-report.md`

## 实现摘要

- 冻结 `P3B00_FEATURES` 与 P3B00 41 列清单逐位一致，只在末尾追加 `pf_mode_low_delta`、`pf_mode_middle_delta`、`pf_mode_high_delta`，共 44 列。
- 冻结实验编号、fold、井数、行数、逐折井/行数、基线 OOF SHA、数据来源 SHA、门槛和完整 LightGBM 参数/训练策略。
- 复用现有 PF02 runner 的开发/影子反连接、Arrow 过滤、B00+F05a 加载、旧 P01 五路径合并、自然键 anchor 合并、训练、配对评分、bootstrap、产物和自动停止骨架。
- Task1 cache 只接受精确八列 Arrow schema；逐井验证井号、fold、行数、重复键、统一非空 fingerprint、runtime fingerprint、`hidden_tvt_read=false` 及 shadow 反连接。
- 三列按 `(well_id,row_index)` 一对一合并，保持原行序；沿用 float32 anchor 逐位相等和有限值检查。
- `all` 始终先训练/重算 folds 0～1，仅在 folds01 晋级后返回 folds 2～4。
- 结论模板明确：负结果只能否定 low/middle/high 三路径直接追加到冻结单模 LightGBM 的当前实现。

## 自审

- 44 列无重复，原 41 列未替换或删除。
- 正式特征不含 direction、mass、seed_count、p2_position、separation、oracle、target、surface 或 geology 摘要。
- config 不指向 PFM01 `oracle/`，runner 没有读取该目录的路径。
- 正式 cache 合并没有 runtime 绕过参数；缺失或读过隐藏 TVT 必然失败。
- 输出覆盖 config、44 列 feature list、parameter list、leakage audit、逐折训练产物、folds01/full5 评分产物、runtime 和 conclusion。
- 没有打开影子目标，也没有把影子井目标读入 pandas。

## 担忧与验证边界

按 brief 明确禁止正式训练，因此本报告没有真实 657 井训练结果，也没有对正式 artifact 的运行时/磁盘占用作经验验证。runner 的合同、合并和控制流已由合成测试与旧回归覆盖；正式训练前仍应按项目流程先做 1～3 井 smoke。除此之外没有已知实现担忧。
