# P3-PFM02 Task 1：无标签三模式合法缓存

## 任务位置

这是 P3-PFM02 的第一项实现。训练 fold 0/1 之前，必须为全部 657 口开发井生成统一的低/中/高三条 PF 模式路径缓存。旧 PFM01 只覆盖 fold 0，不能作为全开发集正式缓存。

## 只允许新增

- `rogii_clean/scripts/generate_p3_pfm02_mode_path_cache.py`
- `rogii_clean/tests/test_generate_p3_pfm02_mode_path_cache.py`
- 实现报告：`rogii_clean/docs/superpowers/tasks/2026-07-19-pfm02-task1-report.md`

不要修改 PFM01、PF03、P3B00 或其他旧代码。不要运行真实 657 井数据。

## 冻结输入与语义

- fold：`artifacts/folds/balanced_well_5fold_v1.csv`
- shadow：`artifacts/P3_shadow_holdout_v1/shadow_holdout.csv`
- 只选排除 116 shadow 后的 657 口、3,211,872 行。
- PF128：`artifacts/P3_shared_pf_seed_paths_v1/<well_id>.npz`
- shared fingerprint：`91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0`
- 每井必须有 `seed_delta[128,N]`、`final_ll[128]`、`seed_ids=0..127`、`row_index[N]`、`last_tvt[1]`。
- 复用 `src.p3_pfm01_ordered_pf_modes`：二维 `[mean_delta,end_delta]`、无标准化 Ward K=3、scale8 LL 簇内加权中心、按中心均值→endpoint→min seed id 命名 low/middle/high。
- 不需要也禁止读取 P2-P02、target、TVT 真值、PFM01 oracle、direction/mass/seed_count/p2_position/separation。

## 输出合同

默认 artifact：`artifacts/P3_PFM02_mode_paths_v1/`

每井 `legal_cache/<well_id>.parquet` schema 严格等于：

```text
well_id
fold
row_index
last_visible_tvt
pf_mode_low_delta
pf_mode_middle_delta
pf_mode_high_delta
_cache_fingerprint
```

每井 `legal_runtime/<well_id>.json` 至少保存：井号、fold、行数、shared NPZ SHA、shared fingerprint、生成器 SHA、PFM01 core SHA、实验 fingerprint、`hidden_tvt_read=false`、是否 cache hit、耗时。

总产物至少保存：`config.json`、`feature_list.json`、`parameter_list.json`、`legal/per_well.csv`、`runtime.json`。正式完成必须锁死 657 井、3,211,872 行、0 shadow，所有井相同 fingerprint。

CLI：

```text
--max-wells 1..3   独立 smoke 子目录
不传               正式全部657井
--workers           默认8
--output-dir
```

逐井原子写入，严格命中才复用；缺任一 shared NPZ 时在打开任何标签之前失败并列出缺井。进度逐井打印。

## TDD 必测

1. 先写测试，运行并看到因为新模块不存在而失败；在报告记录 RED 命令与核心错误。
2. 合成 128 路径验证输出只含三条低/中/高，和 PFM01 核心结果逐位一致。
3. 正式 schema 精确、禁止 target/true/residual/error/rmse/oracle/best/mass/direction/position/separation/seed_count。
4. 错 fingerprint、127 seed、错 seed_ids、重复/错 row_index、非有限值、fold/行数错误、shadow 缓存均失败。
5. cache hit 必须验证 schema、井号、fold、行数、fingerprint 和 runtime；错误缓存不能静默复用。
6. smoke 与正式目录物理隔离。
7. 生成器测试中把任何目标读取函数设为调用即抛错，合法生成仍成功。

完成后运行专项 pytest 和 py_compile。不要 Git commit。把 RED/GREEN 命令、结果、文件、自审和任何担忧写入报告文件，只在消息中简短返回 DONE 或 DONE_WITH_CONCERNS。
