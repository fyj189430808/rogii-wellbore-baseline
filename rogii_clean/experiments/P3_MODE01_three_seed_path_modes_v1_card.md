# P3-MODE01 实验卡：128 条 seed 路径的三个固定模式

- 实验编号：`P3_MODE01_three_seed_path_modes_v1`
- 实验性质：确定性路径诊断；本轮不训练 LightGBM，不打开影子集。
- 唯一假设：128 条 PF seed 路径可能包含三个有方向的稳定模式；三个模式中心比逐行无方向标准差更能表达路径分叉。
- 与当前路径基线唯一不同之处：不修改 PF，也不重新生成 seed；只把已有 128 条完整 seed 路径固定聚成三个模式。
- 合法输入：`P3_shared_pf_seed_paths_v1` 的 `seed_delta/row_index/hidden_md/last_tvt/final_ll/seed_ids`；共享缓存字段、dtype、shape、格式版本和指纹必须完全匹配。
- 聚类表示：每条路径按归一化 MD 采 64 点；减去 128 条路径的逐点均值；不做 zscore；Ward 层次聚类固定 `K=3`。
- 模式中心：簇内 seed 完整路径的简单算术平均，不用似然再次加权。
- 模式质量：先用 `softmax(final_ll / 8)` 得到冻结 scale8 权重，再对簇内权重求和。
- 模式编号：`mass` 降序、簇大小降序、终点 delta 升序、最小 seed id 升序。
- 分叉位置：64 点中首次连续 3 点满足 `abs(mode1-mode2) >= max(1 ft, 最大绝对分离的 25%)` 的归一化进度；找不到记为 `1.0`。
- 稳定性：固定随机种子 `20260719`，重复 8 次无放回抽 64/128 seed；中心用 Hungarian 最小 RMSE 匹配，报告中心 RMSE 与 ARI，仅作诊断。
- 固定开发集：排除 116 口影子井，只允许 657 口开发井；目标只在全部正式路径缓存写完后读取。
- smoke：固定优先使用 `000d7d20/00bbac68/015fe0d2` 中已有共享缓存的最多三口井。
- oracle：比较三条模式中心、旧 mean/scale3/5/8/12 和逐井最佳模式；所有 oracle 结果只写 `oracle/`，不得进入正式特征缓存。

## 首批八项正式特征

```text
pf_mode1_delta
pf_mode2_delta
pf_mode3_delta
pf_mode1_mass
pf_mode2_mass
pf_mode12_margin
pf_mode12_separation
pf_mode_split_fraction
```

本轮不得混入模式 3 mass、簇大小、簇内离散度、稳定性结果或 oracle 选择结果。

## 诊断通过依据

完成 folds 0～1 后，至少出现以下证据之一才考虑进入正式 LightGBM：

1. 某条固定模式中心自身优于现有最佳固定 PF 路径；
2. 逐井最佳模式 oracle 明显优于旧五路径 oracle；
3. `mode1-mode2` 的有向分离能稳定区分 P2-P02 正负残差；
4. 随机半数 seed 的模式中心和成员关系仍稳定。

smoke 只验证实现、字段、路径形态和诊断能否生成，不据此判断路线成败。

## 运行命令

```powershell
python rogii_clean/scripts/run_p3_mode01_three_seed_path_modes.py --mode smoke
python rogii_clean/scripts/run_p3_mode01_three_seed_path_modes.py --mode fold01
python rogii_clean/scripts/run_p3_mode01_three_seed_path_modes.py --mode all
```

`fold01/all` 不生成共享 seed 缓存；若缓存不齐，会直接提示先续跑 PF03。
