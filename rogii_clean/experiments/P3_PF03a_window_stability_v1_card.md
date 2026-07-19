# P3-PF03a 窗口稳定性卡：250/500/1000 ft

- 实验性质：在查看 folds 0～1 汇总结果前固定的快速稳定性筛查；仍然不是正式 128-seed PF03。
- 唯一变化：`segment_length_ft` 分别固定为 `250`、`500`、`1000`。ESS=`4`、每段最少 50 个原始有效 GR、旧 `gr_sigma`、高斯残差公式、均值路径回退和段中心线性插值全部不变。
- 独立产物：三个窗口分别写入 `P3_PF03a_five_path_local250_proxy_v1`、`local500`、`local1000`，禁止互相覆盖缓存。
- 窗口选择只用 fold 0：先保留 fold 0 相对该折最佳旧固定路径改善至少 `0.20 ft` 的窗口，再选择 fold 0 `local500_rmse` 最低者；完全并列时固定优先级为 500、250、1000。
- fold 1 只做确认：选定后不得因其他窗口在 fold 1 更好而换窗口。选中窗口在 fold 1 不得恶化超过 `0.10 ft`，且 folds 0～1 合并改善必须至少 `0.20 ft`。
- 如果 fold 0 没有任何窗口过门槛，直接判定三窗口代理未获得支持，不利用 fold 1 反向挑选。
- 输出：`artifacts/P3_PF03a_window_stability_v1/pooled_fold01.csv`、`per_fold_fold01.csv` 和 `summary.json`。
- 失败范围：只能否定五条冻结聚合路径在这三个窗口下的局部混合，不能否定正式 128-seed 分段似然 PF03。
