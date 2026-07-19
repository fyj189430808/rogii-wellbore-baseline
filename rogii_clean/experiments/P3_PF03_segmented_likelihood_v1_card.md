# P3-PF03 实验卡：128 条 seed 路径的分段似然软加权

- 实验编号：`P3_PF03_segmented_likelihood_v1`
- 实验性质：正式确定性路径实验；不是五路径代理，也不训练模型。
- 唯一假设：同一口井的不同 MD 位置可能应信任不同的 PF seed 路径，沿 MD 缓慢变化的局部权重会优于整井固定权重。
- 与当前路径基线唯一不同之处：保留冻结 P2-P01 的 GR 预处理、sigma、粒子状态转移、128 个 seed 和随机数；只把整井固定 seed 权重改成局部窗口权重。
- 合法输入：当前井 `MD/Z/GR/TVT_input`、配对 Typewell `TVT/GR`、冻结 PF 参数。隐藏 `TVT` 只在全部路径写完后用于评分。
- 窗口：`250/500/1000 ft`，三条路径都完整保留，不用 folds 0～1 在三者中事后挑窗口参数。
- 窗口中心步长：`100 ft`。
- 局部权重：每窗按 seed 路径对应 TVT 上的 Typewell GR 残差计算局部 log-likelihood，二分温度固定 `ESS=16`。
- 沿 MD 平滑：在相邻窗口中心之间线性插值 `log weight`，再逐行 softmax。
- 缺失规则：局部累计只使用原始有限 GR；窗口没有任何原始 GR 时使用 128 条 seed 的均匀权重，因此长缺口会回到 PF mean。
- 整井核对：每井冻结 PF 返回的 128 个 final LL 必须与 D01 缓存逐位一致，否则该井失败且禁止评分。
- 共享缓存：每井额外原子写入 `artifacts/P3_shared_pf_seed_paths_v1/<well>.npz`，保存 `seed_delta float32[128,N]`、`row_index int32[N]`、`hidden_md float32[N]`、`last_tvt float64`、`final_ll float64[128]`、`seed_ids int32[128]` 和配置指纹，供 MODE01 直接复用。
- 正对照：冻结旧 `scale3/5/8/12` 四条整井固定温度路径。
- 负对照：均匀 PF mean；GR 循环错移、反转和窗口权重打乱在路径晋级后再运行，不进入本轮特征。
- 固定开发集：排除 116 口未打开影子井，只用 657 口开发井。
- folds 0～1 门槛：合并 micro RMSE 至少改善 `0.20 ft`，任一折不得恶化超过 `0.10 ft`。
- 通过后：才运行开发集完整五折并进入单模 LightGBM 特征替换实验。
- 失败只能否定：冻结旧 GR/sigma 与 seed 路径下，`250/500/1000 ft + 100 ft 步长 + ESS16 + 当前局部 GR 残差`这一版分段权重。
- 失败不能否定：其他局部观测模型、其他 ESS、前后向路径平滑、多峰路径表示或严格 OOF 残差学习。
- 输出目录：`artifacts/P3_PF03_segmented_likelihood_v1/`。
- 中断恢复：路径和共享 seed 数据均按井原子缓存，重复同一命令只补缺失井。

## 运行命令

```powershell
python rogii_clean/scripts/run_p3_pf03_segmented_likelihood.py --mode smoke
python rogii_clean/scripts/run_p3_pf03_segmented_likelihood.py --mode fold01
python rogii_clean/scripts/run_p3_pf03_segmented_likelihood.py --mode all
```
