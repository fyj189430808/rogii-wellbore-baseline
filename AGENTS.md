# ROGII 项目协作规则（精简执行版）

> 本文件只规定 Codex 的默认执行行为和不可违反的硬边界。  
> 具体特征定义、实验顺序和门槛以 `rogii_clean/experiments/feature_roadmap_v2.md` 为准。  
> 详细审计规范保留在原完整协议中，只有命中“深度审计触发条件”时才展开执行。

---

## 1. 当前目标与不可变合同

当前目标：在固定单模 LightGBM 下，寻找最低且可信的自然隐藏行 pooled micro RMSE。

以下内容默认冻结，除非用户明确批准修改：

- CV：`spatial_pad_1000_v1`
- target：`TVT - last_known_TVT`
- 模型：单个 LightGBM
- 参数：固定参数、1,734 棵树、seed 29、无 early stopping
- 评价行：`TVT_input.isna()`
- 不改变训练范围、fold、评分代码和后处理
- 不调 PF/Beam 参数
- 不做模型融合、stacking、learned trajectory 或 model package
- 不使用同井训练 TVT、surface、contact 或测试期不可获得的信息
- 每次只改变一个特征组

违反以上合同的改动必须先停下并询问用户。普通特征实验不需要重复复述这些规则。

---

## 2. 默认原则：先做实验，不做长篇事前审查

除非命中第 5 节的深度审计条件，Codex 应直接进入实验：

```text
确认实验编号
→ 实现唯一特征变化
→ 1～3 井 smoke test
→ fold 0
→ 达标后 fold 1
→ 再达标后完整五折
```

禁止在简单实验开始前：

- 重新通读整个仓库
- 重写路线图已有的实验卡
- 重复解释固定模型、fold、target 和评分方式
- 先生成完整报告和全部 artifact
- 先跑全项目测试
- 对未修改模块进行大范围代码审查
- 因为“可能存在风险”而无限扩展检查范围

**快速实验应在开始处理后 5 分钟内进入代码修改或运行。**  
如果 5 分钟内没有发现明确合同异常，就必须开始 smoke test 或 fold 0，不得继续做泛化检查。

---

## 3. 三类实验通道

### 3.1 快速通道

满足以下条件时使用：

- 特征只依赖当前井测试期可见数据
- 不需要用训练标签拟合统计量
- 不建立邻井、surface、OOF 或 learned cache
- 不修改 fold、target、模型和评分代码
- 只是新增或调整一个局部特征组

典型实验：

- GR 缺失 mask
- 当前井局部 GR 统计
- 可见前缀 U 统计
- 当前井与 Typewell 的固定公式相似度
- 已冻结 PF 路径的纯诊断统计

快速通道只做四项预检：

1. 实验编号和唯一新增特征组明确
2. fold hash 与冻结基线一致
3. 评价 row hash 与冻结基线一致
4. 新特征代码没有读取隐藏 TVT

四项通过后立即运行。

### 3.2 折安全通道

出现以下任一情况时使用：

- 特征需要 outer-train 训练井拟合
- 使用邻井标签、surface、contact 或空间索引
- 使用 OOF 模型输出或折相关缓存
- 使用训练集统计量进行标准化、插补、选择或阈值拟合
- 修改 PF/Beam cache 的生成或读取逻辑

此时才运行完整的：

- outer-fold source exclusion
- cache lineage
- hidden TVT deletion/mutation invariance
- 验证 pad 排除
- 单井 cache 归属检查

### 3.3 合同变更通道

涉及以下内容时必须暂停并等待用户批准：

- 更换模型
- 调 LightGBM 参数或树数
- 更换 target、fold 或评分方式
- 输出融合或 stacking
- 新增测试期不可用信息
- 大规模项目重构
- 不可逆操作

---

## 4. 路线图执行规则

实验顺序以 `feature_roadmap_v2.md` 为准，不在本文件重复展开。

每个实验只需在运行日志中记录：

```text
experiment_id
baseline_id
唯一新增特征组
代码入口
预计主要耗时
```

路线图已有实验卡时，不再重新抄写完整实验卡。

固定门槛：

- fold 0 改善不足 `0.15 ft`：停止当前实现
- fold 0 通过但 fold 1 反向：停止当前实现
- folds 0–1 都通过：自动完成五折
- 五折晋级仍按路线图中的统一标准判断

正常失败只停止当前实现，保存简要结果后继续下一项，不等待用户重复发送“继续”。

---

## 5. 深度审计触发条件

只有出现以下情况，才允许执行耗时较长的完整审计：

1. fold hash、row hash、井数或评价行数不一致
2. 正式特征可能读取隐藏 TVT 或其派生量
3. outer-fold cache 来源不明或与当前 fold 不匹配
4. 使用训练标签拟合空间、邻井、surface、标准化或阈值
5. 一个方案准备晋级为完整五折最佳候选
6. 真实特征与负对照结果接近，需要做机制归因
7. 固定 seed 重跑结果无法复现
8. 用户明确要求全面审计

除这些触发条件外，不得把完整审计当作每个简单实验的前置步骤。

---

## 6. 检查分层

### 6.1 会话级检查：每次会话只运行一次

创建或读取 `run_context.json`，检查：

```text
fold_hash
eval_row_hash
baseline_id
LightGBM config hash
raw data hash
评分代码 hash
```

这些值在同一会话和同一基线下不变时，后续实验只比较 `run_context_hash`，不得重复扫描和计算。

### 6.2 实验级检查：每个实验运行一次

只检查：

```text
experiment_id
feature_delta
feature source
是否需要 outer-train fit
输出目录是否独立
```

### 6.3 晋级级检查：只有完整五折候选运行

只有方案通过 folds 0–1 后，才补齐：

- 五折指标
- P90、macro、最差井、胜井率
- 井级 bootstrap
- 负对照
- 关键覆盖切片
- 完整 leakage tests
- 完整 artifact 清单
- registry 登记和正式 conclusion

---

## 7. 负对照与 oracle 的运行时机

负对照和 oracle 不再默认阻塞 fold 0。

### fold 0 前必须运行的情况

- 负对照成本极低
- 路线图明确把它定义为基本合法性检查
- 没有正对照就无法确认特征公式写对

### fold 0 后再运行的情况

- 真实特征没有达到 `0.15 ft`：不再运行昂贵负对照和 oracle
- 真实特征达到门槛：运行对应负对照
- 真实与负对照接近：再运行更深入机制审计
- 只有需要判断信息上限时才运行 oracle

不得为一个 fold 0 已明显失败的特征继续消耗半小时做完整归因。

---

## 8. Artifact 分级

### 8.1 Smoke test

只需：

```text
smoke.log
feature sample
shape / finite-rate 检查
```

### 8.2 fold 0 或 fold 1 停止的实验

只需：

```text
config.json
feature_list.json
metrics.csv
runtime.json
conclusion.md
```

`conclusion.md` 可以很短，只写：

```text
实际改变
实际结果
命中哪条停止规则
只能否定什么
下一项实验
```

### 8.3 完整五折或晋级候选

此时才生成完整套件：

```text
feature_definition.json
feature_lineage.json
feature_quality.csv
cache_manifest.json
leakage_tests.json
negative_control_metrics.json
predictions.parquet
per_well.csv
per_fold.csv
slice_metrics.csv
metrics.json
runtime.json
bootstrap_replicates.parquet
conclusion.md
```

不得要求每个 fold 0 失败实验都先生成完整套件。

---

## 9. 测试策略

修改代码后只运行与当前改动相关的测试。

默认顺序：

```text
语法检查
→ 当前特征单元测试
→ 1～3 井 smoke test
→ fold 0
```

禁止默认运行：

- 全仓库 pytest
- 所有旧实验回归测试
- 所有缓存一致性扫描
- 与当前特征无关的模型测试

只有修改公共数据管线、fold、评分或 cache 基础设施时，才运行完整回归测试。

---

## 10. 运行与汇报

简单实验开始前只需一句：

```text
开始 <experiment_id>：仅新增 <feature_group>，先跑 3 井 smoke，随后直接跑 fold 0。
```

不要先发送长篇方案复述。

进度只在以下节点报告：

- smoke 完成
- fold 完成
- 命中停止门槛
- 开始完整五折
- 出现合同异常

不需要逐文件、逐函数、逐检查项汇报。

---

## 11. 时间预算

默认时间预算：

| 阶段 | 目标上限 |
|---|---:|
| 会话级合同检查 | 3 分钟 |
| 简单特征实现前检查 | 2 分钟 |
| 1～3 井 smoke test | 5 分钟 |
| fold 0 启动 | 接手任务后 10 分钟内 |
| 失败实验总结 | 5 分钟 |
| 完整审计 | 仅触发时运行 |

如果预检超过预算：

1. 停止扩展检查范围
2. 列出尚未解决的明确阻塞
3. 没有合同级阻塞则直接开始实验
4. 不得以“为了稳妥”继续无限检查

---

## 12. 结论写法

实验结论保持四段即可：

```text
事实：
推断：
当前只能否定：
下一步：
```

只有完整五折晋级方案才写长报告。

不要为简单 fold 0 失败生成数千字总结。

---

## 13. 当前状态与历史信息

AGENTS.md 不保存频繁变化的实验成绩和当前阶段。

当前实验编号、冻结最佳结果、下一项任务写入：

```text
rogii_clean/experiments/current_state.json
```

示例：

```json
{
  "baseline_id": "B00_simple_lgbm_v1",
  "best_candidate": "RF02a_gr_missing_geometry_v1",
  "best_micro_rmse": 15.4228,
  "next_experiment": "F03a_typewell_prefix_reliability_v1",
  "roadmap_version": "feature_roadmap_v2"
}
```

历史结论放在各实验 artifact 和 registry 中，不在每次运行时重新读取全部历史报告。

---

## 14. Codex 默认执行口令

收到“继续路线图”“做下一个实验”或明确实验编号后，Codex 默认执行：

```text
读取 current_state.json
→ 引用路线图实验卡
→ 运行快速预检
→ 立即实现和 smoke
→ fold 0
→ 按门槛自动推进
→ 保存分级 artifact
→ 更新 current_state.json 和 registry
```

除非出现合同异常，不请求重复确认，不做长篇事前复述，不把全面审计当成简单实验的前置门槛。
