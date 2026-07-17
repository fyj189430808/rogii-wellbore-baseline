# Task 1 实施报告

## 修改摘要

- 将根目录 `AGENTS.md` 从约 758 行的“PF 理解优先”规则，整体重写为“特征实验优先”规则。
- 用户在实施中将权威路线图更正为 `H:\kaggle\716\rogii_clean\experiments\feature_roadmap_v2.md`；已完整读取 v2，并把 `AGENTS.md` 中所有路线图引用统一为 `rogii_clean/experiments/feature_roadmap_v2.md`。
- 按要求保留 12 个固定顶层章节及其顺序。
- 固定单个 LightGBM、冻结参数、1,734 棵树、seed 29、`spatial_pad_1000_v1`、固定 target、训练范围、自然隐藏评价行和评分代码。
- 按 v2 写入 B0、S 系列边际实验、Typewell 指纹压力测试和 C01/C02/C03 条件组合顺序；每一步仍只新增一个特征组。
- 写入统一晋级门槛、井级聚类 bootstrap 正确口径、等维负对照、覆盖限制、立即停止条件和隐藏真值不变性测试。
- 写入 B0 不可覆盖 hash、PF/Beam outer-fold cache、完整配置指纹和 v2 artifact 合同。
- 删除 PF 10.7 复现及八份导读文档作为新实验门槛的旧规定；明确旧材料只作为档案，不阻塞路线图实验。
- 保留中文普通表达、核心代码中文注释、简单代码结构、长任务进度、实验前后留档和防泄漏要求。
- 明确原 RF01 八版本继续“不晋级”、不得事后改判；`RF01_stability_audit_v1` 固定 RF01a，只补 folds 2–4，不能追认晋级。
- 明确不主动执行 Git、worktree、建分支、提交、推送或 PR。
- 未运行任何实验，未修改其他项目代码或配置，未使用任何 Git/worktree/提交操作。

## 验证命令与结果

### 1. brief 指定的冲突扫描

```powershell
rg -n "TBD|TODO|待定|PF 10\.7 复现.*禁止|不得主动开启新的大实验|当前阶段不是继续.*实验" AGENTS.md
```

结果：无匹配；包装检查输出 `PASS: brief 冲突扫描无匹配`，退出码 0。

### 2. brief 指定的关键约束扫描

```powershell
rg -n "feature_roadmap|单模 LightGBM|micro RMSE|RF01_stability_audit_v1|不得事后改判|中文" AGENTS.md
```

结果：六类关键约束均有匹配，退出码 0。关键位置包括唯一目标、统一门槛、中文规则和 RF01 两层结论。

### 3. 旧 PF 门槛扫描

```powershell
rg -n "当前阶段只有一个最高优先级|让我真正看懂|当前 PF 10\.7 基线必须先完成代码考古|PF 10\.7 复现是进入新研究的门槛|当前研究优先级|当前 PF 理解阶段|README 和 8 份 PF 导读文档" AGENTS.md
```

结果：无匹配；包装检查输出 `PASS: 旧 PF 门槛和旧完成标准无匹配`，退出码 0。

### 4. 路线图路径扫描

```powershell
rg -n "feature_roadmap2\.md|experiments/feature_roadmap\.md" AGENTS.md
```

结果：无匹配；包装检查输出 `PASS: 无旧路线图路径或错误 roadmap2 路径`，退出码 0。权威 `feature_roadmap_v2.md` 在唯一目标、执行依据和完成标准中共出现 3 次。

### 5. 章节和 v2 核心规则逐项检查

使用 PowerShell 读取 UTF-8 文件，逐项比较 12 个顶层章节顺序，并用精确子串检查模型合同、S/C 顺序、全部门槛、Typewell 压力测试、B0 hash、泄漏测试、artifact、RF01 和 Git 约束。

结果：

```text
PASS: 顶层章节 12/12 顺序正确
PASS: v2 核心规则 35/35 存在
```

### 6. Markdown 与文件指纹检查

结果：

```text
PASS: Markdown 围栏成对，共 26 条围栏线
PASS: 权威 v2 路径引用 3 处
AGENTS.md SHA256: CADDCD3DFAF8B1D4ECA02A4E6046C6E77B3517AE26D60EE78E85738E40BAE454
```

最终报告写入后再次计算得到相同 SHA256；最终文件共 476 行。

## 自查发现

1. 初版按旧 brief 引用了 `feature_roadmap.md`；用户随后明确更正 v2。已完整读取 `feature_roadmap_v2.md`，没有只做文件名替换，而是据此重写实验顺序、门槛、血缘、缓存、产物和完成标准。
2. 中间文件曾出现无效路径 `feature_roadmap2.md`。最终扫描确认该路径和旧 `feature_roadmap.md` 均为 0 处，全部改为 `feature_roadmap_v2.md`。
3. 首次 35 项精确检查发现规则只用中文概括泄漏测试，没有保留 v2 指定的 `hidden TVT deletion invariance` 等登记名。已补齐八个固定字段名；同一检查重跑后为 35/35 通过。
4. v2 将旧路线细化为 F02a/F02b、F03a/F03b、F05a/F05b 和 F07R，并加入 S/C 两阶段。最终文档已按该结构执行，不再保留旧 F01～F07 简表。
5. v2 的固定主 CV 与 F03/F04 的 Typewell strict 压力测试容易被误解为换 fold。文档已明确 strict 只做额外泛化审计，不替代 `spatial_pad_1000_v1` 主 CV，也不能用于事后换主 fold。
6. v2 的 `code_commit` 与本任务“不主动使用 Git”可能冲突。文档明确：无 Git 提交哈希时使用相关代码文件内容 hash，不能省略代码指纹。
7. `AGENTS.md` 由 758 行精简为约 476 行，同时保留执行所需的 CV、门槛、防泄漏、缓存、留档、进度和 RF01 规则。

## 顾虑

当前无阻塞性顾虑。需要注意的是，权威路线图已经由用户更新为 v2；后续任何自动化或实验脚本都必须读取 `feature_roadmap_v2.md`，不能继续使用旧路径或旧 F01～F07 分组。

## 复核修正

复核依据：`H:\kaggle\716\.superpowers\sdd\task-1-review.md`。本轮逐条处理其中 5 项重要问题和 5 项轻微问题，只修改根目录 `AGENTS.md`，并在本报告末尾追加本节；未使用 Git/worktree/提交操作，未运行实验。

### 五项重要问题

1. **隐藏真实 TVT 与 oracle 用途冲突**
   - 修正：统一规定隐藏真实 TVT 只能用于最终评分，或用于明确标注且与正式特征、缓存、候选选择完全隔离的 oracle；禁止 oracle 反哺公式、窗口、阈值或晋级决定。
   - 覆盖验证：旧绝对句 `只在最终评分时读取真实 TVT` 为 0 处；评分/隔离 oracle/禁止反哺三条语义断言均通过。

2. **停止当前实现与暂停整条路线冲突**
   - 修正：正常负结果只停止当前实现，留档后继续下一项；泄漏、固定 fold/row hash 全局合同异常或 PF/Beam 折不匹配必须暂停整条路线，修复并重验前不得继续。
   - 覆盖验证：同时断言“正常失败继续下一项”“合同异常暂停整条路线”“合同异常不能记成实验负结果”均存在；旧的普通路线图停止即暂停句为 0 处。

3. **C 系列候选资格和数量不清**
   - 修正：C 候选池只允许完整五折满足第 3.3 节全部统一门槛的 S 组，失败组不得补位；0 个不跑 C、1 个只跑 C01、2 个跑到 C02、至少 3 个才跑到 C03。
   - 覆盖验证：候选资格、不得补位和 0/1/2/3 数量逻辑 6 条语义断言全部通过。

4. **PF/Beam 单井 cache 缺少 `well_id`**
   - 修正：规定每个单井 cache manifest 必须按顺序保存 `well_id`、`outer_fold`、`fit_wells_hash`、`pf_config_hash`、`seed_list_hash`、`prediction_row_hash`；通用井列表不能替代单井 `well_id`。
   - 覆盖验证：六字段连续顺序断言及 `well_id` 不可替代断言通过。

5. **`feature_quality.csv` 缺少首列 `feature`**
   - 修正：明确质量表每行对应一个特征，第一列必须为 `feature`，并完整列出 v2 的 13 个固定字段及顺序。
   - 覆盖验证：13 字段连续顺序断言和首列 `feature` 断言通过。

### 五项轻微问题

1. **C 系列缺少不同信息源原则**
   - 修正：排序首先依据完整五折改善；同分或预先定义的近似同分范围内，必须提前登记信息源分类和同分规则，优先保留不同信息源，避免高度同源堆叠。
   - 覆盖验证：提前登记同分规则和避免高度同源两条断言通过。

2. **fold 0 错引完整统一门槛**
   - 修正：明确 fold 0 只判断 micro RMSE 改善是否达到 0.15 ft 子门槛；完整五折、4/5 折、bootstrap 和 P90 此时不能判断。
   - 覆盖验证：新 0.15 ft 子门槛断言通过；旧句 `fold 0 未达到统一门槛` 为 0 处。

3. **“全部适用 S 系列”范围过宽**
   - 修正：F05b 只有在 F05a 完整验证后仍无稳定信号时才可降级或取消，这是唯一预定的 S 系列取消例外；其他实验不能以“暂不适用”跳过。
   - 覆盖验证：唯一例外、不得类推、不得用“暂不适用”跳过三条断言通过；旧词 `全部适用 S 系列` 为 0 处。

4. **首次英文术语没有中文解释**
   - 修正：为 micro RMSE、bootstrap、oracle、private-safe、branch-aware self-template、horizon-matched prefix holdout、stacking、OOF、outer fold 和 Typewell fingerprint stress test 补充普通中文解释。
   - 覆盖验证：10/10 个固定中文解释短语存在。

5. **原报告只做关键词存在检查，缺少语义覆盖**
   - 修正：新增逐问题语义断言，同时检查必需句存在、冲突旧句不存在、字段连续顺序、12 章顺序、Markdown 围栏和旧路径/旧门槛。
   - 覆盖验证：综合检查输出如下，退出码 0：

```text
PASS I1: 隐藏 TVT/oracle 用途统一
PASS I2: 正常失败与整条路线暂停已区分
PASS I3: C 候选资格及 0/1/2/3 数量逻辑完整
PASS I4: 单井 PF/Beam cache 含 well_id 六字段
PASS I5: feature_quality 首列 feature，13 字段齐全
PASS M1-M4: 异源组合、fold0 子门槛、F05b 例外、中文术语解释齐全
PASS STRUCTURE: 顶层章节 12/12，Markdown 围栏 30 条
PASS CONFLICT: 旧门槛、旧路径和待定项 0 处
```

### 复核修正后的文件状态

```text
AGENTS.md 行数：521
AGENTS.md SHA256：03AB662AD8FBF25A8E48F45796138840565A38B3A11145B7A28B7018A481A1C9
```

当前没有阻塞性顾虑。后续执行必须继续以 `feature_roadmap_v2.md` 为权威路线图，并按修正后的语义规则判断 oracle、停止控制流、C 候选池、缓存血缘和产物字段。
