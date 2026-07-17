# RF03-D0：可见前缀 Typewell 对齐正对照设计

## 1. 已确认的基线定义

用户已明确：`B00_simple_lgbm_v1` 就是 `feature_roadmap_v2` 中的 `B0`。

因此本阶段固定采用：

- 基线编号：`B00_simple_lgbm_v1`；
- 12 个基础几何与原始 GR 特征；
- 单个 LightGBM，1,734 棵树，seed 29；
- `spatial_pad_1000_v1` 固定五折；
- 773 口井、3,783,989 条自然隐藏评价行；
- 不新增 PF/Beam 或 Typewell 全局摘要来重新定义基线。

RF03-D0 是无模型诊断，不会训练或修改 B0。

## 2. 唯一问题

在当前井可见前缀中，真实 `TVT_input` 位置对应的 Typewell GR，是否稳定优于固定错位的 Typewell GR？

如果连已知 TVT 位置都不能从 GR 中辨认出来，就没有充分依据继续构造复杂的 F03a Typewell 可信度特征。

## 3. 合法输入

每口井只读取：

- 水平井：`MD`、`GR`、`TVT_input`；
- 配对 Typewell：`TVT`、`GR`；
- 固定 fold 注册表中的 `well_id` 与 `fold`。

自然隐藏段真实 `TVT` 不读取、不进入内存，也不用于公式、筛选或停止判断。

## 4. 固定比较

候选 offset 固定为：

```text
-20 ft
-10 ft
  0 ft
+10 ft
+20 ft
```

两个观察范围固定为：

```text
all_visible：全部可见前缀
tail_1000ft：距可见末端最近 1000 ft
```

所有 offset 必须使用共同有效行，避免支持量不同制造假优势。每个井和范围至少需要 50 个共同有效点。

## 5. 固定指标

每个井、范围和 offset 计算：

- 原始归一化相关系数 `raw_ncc`，越大越好；
- 先拟合 `horizontal_GR ≈ a × typewell_GR + b` 后的中位绝对误差 `affine_median_ae`，越小越好；
- 共同有效点数量 `n_points`。

再计算 0 ft 相对四个错误 offset 中最佳者的 margin，以及 0 ft 是否为最佳。

## 6. 通过与停止规则

保持原实验卡规则：

- raw NCC 或 affine-MAE 至少一个指标，在两个范围的总体与五个 fold 中，0 ft 最佳率都严格超过 50%，则 RF03-D0 通过；
- 若两个指标都未达到上述条件，则停止当前未平滑对齐实现；
- 失败只否定“全前缀/尾部 1000 ft、未平滑 raw NCC 与逐 offset affine-MAE”这一实现，不否定 Typewell 信息源；
- 结果出来后不改变 offset、范围、最少点数、指标或门槛。

若第一层正对照通过，再按路线图运行预先登记的 Typewell GR 循环平移负对照；负对照不得用于事后修改第一层公式。

## 7. 运行顺序

```text
现有单元测试
→ 3 口井小样本检查
→ 773 井完整 RF03-D0
→ 独立复算 summary
→ 根据固定门槛决定是否进入 F03a
```

本诊断不训练 LightGBM，因此没有 fold 0 / fold 1 模型筛查。

## 8. 产物和复现

正式目录固定为：

```text
rogii_clean/artifacts/RF03_D0_prefix_alignment_v1/
```

必须保存原实验卡规定的：

```text
offset_scores.csv
per_well_margins.csv
summary.json
config.json
conclusion.md
```

并补齐与诊断适配的项目统一记录：输入/代码 hash、数据血缘、泄漏检查、逐 fold 汇总、支持量切片和运行时间。诊断文件必须明确标注“不是 TVT 模型预测”，不能伪造模型指标或负对照结果。

## 9. 代码边界

- 复用 `src/rf03_prefix_alignment.py` 的现有纯函数；
- 只对诊断 runner 增加合同校验、可恢复落盘和完整产物；
- 不修改 B0、LightGBM runner、模型参数、fold、目标或历史实验；
- 不执行 Git 操作。

## 10. 设计自检

- 无 `TBD` 或结果后再决定的公式；
- offset、范围、指标、门槛和最少支持量均在运行前固定；
- 诊断不依赖隐藏 TVT 或 B0 模型输出；
- RF03-D0 通过只允许进入 F03a，不能直接宣称 Typewell 特征会改善 CV。
