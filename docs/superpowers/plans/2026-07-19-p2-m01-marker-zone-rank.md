# P2-M01 marker 区间排名审计实现计划

1. 先写程序测试：模板尾段指纹、outer-train donor 排除、marker 中位数与歧义、q/TVT 往返、区间门控排名和确定性随机门控。
2. 新建小型核心模块，只处理模板、marker、q 坐标和排名；不读取隐藏 TVT，不修改 F01b scorer。
3. 新建诊断 runner。第一阶段为全部 155 口 fold 0 井生成 `legal_markers.csv`，其中不得出现验证 Geology、隐藏 TVT 或 oracle 排名。
4. legal marker 表完整落盘后，才读取验证井自身 Geology做 withheld marker 核对，并读取隐藏 TVT/F01b score 做 block 排名。
5. 先跑 3 井 smoke 检查行键、marker 和 q 公式，再跑固定 fold 0。
6. 按井做 2,000 次聚类 bootstrap，严格应用实验卡中的全部门槛。
7. 只有全部门槛通过才另立 q-state 连续路径实验；失败则按具体证据转向更细 marker、未知模板边界检测或其他 GR 代价，不直接加入 LightGBM。

