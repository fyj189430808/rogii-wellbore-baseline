# P2-S02 实现计划

1. 测试 dense source 采样、不同井近邻、outer-train P99 梯度上限、梯度裁剪和相对路径积分。
2. 实现合法核心模块，确认改变目标 TVT/surface 不改变路径。
3. 实现逐折 dense source 索引、井内梯度反转负对照和可续跑 runner。
4. 运行三井 smoke；实现正确后自动跑 773 井。
5. 按预注册门槛冻结结论；通过才构造正式 LightGBM 特征，失败则保留为明确负结果。

