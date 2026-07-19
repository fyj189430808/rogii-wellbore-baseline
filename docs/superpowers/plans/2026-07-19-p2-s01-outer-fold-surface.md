# P2-S01 实现计划

1. 为 source 井级代表点、控制点采样、局部平面、前缀校准和折外排除写单元测试。
2. 实现 `src/p2_s01_outer_fold_surface.py`，保证合法路径接口不接收目标 TVT/surface。
3. 实现可续跑的 `scripts/diagnose_p2_s01_outer_fold_surface.py`，先跑三井 smoke。
4. smoke 通过后跑 773 井，保存逐井合法路径、负对照、oracle 正对照和指标。
5. 对照预注册门槛写结论；通过才进入正式 LightGBM 特征阶段，失败则只关闭当前简单局部平面表示。

