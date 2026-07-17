# RF03-D0 Task 1 Brief

## 任务位置

这是 `docs/superpowers/plans/2026-07-17-rf03-d0-prefix-alignment.md` 的 Task 1。只完成冻结配置、实验卡和运行合同；不得提前实现 Task 2 的完整产物 builder，不得运行三井或 773 井正式实验。

## 全局约束

- 用户已确认 `B00_simple_lgbm_v1` 就是 `B0`；本实验不建立其他基线。
- 只读取水平井 `MD/GR/TVT_input` 和 Typewell `TVT/GR`。
- offset 固定为 `(-20,-10,0,10,20)` ft。
- scope 固定为 `all_visible` 与 `tail_1000ft`。
- 共同有效点最低为 50。
- `spatial_pad_1000_v1` 与 SHA-256 `0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab` 固定。
- 不允许 CLI 改 offset、scope、最低点数或门槛。
- 不训练模型，不修改 B0、LightGBM runner、参数、目标、fold 或历史实验。
- 不使用 Git 或 worktree。

## 修改文件

- 创建 `H:\kaggle\716\rogii_clean\configs\rf03_d0_prefix_alignment_v1.json`
- 修改 `H:\kaggle\716\rogii_clean\experiments\RF03_D0_prefix_alignment_card.md`
- 修改 `H:\kaggle\716\rogii_clean\scripts\diagnose_rf03_prefix_alignment.py`
- 修改 `H:\kaggle\716\rogii_clean\tests\test_rf03_prefix_alignment.py`

## 固定接口

实现并公开：

```python
def load_and_validate_diagnostic_config(path: Path) -> dict: ...

def run_alignment_for_registry(
    registry: pd.DataFrame,
    train_dir: Path,
    minimum_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...
```

保留现有：

```python
def build_summary(offset_scores: pd.DataFrame, margins: pd.DataFrame) -> dict[str, object]: ...
```

## 冻结配置内容

```json
{
  "experiment_id": "RF03_D0_prefix_alignment_v1",
  "experiment_type": "diagnostic_only",
  "baseline_id": "B00_simple_lgbm_v1",
  "fold_registry": "artifacts/folds/spatial_pad_1000_v1.csv",
  "fold_registry_sha256": "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab",
  "expected_wells": 773,
  "offsets_ft": [-20.0, -10.0, 0.0, 10.0, 20.0],
  "scopes": {"all_visible": null, "tail_1000ft": 1000.0},
  "minimum_points": 50,
  "bootstrap_resamples": 2000,
  "bootstrap_seed": 42,
  "hidden_tvt_loaded": false
}
```

配置还应固定 `train_dir`、smoke/full 输出目录；路径相对 `rogii_clean` 或项目根解析，不能依赖当前工作目录。

## TDD 必测合同

先写失败测试，再实现：

1. 正确配置加载成功，字段和类型完全一致。
2. 修改 experiment_id、experiment_type、baseline_id、fold registry/hash、expected_wells、offset 顺序/值、scope 名/值、minimum_points、bootstrap 次数/seed 或 hidden_tvt_loaded 时拒绝。
3. CLI 只接受 `--config` 与 `--mode smoke|full`，没有路径、offset、scope 或 minimum-points 覆盖参数。
4. `run_alignment_for_registry` 用临时三井数据只读取水平井 `MD/GR/TVT_input` 和 Typewell `TVT/GR`，返回两个 scope × 五个 offset 的分数及 margins。
5. smoke 井集合固定为 registry 按 `well_id` 排序后的前三井；full 必须等于 expected_wells。
6. 水平井 CSV 即使没有隐藏真值 `TVT` 列也能运行，证明代码不读取它。
7. 输入文件缺失、井数不符、fold hash 不符时在创建正式输出目录前报错。

## CLI 与执行边界

```text
--config configs/rf03_d0_prefix_alignment_v1.json
--mode smoke|full
```

本 Task 只完成接口与聚焦单测，不调用 `main()`，不生成 `artifacts/RF03_D0_prefix_alignment_v1`，不运行正式数据。

## 实验卡补充

明确记录：

```text
baseline_id = B00_simple_lgbm_v1 = B0
不训练模型
不读取隐藏 TVT
通过只允许进入循环平移负对照/F03a，不代表 CV 已改善
```

## 验证命令

```powershell
Set-Location H:\kaggle\716\rogii_clean
& 'D:\anaconda\python.exe' -m pytest tests\test_rf03_prefix_alignment.py -q
& 'D:\anaconda\python.exe' -m py_compile scripts\diagnose_rf03_prefix_alignment.py
```

完成后写报告到：

`H:\kaggle\716\.superpowers\sdd\rf03-d0-task-1-report.md`

报告必须包含 RED 证据、实现摘要、GREEN 输出、未运行正式实验的证据、自查和顾虑。最终只回复 `DONE`、`DONE_WITH_CONCERNS`、`NEEDS_CONTEXT` 或 `BLOCKED` 加一行摘要。
