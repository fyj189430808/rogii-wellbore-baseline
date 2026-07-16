# B00 Simple LightGBM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, private-safe, single-model LightGBM baseline on the frozen spatial-pad five-fold CV using exactly the 12 approved horizontal-well features.

**Architecture:** `src/lgbm_features.py` converts one well into explicit hidden-row features without reading hidden labels as inputs. `src/lgbm_data.py` validates the frozen registry and builds a fingerprinted local Parquet cache. `scripts/reproduce_simple_lgbm.py` trains one fixed LightGBM instance per outer fold, supports fold-level resume, merges OOF predictions, and reuses `src/metrics.py` for reporting.

**Tech Stack:** Python 3.11, pandas 2.2.2, NumPy 1.26.4, LightGBM 4.6.0, PyArrow 16.1.0, pytest 7.4.4.

## Global Constraints

- Fixed CV: `spatial_pad_1000_v1`, 773 wells, 290 pads, 3,783,989 hidden rows.
- Fixed registry SHA-256: `0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab`.
- Fixed target: `TVT - last_visible_TVT_input`; final prediction adds `last_visible_TVT_input` back.
- Fixed model parameters: read verbatim from `configs/lgbm_feature_baseline_v1.json`, including 1,734 trees and seed 29; no early stopping or tuning.
- Exactly 12 features from the approved design; never auto-select columns.
- No Typewell, PF, Beam, neighbour labels, surfaces, old model predictions, stacking, smoothing, shrinkage, projection, or test/train same-well override.
- Validation hidden `TVT` is used only to construct the validation target and after predictions exist to score them.
- New core code uses explicit names and explanatory Chinese comments.
- Large caches, models, and predictions stay under ignored `rogii_clean/artifacts/`.

---

### Task 1: Implement the exact 12-feature builder with leakage tests

**Files:**
- Create: `rogii_clean/tests/test_lgbm_features.py`
- Create: `rogii_clean/src/lgbm_features.py`

**Interfaces:**
- Consumes: one horizontal-well `pandas.DataFrame`, `well_id: str`, and `fold: int`.
- Produces: `build_simple_lgbm_rows(horizontal_df, well_id, fold) -> pandas.DataFrame` and immutable `FEATURE_COLUMNS`.

- [ ] **Step 1: Write the failing feature-formula tests**

```python
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.lgbm_features import FEATURE_COLUMNS, build_simple_lgbm_rows


def make_small_well() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [100.0, 101.0, 102.0, 103.0, 104.0],
            "X": [10.0, 11.0, 12.0, 14.0, 17.0],
            "Y": [20.0, 20.0, 20.0, 23.0, 27.0],
            "Z": [1000.0, 1001.0, 1002.0, 1004.0, 1007.0],
            "GR": [50.0, 51.0, np.nan, 70.0, 80.0],
            "TVT_input": [200.0, 201.0, np.nan, np.nan, np.nan],
            "TVT": [200.0, 201.0, 202.0, 204.0, 207.0],
            "ANCC": [9999.0] * 5,
        }
    )


def test_build_simple_lgbm_rows_uses_exact_formulas() -> None:
    result = build_simple_lgbm_rows(make_small_well(), "well_a", 2)
    assert list(result[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert result["row_index"].tolist() == [2, 3, 4]
    np.testing.assert_allclose(result["last_visible_tvt"], [201.0] * 3)
    np.testing.assert_allclose(result["md_since_visible_end"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result["hidden_fraction"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(result["dx_from_visible_end"], [1.0, 3.0, 6.0])
    np.testing.assert_allclose(result["dy_from_visible_end"], [0.0, 3.0, 7.0])
    np.testing.assert_allclose(result["dz_from_visible_end"], [1.0, 3.0, 6.0])
    np.testing.assert_allclose(
        result["dxy_from_visible_end"], [1.0, np.sqrt(18.0), np.sqrt(85.0)]
    )
    assert result["gr_missing"].tolist() == [1.0, 0.0, 0.0]
    assert np.isnan(result.loc[0, "gr_raw"])
    np.testing.assert_allclose(result["target_delta"], [1.0, 3.0, 6.0])


def test_hidden_tvt_changes_only_targets_not_features() -> None:
    original = make_small_well()
    changed = make_small_well()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0
    original_rows = build_simple_lgbm_rows(original, "well_a", 2)
    changed_rows = build_simple_lgbm_rows(changed, "well_a", 2)
    assert_frame_equal(
        original_rows[FEATURE_COLUMNS], changed_rows[FEATURE_COLUMNS], check_exact=True
    )
    assert not np.array_equal(original_rows["target_delta"], changed_rows["target_delta"])


def test_extra_surface_column_never_enters_features() -> None:
    result = build_simple_lgbm_rows(make_small_well(), "well_a", 2)
    assert "ANCC" not in FEATURE_COLUMNS
    assert "ANCC" not in result.columns
    assert len(FEATURE_COLUMNS) == 12
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_lgbm_features.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.lgbm_features'`.

- [ ] **Step 3: Implement the minimal feature builder**

```python
"""为 B00 构造一口井的 12 个固定合法特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "last_visible_tvt",
    "md_since_visible_end",
    "hidden_fraction",
    "x_current",
    "y_current",
    "z_current",
    "dx_from_visible_end",
    "dy_from_visible_end",
    "dz_from_visible_end",
    "dxy_from_visible_end",
    "gr_raw",
    "gr_missing",
]

REQUIRED_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"]


def build_simple_lgbm_rows(
    horizontal_df: pd.DataFrame,
    well_id: str,
    fold: int,
) -> pd.DataFrame:
    """输入一口完整训练井，输出该井隐藏行的特征、锚点和训练 target。"""
    missing_columns = set(REQUIRED_COLUMNS) - set(horizontal_df.columns)
    if missing_columns:
        raise ValueError(f"水平井缺少列：{sorted(missing_columns)}")

    visible_mask = horizontal_df["TVT_input"].notna().to_numpy()
    hidden_mask = horizontal_df["TVT_input"].isna().to_numpy()
    if not visible_mask.any() or not hidden_mask.any():
        raise ValueError(f"井 {well_id} 必须同时有可见前缀和隐藏后缀")

    visible_positions = np.flatnonzero(visible_mask)
    hidden_positions = np.flatnonzero(hidden_mask)
    last_visible_position = int(visible_positions[-1])
    expected_visible = np.arange(last_visible_position + 1)
    expected_hidden = np.arange(last_visible_position + 1, len(horizontal_df))
    if not np.array_equal(visible_positions, expected_visible):
        raise ValueError(f"井 {well_id} 的 TVT_input 可见区不是连续前缀")
    if not np.array_equal(hidden_positions, expected_hidden):
        raise ValueError(f"井 {well_id} 的 TVT_input 隐藏区不是连续后缀")

    md_all = horizontal_df["MD"].to_numpy(dtype=np.float64)
    if np.any(np.diff(md_all) < 0.0):
        raise ValueError(f"井 {well_id} 的 MD 不是单调非降")

    hidden_df = horizontal_df.iloc[hidden_positions]
    last_visible_row = horizontal_df.iloc[last_visible_position]
    hidden_md = hidden_df["MD"].to_numpy(dtype=np.float64)
    hidden_span = max(float(hidden_md[-1] - hidden_md[0]), 1.0)

    dx = hidden_df["X"].to_numpy(dtype=np.float64) - float(last_visible_row["X"])
    dy = hidden_df["Y"].to_numpy(dtype=np.float64) - float(last_visible_row["Y"])
    dz = hidden_df["Z"].to_numpy(dtype=np.float64) - float(last_visible_row["Z"])
    last_visible_tvt = float(last_visible_row["TVT_input"])
    target_tvt = hidden_df["TVT"].to_numpy(dtype=np.float64)
    gr_raw = hidden_df["GR"].to_numpy(dtype=np.float64)

    result = pd.DataFrame(
        {
            "well_id": str(well_id),
            "fold": int(fold),
            "row_index": hidden_positions.astype(np.int32),
            "md": hidden_md,
            "target_tvt": target_tvt,
            "carry_tvt": np.full(len(hidden_positions), last_visible_tvt),
            "target_delta": target_tvt - last_visible_tvt,
            "last_visible_tvt": last_visible_tvt,
            "md_since_visible_end": hidden_md - float(last_visible_row["MD"]),
            "hidden_fraction": (hidden_md - hidden_md[0]) / hidden_span,
            "x_current": hidden_df["X"].to_numpy(dtype=np.float64),
            "y_current": hidden_df["Y"].to_numpy(dtype=np.float64),
            "z_current": hidden_df["Z"].to_numpy(dtype=np.float64),
            "dx_from_visible_end": dx,
            "dy_from_visible_end": dy,
            "dz_from_visible_end": dz,
            "dxy_from_visible_end": np.sqrt(dx * dx + dy * dy),
            "gr_raw": gr_raw,
            "gr_missing": np.isnan(gr_raw).astype(np.float64),
        }
    )
    return result
```

- [ ] **Step 4: Run the focused and full test suites**

Run: `python -m pytest tests/test_lgbm_features.py -q`

Expected: `3 passed`.

Run: `python -m pytest tests -q`

Expected: all previous tests plus these three pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add rogii_clean/src/lgbm_features.py rogii_clean/tests/test_lgbm_features.py
git commit -m "add exact simple lgbm features"
```

---

### Task 2: Build and validate the full feature cache

**Files:**
- Create: `rogii_clean/tests/test_lgbm_data.py`
- Create: `rogii_clean/src/lgbm_data.py`

**Interfaces:**
- Consumes: the frozen fold registry, raw horizontal-well directory, and `build_simple_lgbm_rows`.
- Produces: `load_and_validate_registry(path)`, `build_feature_table(registry_df, train_dir, progress_interval)`, and `file_sha256(path)`.

- [ ] **Step 1: Write failing registry and assembly tests**

```python
from pathlib import Path

import pandas as pd
import pytest

from src.lgbm_data import build_feature_table, load_and_validate_registry


def test_registry_rejects_a_pad_split_across_folds(tmp_path: Path) -> None:
    registry_path = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "pad_id": ["same", "same"],
            "fold": [0, 1],
            "hidden_rows": [1, 1],
        }
    ).to_csv(registry_path, index=False)
    with pytest.raises(ValueError, match="pad"):
        load_and_validate_registry(registry_path, expected_wells=None, expected_rows=None)


def test_build_feature_table_matches_registry_hidden_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = pd.DataFrame(
        {"well_id": ["a"], "pad_id": ["a"], "fold": [0], "hidden_rows": [2]}
    )
    well_path = tmp_path / "a__horizontal_well.csv"
    pd.DataFrame(
        {
            "MD": [1.0, 2.0, 3.0], "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.0, 0.0], "Z": [10.0, 11.0, 12.0],
            "GR": [50.0, 51.0, 52.0], "TVT": [100.0, 101.0, 102.0],
            "TVT_input": [100.0, None, None],
        }
    ).to_csv(well_path, index=False)
    result = build_feature_table(registry, tmp_path, progress_interval=0)
    assert len(result) == 2
    assert result["well_id"].tolist() == ["a", "a"]
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_lgbm_data.py -q`

Expected: collection fails because `src.lgbm_data` does not exist.

- [ ] **Step 3: Implement registry validation and deterministic assembly**

```python
"""读取冻结 fold，并把逐井 B00 特征组装成完整行级表。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.lgbm_features import build_simple_lgbm_rows


def file_sha256(path: Path) -> str:
    """返回文件内容的 SHA-256 十六进制字符串。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_registry(
    registry_path: Path,
    expected_wells: int | None = 773,
    expected_rows: int | None = 3_783_989,
) -> pd.DataFrame:
    """读取固定注册表，并检查井、行数和 pad 不跨折。"""
    registry = pd.read_csv(registry_path, dtype={"well_id": str, "pad_id": str})
    required = {"well_id", "pad_id", "fold", "hidden_rows"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"fold 注册表缺少列：{sorted(missing)}")
    if registry["well_id"].duplicated().any():
        raise ValueError("fold 注册表含重复 well_id")
    if registry.groupby("pad_id")["fold"].nunique().max() != 1:
        raise ValueError("同一个 pad 出现在多个 fold")
    if expected_wells is not None and len(registry) != expected_wells:
        raise ValueError("fold 注册表井数不匹配")
    if expected_rows is not None and int(registry["hidden_rows"].sum()) != expected_rows:
        raise ValueError("fold 注册表隐藏行数不匹配")
    return registry.sort_values("well_id").reset_index(drop=True)


def build_feature_table(
    registry_df: pd.DataFrame,
    train_dir: Path,
    progress_interval: int = 50,
) -> pd.DataFrame:
    """逐井构造特征；返回顺序固定为 well_id、row_index。"""
    well_tables: list[pd.DataFrame] = []
    for well_number, row in enumerate(registry_df.itertuples(index=False), start=1):
        well_path = train_dir / f"{row.well_id}__horizontal_well.csv"
        if not well_path.is_file():
            raise FileNotFoundError(f"缺少水平井文件：{well_path}")
        horizontal_df = pd.read_csv(well_path)
        well_rows = build_simple_lgbm_rows(horizontal_df, str(row.well_id), int(row.fold))
        if len(well_rows) != int(row.hidden_rows):
            raise ValueError(f"井 {row.well_id} 的隐藏行数与注册表不一致")
        well_tables.append(well_rows)
        if progress_interval > 0 and (
            well_number % progress_interval == 0 or well_number == len(registry_df)
        ):
            print(f"特征进度：{well_number}/{len(registry_df)} 井", flush=True)
    result = pd.concat(well_tables, ignore_index=True)
    return result.sort_values(["well_id", "row_index"]).reset_index(drop=True)
```

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_lgbm_data.py -q`

Expected: `2 passed`.

Run: `python -m pytest tests -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add rogii_clean/src/lgbm_data.py rogii_clean/tests/test_lgbm_data.py
git commit -m "add fixed-fold lgbm data assembly"
```

---

### Task 3: Add the resumable single-model fold runner

**Files:**
- Create: `rogii_clean/configs/b00_simple_lgbm_v1.json`
- Create: `rogii_clean/tests/test_simple_lgbm_runner.py`
- Create: `rogii_clean/scripts/reproduce_simple_lgbm.py`
- Modify: `requirements.txt`
- Modify: `rogii_clean/README.md`

**Interfaces:**
- Consumes: the Task 2 feature table, frozen JSON parameters, selected fold IDs, and `src.metrics`.
- Produces: one model and one prediction file per fold; when all folds exist, full OOF artifacts and metrics.

- [ ] **Step 1: Write failing tests for fold isolation and delta restoration**

```python
import numpy as np
import pandas as pd

from scripts.reproduce_simple_lgbm import fold_indices, restore_absolute_tvt


def test_fold_indices_never_mix_validation_rows() -> None:
    table = pd.DataFrame({"fold": [0, 0, 1, 1, 2]})
    train_indices, validation_indices = fold_indices(table, validation_fold=1)
    assert train_indices.tolist() == [0, 1, 4]
    assert validation_indices.tolist() == [2, 3]
    assert set(train_indices).isdisjoint(set(validation_indices))


def test_restore_absolute_tvt_adds_well_anchor() -> None:
    anchor = np.array([100.0, 100.0, 200.0])
    delta = np.array([1.5, -2.0, 3.0])
    np.testing.assert_allclose(
        restore_absolute_tvt(anchor, delta), [101.5, 98.0, 203.0]
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_simple_lgbm_runner.py -q`

Expected: import fails because `scripts.reproduce_simple_lgbm` does not exist.

- [ ] **Step 3: Create the frozen B00 config**

Create `configs/b00_simple_lgbm_v1.json` with:

```json
{
  "experiment_id": "B00_simple_lgbm_v1",
  "feature_version": "simple_horizontal_12_v1",
  "feature_columns": [
    "last_visible_tvt", "md_since_visible_end", "hidden_fraction",
    "x_current", "y_current", "z_current", "dx_from_visible_end",
    "dy_from_visible_end", "dz_from_visible_end", "dxy_from_visible_end",
    "gr_raw", "gr_missing"
  ],
  "model_config": "configs/lgbm_feature_baseline_v1.json",
  "fold_registry": "artifacts/folds/spatial_pad_1000_v1.csv",
  "fold_registry_sha256": "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab",
  "expected_wells": 773,
  "expected_rows": 3783989,
  "target": "target_delta",
  "final_prediction": "last_visible_tvt_plus_predicted_delta"
}
```

- [ ] **Step 4: Implement explicit runner helpers and CLI**

The runner must define these exact testable helpers:

```python
def fold_indices(feature_table: pd.DataFrame, validation_fold: int) -> tuple[np.ndarray, np.ndarray]:
    fold_values = feature_table["fold"].to_numpy(dtype=np.int8)
    validation_indices = np.flatnonzero(fold_values == int(validation_fold))
    train_indices = np.flatnonzero(fold_values != int(validation_fold))
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("训练行或验证行为空")
    return train_indices, validation_indices


def restore_absolute_tvt(anchor: np.ndarray, predicted_delta: np.ndarray) -> np.ndarray:
    anchor_values = np.asarray(anchor, dtype=np.float64)
    delta_values = np.asarray(predicted_delta, dtype=np.float64)
    if anchor_values.shape != delta_values.shape:
        raise ValueError("锚点与 delta 预测 shape 不一致")
    return anchor_values + delta_values
```

The CLI must provide:

```text
--train-dir PATH       required unless the repository default exists
--fold 0|1|2|3|4|all  required
--config PATH          defaults to configs/b00_simple_lgbm_v1.json
--rebuild-cache        optional explicit cache invalidation
```

The implementation must:

1. Print cache file, creation time, stored fingerprint, current fingerprint, and exact-match status.
2. Build the cache through `build_feature_table` only when absent, explicitly rebuilt, or mismatched.
3. Convert only `FEATURE_COLUMNS` to `float32`; preserve native NaN only in `gr_raw`.
4. For each selected fold, assert training and validation well IDs are disjoint.
5. Instantiate exactly `LGBMRegressor(**frozen_params)` and call `fit` once per fold.
6. Save `fold_N/model.txt`, `fold_N/predictions.parquet`, `fold_N/runtime.json`, and `fold_N/feature_importance.csv` before moving to the next fold.
7. Reuse a fold only if its stored experiment fingerprint exactly matches.
8. When all five folds exist, merge predictions, assert every hidden row appears exactly once, call the existing metric functions, and write the standard artifact set.

- [ ] **Step 5: Pin PyArrow because Parquet is now a required runtime dependency**

Add this exact line to root `requirements.txt`:

```text
pyarrow==16.1.0
```

- [ ] **Step 6: Document the exact smoke, fold-0, resume, and full-CV commands**

Add to `rogii_clean/README.md`:

```text
python scripts/reproduce_simple_lgbm.py --train-dir ../input/data/raw/train --fold 0
python scripts/reproduce_simple_lgbm.py --train-dir ../input/data/raw/train --fold all
```

Explain that the second command reuses a matching completed fold 0 and trains folds 1–4.

- [ ] **Step 7: Run focused and full tests**

Run: `python -m pytest tests/test_simple_lgbm_runner.py -q`

Expected: `2 passed`.

Run: `python -m pytest tests -q`

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add requirements.txt rogii_clean/configs/b00_simple_lgbm_v1.json rogii_clean/scripts/reproduce_simple_lgbm.py rogii_clean/tests/test_simple_lgbm_runner.py rogii_clean/README.md
git commit -m "add resumable simple lgbm cv runner"
```

---

### Task 4: Run B00 and record the measured CV

**Files:**
- Generate locally: `rogii_clean/artifacts/B00_simple_lgbm_v1/**`
- Modify: `rogii_clean/experiments/registry.jsonl`

**Interfaces:**
- Consumes: all Task 1–3 code, raw training CSVs, the frozen fold registry, and the B00 config.
- Produces: fold 0 checkpoint, complete five-fold OOF predictions, metrics, feature importance, runtime, and final conclusion.

- [ ] **Step 1: Verify the clean test baseline immediately before the experiment**

Run: `python -m pytest tests -q`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run the three-well feature smoke check**

Use the first three registry wells, call `build_simple_lgbm_rows`, and assert:

```text
wells=3
features=12
feature_rows=sum(registry.hidden_rows for those wells)
forbidden_feature_count=0
non_finite_non_gr_count=0
```

- [ ] **Step 3: Train and score fold 0**

Run from `rogii_clean`:

```text
python scripts/reproduce_simple_lgbm.py --train-dir H:\kaggle\716\input\data\raw\train --fold 0
```

Before running, report the 626 training wells, 147 validation wells, 3,026,939 training rows, 757,050 validation rows, 12 features, 1,734 trees, expected artifact paths, and resume behavior.

- [ ] **Step 4: Audit fold 0 before continuing**

Verify:

```text
validation rows = 757050
validation wells = 147
all predictions finite
train/validation well overlap = 0
train/validation pad overlap = 0
feature list equals the approved 12 columns
fold fingerprint equals current experiment fingerprint
```

- [ ] **Step 5: Train folds 1–4 and merge complete OOF**

Run from `rogii_clean`:

```text
python scripts/reproduce_simple_lgbm.py --train-dir H:\kaggle\716\input\data\raw\train --fold all
```

The runner must reuse fold 0 and print progress at least once per 100 trees or another interval shorter than 60 seconds.

- [ ] **Step 6: Independently verify final metrics and artifacts**

Run a separate verification command that checks:

```text
OOF rows = 3783989
OOF wells = 773
each row key is unique
all predictions are finite
five fold row counts match the registry
independently recomputed micro RMSE equals metrics.json within 1e-12
all standard artifact files exist
```

- [ ] **Step 7: Append the experiment registry and conclusion**

Append one JSON object containing exact measured metrics, commit hash, feature version, fold version, runtime, and artifact directory. Write `conclusion.md` using the required sections:

```text
数据直接证明的事实
基于事实的合理推断
仍然没有验证的猜测
当前实验只能否定的具体实现
下一步最便宜的验证
```

- [ ] **Step 8: Commit only small reproducibility files**

Do not add Parquet, model files, or caches. Stage the registry and any small generated summaries explicitly, verify the staged list, then commit:

```bash
git add rogii_clean/experiments/registry.jsonl
git commit -m "record simple lgbm baseline cv"
```

---

## Plan Self-Review

- Spec coverage: all 12 features, frozen target/model/fold, leakage checks, staged execution, resume, artifacts, and metrics are assigned to explicit tasks.
- Placeholder scan: no unfinished marker or unspecified decision remains.
- Type consistency: `FEATURE_COLUMNS`, `build_simple_lgbm_rows`, `build_feature_table`, `fold_indices`, and `restore_absolute_tvt` keep the same names and return types across tasks.
- Scope: no Typewell, PF, Beam, spatial labels, model tuning, or postprocessing is introduced.
