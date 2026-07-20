"""P3-PFM02 44 列正式 LightGBM runner 的冻结合同测试。"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pfm02_direct_mode_paths_cv as runner


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="pfm02_cv_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _load_config() -> dict[str, object]:
    return json.loads(
        (CLEAN_ROOT / "configs" / "p3_pfm02_direct_mode_paths_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _base_and_registry(
    wells: tuple[str, ...] = ("dev_a",), rows_per_well: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    for fold, well_id in enumerate(wells):
        registry_rows.append(
            {"well_id": well_id, "fold": fold, "hidden_rows": rows_per_well}
        )
        for offset in range(rows_per_well):
            base_rows.append(
                {
                    "well_id": well_id,
                    "row_index": 20 + offset,
                    "last_visible_tvt": 100.0 + fold,
                    "old_feature": float(offset),
                }
            )
    return pd.DataFrame(base_rows), pd.DataFrame(registry_rows)


def _legal_frame(
    well_id: str = "dev_a",
    fold: int = 0,
    rows: int = 2,
    fingerprint: str = "fp",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "well_id": [well_id] * rows,
            "fold": np.full(rows, fold, dtype=np.int64),
            "row_index": np.arange(20, 20 + rows, dtype=np.int64),
            "last_visible_tvt": np.full(rows, 100.0 + fold, dtype=np.float64),
            "pf_mode_low_delta": np.linspace(1.0, 2.0, rows),
            "pf_mode_middle_delta": np.linspace(1.1, 2.1, rows),
            "pf_mode_high_delta": np.linspace(1.2, 2.2, rows),
            "_cache_fingerprint": [fingerprint] * rows,
        }
    )


def _write_well(
    cache_dir: Path,
    runtime_dir: Path,
    frame: pd.DataFrame,
    *,
    runtime: bool = True,
    hidden_tvt_read: bool = False,
) -> None:
    well_id = str(frame["well_id"].iloc[0])
    frame.to_parquet(cache_dir / f"{well_id}.parquet", index=False)
    if runtime:
        payload = {
            "well_id": well_id,
            "fold": int(frame["fold"].iloc[0]),
            "rows": len(frame),
            "experiment_fingerprint": str(frame["_cache_fingerprint"].iloc[0]),
            "hidden_tvt_read": hidden_tvt_read,
        }
        (runtime_dir / f"{well_id}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _cache_dirs(tmp_path: Path) -> tuple[Path, Path]:
    cache_dir = tmp_path / "legal_cache"
    runtime_dir = tmp_path / "legal_runtime"
    cache_dir.mkdir()
    runtime_dir.mkdir()
    return cache_dir, runtime_dir


def test_feature_contract_is_exact_frozen_41_plus_three() -> None:
    features = runner.build_formal_feature_names()

    assert features == [*runner.P3B00_FEATURES, *runner.NEW_MODE_FEATURES]
    assert runner.NEW_MODE_FEATURES == [
        "pf_mode_low_delta",
        "pf_mode_middle_delta",
        "pf_mode_high_delta",
    ]
    assert len(features) == len(set(features)) == 44


def test_config_freezes_dataset_model_sources_and_counts() -> None:
    config = _load_config()

    assert config["experiment_id"] == "P3_PFM02_direct_mode_paths_v1"
    assert config["source_pfm02_legal_cache_dir"] == (
        "artifacts/P3_PFM02_mode_paths_v1/legal_cache"
    )
    assert config["source_pfm02_runtime_dir"] == (
        "artifacts/P3_PFM02_mode_paths_v1/legal_runtime"
    )
    assert config["formal_feature_count"] == 44
    assert config["development_wells"] == 657
    assert config["development_hidden_rows"] == 3_211_872
    assert config["development_fold_well_counts"] == {
        "0": 131,
        "1": 132,
        "2": 131,
        "3": 132,
        "4": 131,
    }
    assert config["development_fold_hidden_row_counts"] == {
        "0": 651_881,
        "1": 630_395,
        "2": 645_557,
        "3": 649_717,
        "4": 634_322,
    }
    assert config["source_p3b00_predictions_sha256"] == (
        "8109514eb125f8beda2559f39e1d9f54396d41fd38fa76114423c8dd6bca4d37"
    )


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("experiment_id", "changed"),
        ("baseline_id", "changed"),
        ("fold_version", "changed"),
        ("total_wells", 772),
        ("shadow_wells", 115),
        ("development_wells", 656),
        ("development_hidden_rows", 3_211_871),
        ("development_fold_well_counts", {"0": 657}),
        ("development_fold_hidden_row_counts", {"0": 3_211_872}),
        ("baseline_development_micro_rmse", 10.0),
        ("model_training", False),
        ("shadow_target_access", True),
    ],
)
def test_frozen_config_contract_rejects_any_registered_change(
    key: str, bad_value: object
) -> None:
    config = _load_config()
    config[key] = bad_value

    with pytest.raises(ValueError, match="冻结|配置|合同"):
        runner.validate_frozen_contract(config)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda model: model.__setitem__("model_family", "other"),
        lambda model: model["params"].__setitem__("n_estimators", 1),
        lambda model: model["params"].__setitem__("random_state", 1),
        lambda model: model["params"].__setitem__("bagging_seed", 1),
        lambda model: model["training_policy"].__setitem__("early_stopping", True),
        lambda model: model["training_policy"].__setitem__("uniform_row_weight", False),
    ],
)
def test_frozen_model_contract_rejects_parameter_seed_or_policy_change(
    monkeypatch: pytest.MonkeyPatch, mutator: object
) -> None:
    config = _load_config()
    manifest = {"feature_count": 41, "features": runner.P3B00_FEATURES}
    model = json.loads(
        (CLEAN_ROOT / "configs" / "lgbm_feature_baseline_v1.json").read_text(
            encoding="utf-8"
        )
    )
    mutator(model)

    def fake_read_json(path: Path) -> dict[str, object]:
        return manifest if Path(path).name == "feature_list.json" else model

    monkeypatch.setattr(runner, "read_json", fake_read_json)
    with pytest.raises(ValueError, match="LightGBM|模型|training policy"):
        runner.validate_frozen_contract(config)


def test_fold_spec_and_automatic_stop_contract() -> None:
    assert runner.parse_fold_spec("0,1") == [0, 1]
    assert runner.parse_fold_spec("all") == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="0,1"):
        runner.parse_fold_spec("0")

    failed = {"success_checks": {"folds01_pass": False}}
    passed = {"success_checks": {"folds01_pass": True}}
    assert runner.remaining_folds_after_screen([0, 1], passed) == []
    assert runner.remaining_folds_after_screen([0, 1, 2, 3, 4], failed) == []
    assert runner.remaining_folds_after_screen([0, 1, 2, 3, 4], passed) == [2, 3, 4]


def test_mode_cache_merge_is_one_to_one_order_preserving_and_finite(
    workspace_tmp_path: Path,
) -> None:
    cache_dir, runtime_dir = _cache_dirs(workspace_tmp_path)
    base, registry = _base_and_registry()
    _write_well(cache_dir, runtime_dir, _legal_frame())
    base = base.iloc[::-1].reset_index(drop=True)

    merged, fingerprint = runner.merge_pfm02_legal_cache(
        base, registry, cache_dir, runtime_dir, shadow_ids={"shadow_x"}
    )

    assert fingerprint == "fp"
    assert merged["row_index"].tolist() == base["row_index"].tolist()
    assert merged[runner.NEW_MODE_FEATURES].shape == (2, 3)
    assert np.isfinite(merged[runner.NEW_MODE_FEATURES].to_numpy()).all()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_cache", "缺少"),
        ("wrong_schema", "schema"),
        ("wrong_fold", "fold"),
        ("wrong_rows", "行数"),
        ("duplicate_key", "重复"),
        ("missing_runtime", "运行记录"),
        ("hidden_tvt", "隐藏 TVT"),
        ("anchor", "last_visible_tvt"),
        ("nonfinite", "NaN|Inf|有限"),
    ],
)
def test_mode_cache_rejects_invalid_per_well_contract(
    workspace_tmp_path: Path, case: str, message: str
) -> None:
    cache_dir, runtime_dir = _cache_dirs(workspace_tmp_path)
    base, registry = _base_and_registry()
    frame = _legal_frame()
    write_runtime = True
    hidden_tvt_read = False
    if case == "wrong_schema":
        frame["mass_summary"] = 1.0
    elif case == "wrong_fold":
        frame["fold"] = 3
    elif case == "wrong_rows":
        frame = frame.iloc[:1].copy()
    elif case == "duplicate_key":
        frame.loc[1, "row_index"] = frame.loc[0, "row_index"]
    elif case == "missing_runtime":
        write_runtime = False
    elif case == "hidden_tvt":
        hidden_tvt_read = True
    elif case == "anchor":
        frame["last_visible_tvt"] = 101.0
    elif case == "nonfinite":
        frame.loc[0, "pf_mode_low_delta"] = np.inf
    if case != "missing_cache":
        _write_well(
            cache_dir,
            runtime_dir,
            frame,
            runtime=write_runtime,
            hidden_tvt_read=hidden_tvt_read,
        )

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        runner.merge_pfm02_legal_cache(
            base, registry, cache_dir, runtime_dir, shadow_ids=set()
        )


def test_mode_cache_rejects_nonuniform_fingerprint_and_shadow_overlap(
    workspace_tmp_path: Path,
) -> None:
    cache_dir, runtime_dir = _cache_dirs(workspace_tmp_path)
    base, registry = _base_and_registry(("dev_a", "dev_b"))
    _write_well(cache_dir, runtime_dir, _legal_frame("dev_a", 0, fingerprint="fp-a"))
    _write_well(cache_dir, runtime_dir, _legal_frame("dev_b", 1, fingerprint="fp-b"))
    with pytest.raises(ValueError, match="指纹不统一"):
        runner.merge_pfm02_legal_cache(
            base, registry, cache_dir, runtime_dir, shadow_ids=set()
        )

    shutil.copy2(cache_dir / "dev_a.parquet", cache_dir / "shadow_x.parquet")
    with pytest.raises(ValueError, match="影子井"):
        runner.merge_pfm02_legal_cache(
            base, registry, cache_dir, runtime_dir, shadow_ids={"shadow_x"}
        )


def test_formal_features_exclude_forbidden_summaries_and_oracle_sources() -> None:
    config = _load_config()
    features = runner.build_formal_feature_names()
    forbidden = (
        "direction",
        "mass",
        "seed_count",
        "p2_position",
        "separation",
        "oracle",
        "target",
        "surface",
        "geology",
    )

    assert not any(token in name.lower() for name in features for token in forbidden)
    assert not any("oracle" in str(value).lower() for value in config.values())
    assert "PFM01" not in str(config)
