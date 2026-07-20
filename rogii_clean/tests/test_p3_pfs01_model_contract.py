"""P3-PFS01 路径诊断与 44 列 LightGBM 阶段的冻结合同测试。"""

from __future__ import annotations

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

from scripts import run_p3_pfs01_fixed_lag_model_cv as runner  # noqa: E402
from scripts.run_p3_pfs01_fixed_lag_particle_smoothing import (  # noqa: E402
    CACHE_COLUMNS,
    EXPERIMENT_ID,
    HORIZONTAL_USECOLS,
    RUNTIME_NONNEGATIVE_INTEGER_FIELDS,
    TYPEWELL_USECOLS,
    build_legal_cache_frame,
    file_sha256,
)


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """在项目目录内创建临时目录，绕开 Windows 全局 pytest 临时目录权限问题。"""

    parent = CLEAN_ROOT / "artifacts" / "_pytest_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="pfs01_model_", dir=parent))
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _load_config() -> dict[str, object]:
    """读取待冻结的 PFS01 模型阶段配置。"""

    path = CLEAN_ROOT / "configs" / "p3_pfs01_fixed_lag_model_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _registry(
    wells: tuple[str, ...] = ("dev_a",),
    rows_per_well: int = 2,
) -> pd.DataFrame:
    """构造只含开发井的合成注册表。"""

    rows: list[dict[str, object]] = []
    for fold, well_id in enumerate(wells):
        rows.append(
            {
                "well_id": well_id,
                "pad_id": f"pad_{well_id}",
                "fold": fold,
                "hidden_rows": rows_per_well,
            }
        )
    return pd.DataFrame(rows)


def _legal_frame(
    well_id: str = "dev_a",
    fold: int = 0,
    rows: int = 2,
    fingerprint: str = "f" * 64,
) -> pd.DataFrame:
    """用真实生成器的构造函数得到严格同 schema 的合成合法缓存。"""

    row_index = np.arange(20, 20 + rows, dtype=np.int64)
    anchor = 100.0 + fold
    paths = np.vstack(
        [
            anchor + np.linspace(1.0, 2.0, rows),
            anchor + np.linspace(1.2, 2.2, rows),
            anchor + np.linspace(1.4, 2.4, rows),
        ]
    )
    return build_legal_cache_frame(
        well_id=well_id,
        fold=fold,
        row_index=row_index,
        last_visible_tvt=anchor,
        smoothed_tvt_paths=paths,
        fingerprint=fingerprint,
    )


def _runtime_payload(cache_path: Path, frame: pd.DataFrame) -> dict[str, object]:
    """构造包含全部必需审计字段的逐井运行记录。"""

    hidden_rows = len(frame)
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "well_id": str(frame["well_id"].iloc[0]),
        "fold": int(frame["fold"].iloc[0]),
        "hidden_rows": hidden_rows,
        "legal_horizontal_columns": list(HORIZONTAL_USECOLS),
        "typewell_columns": list(TYPEWELL_USECOLS),
        "experiment_fingerprint": str(frame["_cache_fingerprint"].iloc[0]),
        "hidden_tvt_read": False,
        "cache_sha256": file_sha256(cache_path),
        "generation_seconds": 1.0,
        "resume_check_seconds": 0.0,
        "resample_count": 0,
        "history_rows_reordered_total": 0,
        "history_particle_copies": 0,
        "max_active_rows": hidden_rows,
        "history_bytes_peak": hidden_rows * 500 * 8 * 2,
    }
    seed_rows = hidden_rows * 128
    for lag in (250, 500, 1000):
        payload[f"lag{lag}_smooth_rows"] = 0
        payload[f"lag{lag}_fallback_rows"] = seed_rows
    assert set(RUNTIME_NONNEGATIVE_INTEGER_FIELDS).issubset(payload)
    return payload


def _write_well(
    cache_dir: Path,
    runtime_dir: Path,
    frame: pd.DataFrame,
    runtime_changes: dict[str, object] | None = None,
) -> None:
    """写入一口合成井的 parquet 与逐井运行记录。"""

    well_id = str(frame["well_id"].iloc[0])
    cache_path = cache_dir / f"{well_id}.parquet"
    frame.to_parquet(cache_path, index=False)
    runtime = _runtime_payload(cache_path, frame)
    if runtime_changes:
        runtime.update(runtime_changes)
    (runtime_dir / f"{well_id}.json").write_text(
        json.dumps(runtime),
        encoding="utf-8",
    )


def _cache_dirs(root: Path) -> tuple[Path, Path]:
    """创建与正式产物一致的两个子目录。"""

    cache_dir = root / "legal_cache"
    runtime_dir = root / "legal_runtime"
    cache_dir.mkdir()
    runtime_dir.mkdir()
    return cache_dir, runtime_dir


def test_feature_contract_is_exact_frozen_41_plus_three() -> None:
    """旧 41 列必须逐名保留，且末尾只能追加三条 PFS 路径。"""

    features = runner.build_formal_feature_names()

    assert runner.NEW_PFS_FEATURES == [
        "pfs_lag250_delta",
        "pfs_lag500_delta",
        "pfs_lag1000_delta",
    ]
    assert features == [*runner.P3B00_FEATURES, *runner.NEW_PFS_FEATURES]
    assert len(features) == len(set(features)) == 44


def test_config_freezes_model_data_and_screening_gate() -> None:
    """模型、开发集、评价行和 folds0-1 门槛必须写死。"""

    config = _load_config()

    assert config["experiment_id"] == "P3_PFS01_fixed_lag_particle_smoothing_v1"
    assert config["baseline_id"] == "P3B00_group5_p2p02_v1"
    assert config["fold_version"] == "balanced_well_5fold_v1"
    assert config["source_pfs_legal_cache_dir"] == (
        "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_cache"
    )
    assert config["source_pfs_runtime_dir"] == (
        "artifacts/P3_PFS01_fixed_lag_particle_smoothing_v1/legal_runtime"
    )
    assert config["source_pfs_fingerprint"] == (
        "4bf62f495ef530bb3ca53d65d15475f7bc3784fe5abaebbbb1053584704c4480"
    )
    assert config["formal_feature_count"] == 44
    assert config["development_wells"] == 657
    assert config["development_hidden_rows"] == 3_211_872
    assert config["model_training"] is True
    assert config["shadow_target_access"] is False
    assert config["success_conditions"][
        "fold01_minimum_combined_improvement_ft"
    ] == 0.20
    assert config["success_conditions"][
        "fold01_maximum_single_fold_degradation_ft"
    ] == 0.25


def test_expected_generator_fingerprint_is_recomputed_from_frozen_safe_sources() -> None:
    """模型阶段应从生成配置和代码哈希重算预期指纹，而不是信任缓存自报。"""

    config = _load_config()

    assert runner.build_expected_generator_fingerprint(config) == (
        "4bf62f495ef530bb3ca53d65d15475f7bc3784fe5abaebbbb1053584704c4480"
    )


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("experiment_id", "changed"),
        ("baseline_id", "changed"),
        ("fold_version", "changed"),
        ("development_wells", 656),
        ("development_hidden_rows", 3_211_871),
        ("formal_feature_count", 43),
        ("model_training", False),
        ("shadow_target_access", True),
    ],
)
def test_frozen_config_rejects_any_registered_change(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad_value: object,
) -> None:
    """正式配置中的任何冻结项漂移都必须在读目标前失败。"""

    config = _load_config()
    config[key] = bad_value
    model_manifest = {
        "feature_count": 41,
        "features": runner.P3B00_FEATURES,
    }
    model_config = json.loads(
        (CLEAN_ROOT / "configs" / "lgbm_feature_baseline_v1.json").read_text(
            encoding="utf-8"
        )
    )

    def fake_read_json(path: Path) -> dict[str, object]:
        if Path(path).name == "feature_list.json":
            return model_manifest
        return model_config

    monkeypatch.setattr(runner, "read_json", fake_read_json)
    with pytest.raises(ValueError, match="冻结|合同|配置"):
        runner.validate_frozen_contract(config)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda model: model.__setitem__("model_family", "other"),
        lambda model: model["params"].__setitem__("n_estimators", 1),
        lambda model: model["params"].__setitem__("random_state", 1),
        lambda model: model["params"].__setitem__("bagging_seed", 1),
        lambda model: model["training_policy"].__setitem__("early_stopping", True),
    ],
)
def test_model_parameter_tree_seed_and_policy_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
    mutator: object,
) -> None:
    """不能借 PFS01 改树数、随机种子、LightGBM 参数或训练策略。"""

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
    with pytest.raises(ValueError, match="LightGBM|训练策略|模型"):
        runner.validate_frozen_contract(config)


def test_fold_spec_and_pre_registered_gate() -> None:
    """只能先跑 folds0-1；完整五折必须通过 0.20/0.25 的预注册门槛。"""

    assert runner.parse_fold_spec("0,1") == [0, 1]
    assert runner.parse_fold_spec("all") == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="0,1"):
        runner.parse_fold_spec("0")

    failed_gain = {
        "success_checks": {
            "combined_improvement_ft": 0.199,
            "worst_fold_improvement_ft": 0.0,
            "folds01_pass": False,
        }
    }
    failed_fold = {
        "success_checks": {
            "combined_improvement_ft": 0.30,
            "worst_fold_improvement_ft": -0.251,
            "folds01_pass": False,
        }
    }
    passed = {
        "success_checks": {
            "combined_improvement_ft": 0.20,
            "worst_fold_improvement_ft": -0.25,
            "folds01_pass": True,
        }
    }
    assert runner.remaining_folds_after_screen([0, 1, 2, 3, 4], failed_gain) == []
    assert runner.remaining_folds_after_screen([0, 1, 2, 3, 4], failed_fold) == []
    assert runner.remaining_folds_after_screen([0, 1, 2, 3, 4], passed) == [2, 3, 4]
    assert runner.remaining_folds_after_screen([0, 1], passed) == []


def test_complete_legal_cache_is_validated_without_truth_and_keeps_row_order(
    workspace_tmp_path: Path,
) -> None:
    """只有所有开发井缓存和 runtime 齐全且未读真值，才颁发真值阶段许可。"""

    cache_dir, runtime_dir = _cache_dirs(workspace_tmp_path)
    registry = _registry(("dev_a", "dev_b"))
    _write_well(cache_dir, runtime_dir, _legal_frame("dev_a", 0))
    _write_well(cache_dir, runtime_dir, _legal_frame("dev_b", 1))

    bundle = runner.validate_complete_legal_cache(
        registry=registry,
        cache_dir=cache_dir,
        runtime_dir=runtime_dir,
        shadow_ids={"shadow_x"},
        expected_wells=2,
        expected_rows=4,
        expected_fingerprint="f" * 64,
    )

    assert bundle.audit["legal_cache_complete"] is True
    assert bundle.audit["hidden_tvt_read"] is False
    assert bundle.audit["development_wells"] == 2
    assert bundle.audit["development_hidden_rows"] == 4
    assert tuple(bundle.path_table.columns) == (
        "well_id",
        "row_index",
        "last_visible_tvt",
        *runner.NEW_PFS_FEATURES,
    )
    assert bundle.path_table["well_id"].tolist() == [
        "dev_a",
        "dev_a",
        "dev_b",
        "dev_b",
    ]


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing_cache", "缺少|完整"),
        ("missing_runtime", "运行记录|完整"),
        ("hidden_tvt", "隐藏 TVT"),
        ("wrong_rows", "行数"),
        ("wrong_fold", "fold"),
        ("duplicate_key", "重复"),
        ("nonfinite", "NaN|Inf|有限"),
        ("extra_column", "schema|列"),
        ("bad_fingerprint", "指纹"),
        ("bad_sha", "SHA|哈希"),
    ],
)
def test_legal_cache_gate_rejects_incomplete_or_tainted_inputs(
    workspace_tmp_path: Path,
    damage: str,
    message: str,
) -> None:
    """任何缺井、错键、污染或运行血缘异常都必须阻止读取目标。"""

    cache_dir, runtime_dir = _cache_dirs(workspace_tmp_path)
    registry = _registry()
    frame = _legal_frame()
    runtime_changes: dict[str, object] | None = None
    if damage == "wrong_rows":
        frame = frame.iloc[:1].copy()
    elif damage == "wrong_fold":
        frame["fold"] = 4
    elif damage == "duplicate_key":
        frame.loc[1, "row_index"] = frame.loc[0, "row_index"]
    elif damage == "nonfinite":
        frame.loc[0, "pfs_lag250_delta"] = np.inf
    elif damage == "extra_column":
        frame["not_legal"] = 1.0
    elif damage == "hidden_tvt":
        runtime_changes = {"hidden_tvt_read": True}
    elif damage == "bad_fingerprint":
        runtime_changes = {"experiment_fingerprint": "bad"}
    elif damage == "bad_sha":
        runtime_changes = {"cache_sha256": "0" * 64}

    if damage != "missing_cache":
        _write_well(cache_dir, runtime_dir, frame, runtime_changes)
    if damage == "missing_runtime":
        (runtime_dir / "dev_a.json").unlink()

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        runner.validate_complete_legal_cache(
            registry=registry,
            cache_dir=cache_dir,
            runtime_dir=runtime_dir,
            shadow_ids=set(),
            expected_wells=1,
            expected_rows=2,
            expected_fingerprint="f" * 64,
        )


def test_legal_cache_gate_rejects_shadow_or_extra_well(
    workspace_tmp_path: Path,
) -> None:
    """合法目录只能包含开发井，影子井和未登记井都不能混入。"""

    cache_dir, runtime_dir = _cache_dirs(workspace_tmp_path)
    registry = _registry()
    _write_well(cache_dir, runtime_dir, _legal_frame())
    _write_well(cache_dir, runtime_dir, _legal_frame("shadow_x", 0))

    with pytest.raises(ValueError, match="影子|未登记|额外"):
        runner.validate_complete_legal_cache(
            registry=registry,
            cache_dir=cache_dir,
            runtime_dir=runtime_dir,
            shadow_ids={"shadow_x"},
            expected_wells=1,
            expected_rows=2,
            expected_fingerprint="f" * 64,
        )


def test_legal_cache_gate_rejects_wrong_but_internally_consistent_fingerprint(
    workspace_tmp_path: Path,
) -> None:
    """缓存和 runtime 即使彼此自洽，也必须等于冻结生成配置应产生的指纹。"""

    cache_dir, runtime_dir = _cache_dirs(workspace_tmp_path)
    registry = _registry()
    wrong_fingerprint = "1" * 64
    _write_well(
        cache_dir,
        runtime_dir,
        _legal_frame(fingerprint=wrong_fingerprint),
    )

    with pytest.raises(ValueError, match="冻结生成配置|预期指纹"):
        runner.validate_complete_legal_cache(
            registry=registry,
            cache_dir=cache_dir,
            runtime_dir=runtime_dir,
            shadow_ids=set(),
            expected_wells=1,
            expected_rows=2,
            expected_fingerprint="f" * 64,
        )


def test_truth_stage_permission_requires_exact_complete_audit() -> None:
    """真值读取必须依赖完整缓存审计结果，不能靠调用者口头承诺。"""

    valid = {
        "legal_cache_complete": True,
        "hidden_tvt_read": False,
        "development_wells": 657,
        "development_hidden_rows": 3_211_872,
        "shadow_overlap": 0,
    }
    runner.require_truth_stage_permission(valid, expected_wells=657, expected_rows=3_211_872)

    for key, bad_value in (
        ("legal_cache_complete", False),
        ("hidden_tvt_read", True),
        ("development_wells", 656),
        ("development_hidden_rows", 3_211_871),
        ("shadow_overlap", 1),
    ):
        damaged = dict(valid)
        damaged[key] = bad_value
        with pytest.raises(RuntimeError, match="真值|合法缓存|影子"):
            runner.require_truth_stage_permission(
                damaged,
                expected_wells=657,
                expected_rows=3_211_872,
            )


def test_merge_is_one_to_one_order_preserving_and_anchor_checked() -> None:
    """三列只能按井号和自然行号一对一追加，不能覆盖旧列或错位。"""

    feature_table = pd.DataFrame(
        {
            "well_id": ["dev_a", "dev_a"],
            "row_index": [21, 20],
            "last_visible_tvt": np.array([100.0, 100.0], dtype=np.float32),
            "old_feature": [2.0, 1.0],
        }
    )
    paths = _legal_frame().loc[
        :, ["well_id", "row_index", "last_visible_tvt", *runner.NEW_PFS_FEATURES]
    ]

    merged = runner.merge_pfs_paths(feature_table, paths)

    assert merged["row_index"].tolist() == [21, 20]
    assert merged[runner.NEW_PFS_FEATURES].shape == (2, 3)
    assert merged.loc[0, "pfs_lag250_delta"] == pytest.approx(2.0)

    paths.loc[0, "last_visible_tvt"] = 999.0
    with pytest.raises(ValueError, match="last_visible_tvt"):
        runner.merge_pfs_paths(feature_table, paths)


def test_path_diagnostics_scores_old_scale8_and_three_pfs_paths() -> None:
    """模型训练前必须先输出四条路径自身的 micro、macro 和逐折 RMSE。"""

    table = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "fold": [0, 0, 1, 1],
            "row_index": [0, 1, 0, 1],
            "last_visible_tvt": [100.0, 100.0, 200.0, 200.0],
            "target_tvt": [101.0, 102.0, 201.0, 202.0],
            "pf128_scale_8_delta": [0.0, 1.0, 0.0, 1.0],
            "pfs_lag250_delta": [1.0, 2.0, 1.0, 2.0],
            "pfs_lag500_delta": [2.0, 3.0, 2.0, 3.0],
            "pfs_lag1000_delta": [3.0, 4.0, 3.0, 4.0],
        }
    )

    metrics, per_well, per_fold = runner.score_path_candidates(table)

    assert list(metrics["paths"]) == [
        "pf128_scale_8_delta",
        "pfs_lag250_delta",
        "pfs_lag500_delta",
        "pfs_lag1000_delta",
    ]
    assert metrics["paths"]["pf128_scale_8_delta"]["micro_rmse"] == pytest.approx(1.0)
    assert metrics["paths"]["pfs_lag250_delta"]["micro_rmse"] == pytest.approx(0.0)
    assert metrics["paths"]["pfs_lag500_delta"]["micro_rmse"] == pytest.approx(1.0)
    assert metrics["paths"]["pfs_lag1000_delta"]["micro_rmse"] == pytest.approx(2.0)
    assert len(per_well) == 8
    assert len(per_fold) == 8
    assert set(per_fold["fold"]) == {0, 1}


def test_formal_features_exclude_unregistered_summaries_and_truth() -> None:
    """正式模型不得混入方向、质量、seed 数、位置、分离度或真值摘要。"""

    features = runner.build_formal_feature_names()
    forbidden = (
        "direction",
        "mass",
        "seed_count",
        "p2_position",
        "separation",
        "oracle",
        "target",
        "truth",
        "surface",
    )
    assert not any(token in name.lower() for name in features for token in forbidden)
