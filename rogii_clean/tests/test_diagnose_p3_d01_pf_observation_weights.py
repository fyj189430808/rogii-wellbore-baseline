"""P3-D01 Task 3：影子隔离、合法缓存和可恢复运行器测试。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import diagnose_p3_d01_pf_observation_weights as runner  # noqa: E402


def _write_small_raw_well(raw_train_dir: Path, well_id: str = "well_a") -> None:
    """写入带真值和 surface 的小 CSV，用来证明读取层只返回合法列。"""

    raw_train_dir.mkdir(parents=True, exist_ok=True)
    horizontal = pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0],
            "X": [10.0, 11.0, 12.0],
            "Y": [20.0, 21.0, 22.0],
            "Z": [1000.0, 1001.0, 1002.0],
            "GR": [30.0, 31.0, 32.0],
            "TVT_input": [100.0, np.nan, np.nan],
            "TVT": [100.0, 101.0, 102.0],
            "ANCC": [9000.0, 9001.0, 9002.0],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": [90.0, 100.0, 110.0],
            "GR": [20.0, 30.0, 40.0],
            "Geology": ["a", "b", "c"],
        }
    )
    horizontal.to_csv(raw_train_dir / f"{well_id}__horizontal_well.csv", index=False)
    typewell.to_csv(raw_train_dir / f"{well_id}__typewell.csv", index=False)


def _fake_paths() -> dict[str, np.ndarray]:
    """构造两行、字段与冻结 PF 完全一致的小路径。"""

    return {
        "row_index": np.array([1, 2], dtype=np.int64),
        "last_visible_tvt": np.array([100.0, 100.0], dtype=np.float32),
        "pf128_mean_tvt": np.array([101.0, 102.0], dtype=np.float32),
        "pf128_mean_delta": np.array([1.0, 2.0], dtype=np.float32),
        "pf128_scale_3_delta": np.array([1.1, 2.1], dtype=np.float32),
        "pf128_scale_5_delta": np.array([1.2, 2.2], dtype=np.float32),
        "pf128_scale_8_delta": np.array([1.3, 2.3], dtype=np.float32),
        "pf128_scale_12_delta": np.array([1.4, 2.4], dtype=np.float32),
    }


def _fake_report() -> dict[str, Any]:
    """构造 runner 必须落盘的最小合法单井报告。"""

    report = {
        "hidden_rows": 2,
        "observed_gr_rows": 2,
        "observed_gr_fraction": 1.0,
        "gr_sigma": 12.0,
    }
    report.update(
        runner.rebuild_likelihood_report(
            np.array([-3.0, -2.0, -1.0]),
            hidden_rows=2,
            observed_gr_rows=2,
        )
    )
    return report


def _make_task(
    tmp_path: Path,
    *,
    fingerprint: str = "fingerprint-a",
    verify_legacy_cache: bool = False,
) -> dict[str, Any]:
    """创建可供 generate_one_well 使用的小任务。"""

    raw_train_dir = tmp_path / "raw_train"
    _write_small_raw_well(raw_train_dir)
    return {
        "well_id": "well_a",
        "fold": 0,
        "hidden_rows": 2,
        "raw_train_dir": str(raw_train_dir),
        "artifact_dir": str(tmp_path / "artifact"),
        "fingerprint": fingerprint,
        "particle_filter": {"number_of_seeds": 3},
        "expected_number_of_seeds": 3,
        "verify_legacy_cache": verify_legacy_cache,
        "source_legal_cache_dir": str(tmp_path / "old_cache"),
        "source_legal_runtime_dir": str(tmp_path / "old_runtime"),
    }


def _install_fake_pf(monkeypatch: pytest.MonkeyPatch, call_count: dict[str, int]) -> None:
    """用极小确定性回放替代真实 Numba PF，单测不运行昂贵真实井。"""

    def fake_replay(
        horizontal: pd.DataFrame,
        typewell: pd.DataFrame,
        parameters: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
        del horizontal, typewell, parameters
        call_count["value"] += 1
        return np.array([-3.0, -2.0, -1.0]), _fake_report(), _fake_paths()

    monkeypatch.setattr(runner, "run_pf_likelihood_audit", fake_replay)


def test_load_development_registry_excludes_shadow_before_tasks(tmp_path: Path) -> None:
    fold_path = tmp_path / "fold.csv"
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame(
        {
            "well_id": ["well_e", "well_a", "well_d", "well_b", "well_c"],
            "fold": [4, 0, 3, 1, 2],
            "hidden_rows": [50, 10, 40, 20, 30],
        }
    ).to_csv(fold_path, index=False)
    pd.DataFrame(
        {
            "well_id": ["well_a", "well_c", "not_selected"],
            "is_shadow": [True, True, False],
        }
    ).to_csv(shadow_path, index=False)

    development = runner.load_development_registry(fold_path, shadow_path)

    assert development["well_id"].tolist() == ["well_b", "well_d", "well_e"]
    assert set(development["well_id"]).isdisjoint({"well_a", "well_c"})
    assert development["hidden_rows"].sum() == 110


def test_load_development_registry_accepts_shadow_file_without_flag(tmp_path: Path) -> None:
    fold_path = tmp_path / "fold.csv"
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d", "e"],
            "fold": [0, 1, 2, 3, 4],
            "hidden_rows": [1, 1, 1, 1, 1],
        }
    ).to_csv(fold_path, index=False)
    pd.DataFrame({"well_id": ["b", "d"]}).to_csv(shadow_path, index=False)

    development = runner.load_development_registry(fold_path, shadow_path)

    assert development["well_id"].tolist() == ["a", "c", "e"]


def test_read_legal_inputs_physically_limits_csv_columns(tmp_path: Path) -> None:
    raw_train_dir = tmp_path / "raw_train"
    _write_small_raw_well(raw_train_dir)

    horizontal, typewell = runner.read_legal_inputs(raw_train_dir, "well_a")

    assert horizontal.columns.tolist() == ["MD", "Z", "GR", "TVT_input"]
    assert typewell.columns.tolist() == ["TVT", "GR"]
    assert "TVT" not in horizontal.columns
    assert "ANCC" not in horizontal.columns


def test_generate_one_well_resumes_and_fingerprint_change_recalculates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"value": 0}
    _install_fake_pf(monkeypatch, call_count)
    first_task = _make_task(tmp_path, fingerprint="fingerprint-a")

    first = runner.generate_one_well(first_task)
    repeated = runner.generate_one_well(first_task)
    changed_task = dict(first_task, fingerprint="fingerprint-b")
    changed = runner.generate_one_well(changed_task)

    cache_path = Path(first_task["artifact_dir"]) / "legal_likelihood_cache" / "well_a.npz"
    assert first["cache_hit"] is False
    assert repeated["cache_hit"] is True
    assert changed["cache_hit"] is False
    assert call_count["value"] == 2
    assert changed["hidden_tvt_read"] is False
    assert changed["legal_horizontal_columns"] == ["MD", "Z", "GR", "TVT_input"]
    assert changed["typewell_columns"] == ["TVT", "GR"]
    assert changed["cache_sha256"] == runner.file_sha256(cache_path)


def _write_old_legal_outputs(
    tmp_path: Path,
    paths: dict[str, np.ndarray],
    report: dict[str, Any],
) -> tuple[Path, Path]:
    """写出字段与冻结 P2-P01 相同、行序故意打乱的旧合法产物。"""

    cache_path = tmp_path / "old_cache" / "well_a.parquet"
    runtime_path = tmp_path / "old_runtime" / "well_a.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    old_cache = pd.DataFrame(paths)
    old_cache.insert(0, "well_id", "well_a")
    old_cache["pf128_seed0_delta"] = [0.5, 1.5]
    old_cache["pf128_seed_std"] = [0.2, 0.3]
    old_cache["_cache_fingerprint"] = "old-fingerprint"
    old_cache.iloc[::-1].to_parquet(cache_path, index=False)
    runtime_path.write_text(
        json.dumps(
            {
                "experiment_fingerprint": "old-fingerprint",
                "cache_sha256": runner.file_sha256(cache_path),
                "quality": {
                    "pf_best_ll_per_row": report["pf_best_ll_per_row"],
                    "pf_ll_spread": report["pf_ll_spread"],
                    "pf_gr_sigma": report["gr_sigma"],
                }
            }
        ),
        encoding="utf-8",
    )
    return cache_path, runtime_path


def test_reconcile_legacy_outputs_aligns_by_row_index_and_requires_exact_paths(
    tmp_path: Path,
) -> None:
    paths = _fake_paths()
    report = _fake_report()
    cache_path, runtime_path = _write_old_legal_outputs(tmp_path, paths, report)

    reconciliation = runner.reconcile_legacy_outputs(
        well_id="well_a",
        hidden_rows=2,
        path_features=paths,
        report=report,
        legacy_cache_path=cache_path,
        legacy_runtime_path=runtime_path,
    )

    assert reconciliation["maximum_path_difference"] == 0.0
    assert reconciliation["maximum_quality_difference"] == 0.0
    assert set(reconciliation["path_max_abs_difference"]) == set(paths)

    changed_paths = {name: values.copy() for name, values in paths.items()}
    changed_paths["pf128_scale_5_delta"][0] += np.float32(0.25)
    with pytest.raises(ValueError, match="逐位一致"):
        runner.reconcile_legacy_outputs(
            well_id="well_a",
            hidden_rows=2,
            path_features=changed_paths,
            report=report,
            legacy_cache_path=cache_path,
            legacy_runtime_path=runtime_path,
        )


def test_reconcile_legacy_outputs_requires_exact_quality_values(tmp_path: Path) -> None:
    paths = _fake_paths()
    report = _fake_report()
    cache_path, runtime_path = _write_old_legal_outputs(tmp_path, paths, report)
    changed_report = dict(report, pf_ll_spread=report["pf_ll_spread"] + 1e-9)

    with pytest.raises(ValueError, match="质量诊断"):
        runner.reconcile_legacy_outputs(
            well_id="well_a",
            hidden_rows=2,
            path_features=paths,
            report=changed_report,
            legacy_cache_path=cache_path,
            legacy_runtime_path=runtime_path,
        )


@pytest.mark.parametrize(
    "damage_kind",
    [
        "corrupt_npz",
        "npz_fingerprint",
        "runtime_fingerprint",
        "runtime_missing_field",
        "cache_sha",
        "legal_report",
        "ll_with_synchronized_cache_sha",
    ],
)
def test_invalid_cache_or_runtime_never_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage_kind: str,
) -> None:
    call_count = {"value": 0}
    _install_fake_pf(monkeypatch, call_count)
    task = _make_task(tmp_path)
    runtime = runner.generate_one_well(task)
    cache_path = Path(task["artifact_dir"]) / "legal_likelihood_cache" / "well_a.npz"
    runtime_path = Path(task["artifact_dir"]) / "legal_runtime" / "well_a.json"

    if damage_kind == "corrupt_npz":
        cache_path.write_bytes(b"broken archive")
    elif damage_kind == "npz_fingerprint":
        runner.write_likelihood_cache(cache_path, np.array([-3.0, -2.0, -1.0]), "wrong")
        runtime["cache_sha256"] = runner.file_sha256(cache_path)
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    elif damage_kind == "runtime_fingerprint":
        runtime["experiment_fingerprint"] = "wrong"
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    elif damage_kind == "runtime_missing_field":
        runtime.pop("hidden_tvt_read")
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    elif damage_kind == "cache_sha":
        runtime["cache_sha256"] = "0" * 64
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    elif damage_kind == "legal_report":
        runtime["legal_report"]["observed_gr_fraction"] = 0.5
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    elif damage_kind == "ll_with_synchronized_cache_sha":
        runner.write_likelihood_cache(
            cache_path,
            np.array([-30.0, -2.0, -1.0]),
            task["fingerprint"],
        )
        runtime["cache_sha256"] = runner.file_sha256(cache_path)
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    assert runner.load_valid_cache_hit(task) is None


def test_run_legal_generation_collects_errors_and_gate_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        {"well_id": "good", "fold": 0, "hidden_rows": 1},
        {"well_id": "bad", "fold": 1, "hidden_rows": 1},
    ]

    def fake_generate(task: dict[str, Any]) -> dict[str, Any]:
        if task["well_id"] == "bad":
            raise RuntimeError("simulated failure")
        return {
            "well_id": "good",
            "fold": 0,
            "hidden_rows": 1,
            "cache_hit": False,
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(runner, "generate_one_well", fake_generate)
    runtimes, errors = runner.run_legal_generation(tasks, workers=2)

    assert [runtime["well_id"] for runtime in runtimes] == ["good"]
    assert errors[0]["well_id"] == "bad"
    with pytest.raises(RuntimeError, match="合法生成失败"):
        runner.ensure_legal_generation_succeeded(errors)


def test_mode_selection_is_predeclared_and_sorted() -> None:
    development = pd.DataFrame(
        {
            "well_id": ["well_e", "well_a", "well_d", "well_b", "well_c"],
            "fold": [4, 0, 3, 1, 2],
            "hidden_rows": [50, 10, 40, 20, 30],
        }
    )

    assert runner.select_mode_registry(development, "smoke")["well_id"].tolist() == [
        "well_a"
    ]
    assert runner.select_mode_registry(development, "fold01")["well_id"].tolist() == [
        "well_a",
        "well_b",
    ]
    assert runner.select_mode_registry(development, "all")["well_id"].tolist() == [
        "well_a",
        "well_b",
        "well_c",
        "well_d",
        "well_e",
    ]


def test_frozen_config_is_valid_and_rejects_contract_changes() -> None:
    config_path = CLEAN_ROOT / "configs" / "p3_d01_pf_observation_weight_audit_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runner.validate_config(config)

    changed = copy.deepcopy(config)
    changed["shadow_target_access"] = True
    with pytest.raises(ValueError, match="shadow_target_access"):
        runner.validate_config(changed)

    changed_counts = copy.deepcopy(config)
    changed_counts["development_wells"] = 658
    with pytest.raises(ValueError, match="development_wells"):
        runner.validate_config(changed_counts)


@pytest.mark.parametrize(
    "field",
    [
        "fold_registry",
        "fold_registry_sha256",
        "shadow_registry",
        "shadow_registry_sha256",
        "raw_train_dir",
        "source_pf_config",
        "source_pf_config_sha256",
        "source_pf_core",
        "source_pf_core_sha256",
        "audit_core",
        "source_model_predictions",
        "source_model_predictions_sha256",
        "source_pf_legal_cache_dir",
        "source_pf_legal_runtime_dir",
        "source_pf_artifact_fingerprint",
    ],
)
def test_config_rejects_every_frozen_path_hash_and_source_fingerprint(field: str) -> None:
    config_path = CLEAN_ROOT / "configs" / "p3_d01_pf_observation_weight_audit_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(config)
    changed[field] = "changed-but-still-non-empty"

    with pytest.raises(ValueError, match=field):
        runner.validate_config(changed)


def test_cache_hit_rereads_old_quality_even_if_runtime_sha_is_synchronized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"value": 0}
    _install_fake_pf(monkeypatch, call_count)
    paths = _fake_paths()
    report = _fake_report()
    _cache_path, old_runtime_path = _write_old_legal_outputs(tmp_path, paths, report)
    task = _make_task(tmp_path, verify_legacy_cache=True)
    task["source_pf_artifact_fingerprint"] = "old-fingerprint"
    runtime = runner.generate_one_well(task)

    old_runtime = json.loads(old_runtime_path.read_text(encoding="utf-8"))
    old_runtime["quality"]["pf_ll_spread"] += 1.0
    old_runtime_path.write_text(json.dumps(old_runtime), encoding="utf-8")

    new_runtime_path = Path(task["artifact_dir"]) / "legal_runtime" / "well_a.json"
    runtime["legacy_reconciliation"]["source_legal_runtime_sha256"] = runner.file_sha256(
        old_runtime_path
    )
    new_runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    assert runner.load_valid_cache_hit(task) is None
