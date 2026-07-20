from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# 从仓库父目录运行时，强制导入本项目的 scripts/src。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pfmd02_representation_oracle_ladder import (
    FORMAL_REFERENCE_RMSE,
    build_metrics_tables,
    checkpoint_matches,
    evaluate_one_well,
)
from scripts import run_p3_pfmd02_representation_oracle_ladder as runner
from src.p3_pfmd02_representation_oracle import build_ordered_representatives


FROZEN_SHARED_FINGERPRINT = "91053aa823f16f7f58478e5f890c11a4450250c94d1804a85c659dc4556468f0"
FROZEN_MODE_FINGERPRINT = "0e37e94cfe0e7c7a9d9be5b3d456a6b29e6738a1f7df5c2c8c3a3fdf19e2a8d4"
FROZEN_PFM01_CORE_SHA256 = "2794c417dc33c8887135224d119553f168c446230844e1ad8db74831c7e510a0"


def _synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows = 6
    row_index = np.arange(20, 20 + rows)
    md = np.arange(rows, dtype=np.float64) * 100.0 + 1000.0
    last_tvt = 100.0
    target_delta = np.arange(rows, dtype=np.float64)
    target_tvt = target_delta + last_tvt
    seed_delta = np.empty((128, rows), dtype=np.float64)
    seed_delta[:43] = target_delta - 2.0
    seed_delta[43:86] = target_delta
    seed_delta[86:] = target_delta + 2.0
    # 保留三团，同时避免簇内完全重复掩盖 medoid 平局规则。
    seed_delta += np.arange(128, dtype=np.float64)[:, None] * 1e-5
    final_ll = np.zeros(128, dtype=np.float64)
    seed_ids = np.arange(128, dtype=np.int64)
    representatives = build_ordered_representatives(seed_delta, final_ll, seed_ids)

    p2 = pd.DataFrame(
        {
            "well_id": "w",
            "fold": 2,
            "row_index": row_index,
            "md": md,
            "target_tvt": target_tvt,
            "pred_tvt": target_tvt + 3.0,
        }
    )
    mode = pd.DataFrame(
        {
            "well_id": "w",
            "fold": 2,
            "row_index": row_index,
            "last_visible_tvt": last_tvt,
            "pf_mode_low_delta": representatives.center_paths[0],
            "pf_mode_middle_delta": representatives.center_paths[1],
            "pf_mode_high_delta": representatives.center_paths[2],
        }
    )
    seed = {
        "seed_delta": seed_delta,
        "final_ll": final_ll,
        "seed_ids": seed_ids,
        "row_index": row_index,
        "hidden_md": md.astype(np.float32),
        "last_tvt": np.array([last_tvt]),
    }
    return p2, mode, seed


def test_evaluate_one_well_computes_complete_a_to_f_ladder() -> None:
    p2, mode, seed = _synthetic_well()

    per_well, per_segment = evaluate_one_well(
        p2,
        mode,
        seed,
        well_id="w",
        fold=2,
        expected_rows=6,
    )

    for method in (
        "A4",
        "A3",
        "B0",
        "B25",
        "C250_independent",
        "C250_dp",
        "C500_independent",
        "C500_dp",
        "C1000_independent",
        "C1000_dp",
        "D128",
        "D128_plus_P2",
        "E_center3",
        "E_rowmedian3",
        "E_medoid3",
        "F_mode3",
        "F_seed128",
    ):
        assert per_well[f"{method}_sse"] >= 0.0
        assert per_well[f"{method}_rmse"] == pytest.approx(
            np.sqrt(per_well[f"{method}_sse"] / 6)
        )
    assert per_well["E_center3_sse"] == pytest.approx(per_well["A3_sse"])
    assert per_well["B25_q_P2"] >= 0.75 - 1e-12
    assert per_well["D128_seed_id"] in range(128)
    assert set(per_segment["window_ft"]) == {250, 500, 1000}
    assert {"independent_state", "dp_state", "margin"}.issubset(per_segment.columns)


def test_metrics_tables_report_fixed_comparisons_and_per_fold_micro() -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "fold": [1, 2],
            "hidden_rows": [1, 3],
            "A4_sse": [4.0, 12.0],
            "A4_rmse": [2.0, 2.0],
            "B0_sse": [1.0, 3.0],
            "B0_rmse": [1.0, 1.0],
        }
    )

    metrics, per_fold = build_metrics_tables(frame, methods=("A4", "B0"))

    assert metrics["methods"]["A4"]["pooled_micro_rmse"] == pytest.approx(2.0)
    assert metrics["methods"]["B0"]["pooled_micro_rmse"] == pytest.approx(1.0)
    assert metrics["fixed_comparisons"]["B0_vs_A4_improvement_ft"] == pytest.approx(1.0)
    assert len(per_fold) == 4
    assert FORMAL_REFERENCE_RMSE["A3"] == pytest.approx(8.245752)


def test_checkpoint_reuse_rejects_changed_source_signature() -> None:
    checkpoint = {
        "experiment_fingerprint": "fp",
        "well_id": "w",
        "fold": 2,
        "hidden_rows": 6,
        "mode_source_signature": {"size": 10, "mtime_ns": 20, "sha256": "a" * 64},
        "seed_source_signature": {"size": 30, "mtime_ns": 40, "sha256": "b" * 64},
    }

    assert checkpoint_matches(
        checkpoint,
        fingerprint="fp",
        well_id="w",
        fold=2,
        rows=6,
        mode_signature={"size": 10, "mtime_ns": 20, "sha256": "a" * 64},
        seed_signature={"size": 30, "mtime_ns": 40, "sha256": "b" * 64},
    )
    assert not checkpoint_matches(
        checkpoint,
        fingerprint="fp",
        well_id="w",
        fold=2,
        rows=6,
        mode_signature={"size": 10, "mtime_ns": 20, "sha256": "c" * 64},
        seed_signature={"size": 30, "mtime_ns": 40, "sha256": "b" * 64},
    )


def test_checkpoint_without_content_hash_is_not_reusable() -> None:
    checkpoint = {
        "experiment_fingerprint": "fp",
        "well_id": "00012345",
        "fold": 2,
        "hidden_rows": 6,
        "mode_source_signature": {"size": 10, "mtime_ns": 20},
        "seed_source_signature": {"size": 30, "mtime_ns": 40},
    }

    assert not checkpoint_matches(
        checkpoint,
        fingerprint="fp",
        well_id="00012345",
        fold=2,
        rows=6,
        mode_signature={"size": 10, "mtime_ns": 20},
        seed_signature={"size": 30, "mtime_ns": 40},
    )


def test_numeric_leading_zero_well_id_survives_checkpoint_roundtrip_and_resume() -> None:
    segments = pd.DataFrame(
        {
            "well_id": ["00012345", "00012345"],
            "fold": [2, 2],
            "window_ft": [250, 500],
            "segment_index": [0, 0],
        }
    )
    with tempfile.TemporaryDirectory(prefix=".pfmd02_checkpoint_", dir=CLEAN_ROOT) as temporary:
        checkpoint_path = Path(temporary) / "00012345.csv"
        runner.write_csv_atomic(segments, checkpoint_path)

        resumed = runner.read_segment_checkpoint(checkpoint_path)

    assert resumed["well_id"].tolist() == ["00012345", "00012345"]
    assert resumed["well_id"].dtype == object
    combined = pd.concat(
        [resumed, pd.DataFrame({"well_id": ["000d7d20"], "fold": [1]})],
        ignore_index=True,
    ).sort_values("well_id", kind="stable")
    assert combined["well_id"].tolist() == ["00012345", "00012345", "000d7d20"]


@pytest.mark.parametrize(
    ("cache_fingerprint", "format_version", "message"),
    [
        ("wrong", 1, "fingerprint"),
        (FROZEN_SHARED_FINGERPRINT, 99, "format"),
    ],
)
def test_seed_loader_rejects_wrong_frozen_fingerprint_or_format(
    cache_fingerprint: str,
    format_version: int,
    message: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix=".pfmd02_seed_", dir=CLEAN_ROOT) as temporary:
        path = Path(temporary) / "00012345.npz"
        np.savez(
            path,
            seed_delta=np.zeros((128, 1), dtype=np.float32),
            final_ll=np.zeros(128),
            seed_ids=np.arange(128),
            row_index=np.array([0]),
            hidden_md=np.array([100.0], dtype=np.float32),
            last_tvt=np.array([50.0]),
            _cache_fingerprint=np.array([cache_fingerprint]),
            _format_version=np.array([format_version]),
        )

        with pytest.raises(ValueError, match=message):
            runner.load_seed_npz(path, expected_rows=1)


def test_mode_loader_rejects_cache_fingerprint_not_matching_config() -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["00012345"],
            "fold": [2],
            "row_index": [0],
            "last_visible_tvt": [50.0],
            "pf_mode_low_delta": [0.0],
            "pf_mode_middle_delta": [1.0],
            "pf_mode_high_delta": [2.0],
            "_cache_fingerprint": ["wrong"],
        }
    )
    with tempfile.TemporaryDirectory(prefix=".pfmd02_mode_", dir=CLEAN_ROOT) as temporary:
        path = Path(temporary) / "00012345.parquet"
        frame.to_parquet(path, index=False)

        with pytest.raises(ValueError, match="fingerprint"):
            runner.load_mode_cache(path, expected_fingerprint=FROZEN_MODE_FINGERPRINT)


def test_code_hash_contract_includes_pfm01_mode_dependency() -> None:
    hashes = runner.build_code_hash_contract()

    assert set(hashes) == {"runner", "oracle_core", "pfm01_mode_core"}
    assert hashes["pfm01_mode_core"] == FROZEN_PFM01_CORE_SHA256


def test_mode_config_content_hash_and_frozen_fingerprints_are_validated() -> None:
    frozen_path = CLEAN_ROOT / "artifacts" / "P3_PFM02_mode_paths_v1" / "config.json"

    config = runner.load_and_validate_mode_config(frozen_path)

    assert config["experiment_fingerprint"] == FROZEN_MODE_FINGERPRINT
    assert config["shared_fingerprint"] == FROZEN_SHARED_FINGERPRINT
    assert config["pfm01_core_sha256"] == FROZEN_PFM01_CORE_SHA256
    with tempfile.TemporaryDirectory(prefix=".pfmd02_config_", dir=CLEAN_ROOT) as temporary:
        changed_path = Path(temporary) / "config.json"
        changed_path.write_text(frozen_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="SHA256"):
            runner.load_and_validate_mode_config(changed_path)
