"""用三口井检查 F05a 24 列能否从原始 CSV 确定性重建。"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.f05a_candidate_reproduction import build_candidate_features  # noqa: E402
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "F05a_deterministic_feature_reproduction_v1"
SMOKE_WELLS = ["29de15c8", "2d35f86d", "8b9d8326"]
FORBIDDEN_HORIZONTAL_COLUMNS = [
    "TVT",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def dataframe_hash(frame: pd.DataFrame) -> str:
    values = frame[DIRECT_CANDIDATE_COLUMNS].to_numpy(dtype=np.float32)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def compare_feature(
    well_id: str,
    feature_name: str,
    regenerated: np.ndarray,
    frozen: np.ndarray,
) -> dict:
    difference = regenerated.astype(np.float64) - frozen.astype(np.float64)
    if np.std(regenerated) > 0 and np.std(frozen) > 0:
        correlation = float(np.corrcoef(regenerated, frozen)[0, 1])
    else:
        correlation = float("nan")
    return {
        "well_id": well_id,
        "feature": feature_name,
        "rows": int(len(regenerated)),
        "exact_rate": float(np.mean(regenerated == frozen)),
        "close_1e_5_rate": float(np.mean(np.isclose(regenerated, frozen, atol=1e-5, rtol=0))),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
        "correlation": correlation,
        "family": "stochastic_pf"
        if feature_name in {"pf_ancc_delta", "pf_ancc_std", "pf_z_delta", "pf_vs_z"}
        else "deterministic_beam_ncc",
    }


def main() -> None:
    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_train_dir = PROJECT_ROOT / "input" / "data" / "raw" / "train"
    frozen_cache_path = (
        CLEAN_ROOT
        / "artifacts"
        / "F05a_direct_physical_candidates_v2"
        / "candidate_feature_cache.parquet"
    )
    generator_config_path = CLEAN_ROOT / "configs" / "f05a_candidate_generator_v1.json"
    generator_config = json.loads(generator_config_path.read_text(encoding="utf-8"))
    if generator_config["output_feature_columns"] != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("生成器配置的24列顺序与冻结模型特征顺序不一致")

    frozen = pd.read_parquet(frozen_cache_path)
    frozen["well_id"] = frozen["well_id"].astype(str)
    frozen = frozen.loc[frozen["well_id"].isin(SMOKE_WELLS)].copy()

    regenerated_parts: list[pd.DataFrame] = []
    comparison_rows: list[dict] = []
    well_rows: list[dict] = []
    invariance_rows: list[dict] = []
    total_start = time.perf_counter()
    for well_id in SMOKE_WELLS:
        horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
        typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
        horizontal_full = pd.read_csv(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        horizontal_legal = horizontal_full.drop(
            columns=FORBIDDEN_HORIZONTAL_COLUMNS,
            errors="ignore",
        )

        well_start = time.perf_counter()
        regenerated = build_candidate_features(horizontal_legal, typewell, seed=42)
        repeat = build_candidate_features(horizontal_legal, typewell, seed=42)

        mutated = horizontal_full.copy()
        hidden_mask = mutated["TVT_input"].isna()
        if "TVT" in mutated.columns:
            mutated.loc[hidden_mask, "TVT"] = 1_000_000.0
        for surface_name in FORBIDDEN_HORIZONTAL_COLUMNS[1:]:
            if surface_name in mutated.columns:
                mutated.loc[:, surface_name] = -1_000_000.0
        mutated_result = build_candidate_features(mutated, typewell, seed=42)

        regenerated_hash = dataframe_hash(regenerated)
        repeat_hash = dataframe_hash(repeat)
        mutated_hash = dataframe_hash(mutated_result)
        deterministic_repeat = regenerated_hash == repeat_hash
        hidden_truth_invariant = regenerated_hash == mutated_hash
        if not deterministic_repeat or not hidden_truth_invariant:
            raise ValueError(f"{well_id} 的确定性或隐藏真值不变性测试失败")

        regenerated.insert(0, "well_id", well_id)
        regenerated_parts.append(regenerated)
        frozen_well = frozen.loc[frozen["well_id"] == well_id].reset_index(drop=True)
        if not np.array_equal(
            regenerated["row_index"].to_numpy(dtype=np.int32),
            frozen_well["row_index"].to_numpy(dtype=np.int32),
        ):
            raise ValueError(f"{well_id} 的 row_index 与冻结缓存不一致")
        for feature_name in DIRECT_CANDIDATE_COLUMNS:
            comparison_rows.append(
                compare_feature(
                    well_id,
                    feature_name,
                    regenerated[feature_name].to_numpy(dtype=np.float32),
                    frozen_well[feature_name].to_numpy(dtype=np.float32),
                )
            )
        hidden_rows = int(horizontal_legal["TVT_input"].isna().sum())
        visible_rows = int(horizontal_legal["TVT_input"].notna().sum())
        hidden_gr_missing_rate = float(
            horizontal_legal.loc[horizontal_legal["TVT_input"].isna(), "GR"].isna().mean()
        )
        well_rows.append(
            {
                "well_id": well_id,
                "visible_rows": visible_rows,
                "hidden_rows": hidden_rows,
                "hidden_gr_missing_rate": hidden_gr_missing_rate,
                "seconds_for_three_generations": time.perf_counter() - well_start,
                "regenerated_hash": regenerated_hash,
                "repeat_hash": repeat_hash,
            }
        )
        invariance_rows.append(
            {
                "well_id": well_id,
                "fixed_seed_repeat_exact": deterministic_repeat,
                "hidden_tvt_and_surface_mutation_exact": hidden_truth_invariant,
                "row_index_exact": True,
            }
        )
        print(
            f"{well_id} 完成：可见={visible_rows:,}，隐藏={hidden_rows:,}，"
            f"GR缺失={hidden_gr_missing_rate:.1%}",
            flush=True,
        )

    regenerated_all = pd.concat(regenerated_parts, ignore_index=True)
    comparisons = pd.DataFrame(comparison_rows)
    well_summary = pd.DataFrame(well_rows)
    invariance = pd.DataFrame(invariance_rows)
    regenerated_all.to_parquet(artifact_dir / "regenerated_features.parquet", index=False)
    comparisons.to_csv(artifact_dir / "per_feature_comparison.csv", index=False)
    well_summary.to_csv(artifact_dir / "per_well_summary.csv", index=False)
    invariance.to_csv(artifact_dir / "invariance_tests.csv", index=False)

    family_summary = (
        comparisons.groupby("family", sort=False)
        .agg(
            features=("feature", "nunique"),
            rows=("rows", "sum"),
            mean_exact_rate=("exact_rate", "mean"),
            mean_close_1e_5_rate=("close_1e_5_rate", "mean"),
            mean_mae=("mae", "mean"),
            worst_max_abs=("max_abs", "max"),
            median_correlation=("correlation", "median"),
        )
        .reset_index()
    )
    family_summary.to_csv(artifact_dir / "family_summary.csv", index=False)
    write_json(
        artifact_dir / "config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "source_experiment": "F05a_direct_physical_candidates_v2",
            "wells": SMOKE_WELLS,
            "seed": 42,
            "generator_config": str(generator_config_path),
            "candidate_columns": DIRECT_CANDIDATE_COLUMNS,
            "unique_change_from_notebook": "在两个Numba单次PF内核中显式固定seed",
        },
    )
    write_json(
        artifact_dir / "leakage_tests.json",
        {
            "wells": invariance.to_dict(orient="records"),
            "all_fixed_seed_repeats_exact": bool(invariance["fixed_seed_repeat_exact"].all()),
            "all_hidden_tvt_and_surface_mutations_exact": bool(
                invariance["hidden_tvt_and_surface_mutation_exact"].all()
            ),
            "all_row_indices_exact": bool(invariance["row_index_exact"].all()),
        },
    )
    write_json(
        artifact_dir / "runtime.json",
        {
            "total_seconds": time.perf_counter() - total_start,
            "wells": well_summary.to_dict(orient="records"),
        },
    )
    print(f"重建 smoke 完成：{artifact_dir}", flush=True)
    print(family_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
