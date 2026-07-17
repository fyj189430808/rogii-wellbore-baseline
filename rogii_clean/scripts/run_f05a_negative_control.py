"""运行 F05a 的井内循环错位负对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import read_json, train_fold, write_json  # noqa: E402
from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS  # noqa: E402
from src.f05a_negative_control import circular_shift_candidates_within_well  # noqa: E402
from src.lgbm_data import file_sha256, load_and_validate_registry  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


SOURCE_EXPERIMENT = "F05a_direct_physical_candidates_v2"
CONTROL_EXPERIMENT = "F05a_direct_physical_candidates_v2_negative_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 F05a 井内错位负对照")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_artifact_dir = CLEAN_ROOT / "artifacts" / SOURCE_EXPERIMENT
    control_artifact_dir = CLEAN_ROOT / "artifacts" / CONTROL_EXPERIMENT
    experiment_config = read_json(source_artifact_dir / "config.json")
    model_params = read_json(CLEAN_ROOT / experiment_config["model_config"])["params"]
    model_features = [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]

    base_features = pd.read_parquet(CLEAN_ROOT / experiment_config["base_feature_cache"])
    candidate_features = pd.read_parquet(
        source_artifact_dir / "candidate_feature_cache.parquet"
    )
    shifted_candidates = circular_shift_candidates_within_well(
        candidate_features,
        DIRECT_CANDIDATE_COLUMNS,
    )
    feature_table = pd.concat(
        [
            base_features.reset_index(drop=True),
            shifted_candidates[DIRECT_CANDIDATE_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )

    registry_path = CLEAN_ROOT / experiment_config["fold_registry"]
    registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(experiment_config["expected_wells"]),
        expected_rows=int(experiment_config["expected_rows"]),
    )
    fingerprint_payload = {
        "control": "within_well_half_length_circular_shift",
        "source_config": experiment_config,
        "candidate_cache_meta": read_json(
            source_artifact_dir / "candidate_feature_cache.meta.json"
        ),
        "model_params": model_params,
        "fold_registry_sha256": file_sha256(registry_path),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "transform_sha256": file_sha256(
            CLEAN_ROOT / "src" / "f05a_negative_control.py"
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    control_artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        control_artifact_dir / "config.json",
        {
            "experiment_id": CONTROL_EXPERIMENT,
            "source_experiment": SOURCE_EXPERIMENT,
            "control": "每口井的24个候选列整体循环平移半个隐藏段",
            "preserved": "每口井、每列的数值分布和候选列之间的同时关系",
            "destroyed": "候选轨迹与当前隐藏行位置的对应关系",
            "fold": args.fold,
            "features": model_features,
        },
    )
    write_json(
        control_artifact_dir / "feature_list.json",
        {"feature_count": len(model_features), "features": model_features},
    )
    runtime = train_fold(
        feature_table,
        registry,
        args.fold,
        model_params,
        control_artifact_dir,
        fingerprint,
        model_features,
    )
    pd.DataFrame([runtime]).to_csv(
        control_artifact_dir / "metrics.csv",
        index=False,
    )
    write_json(control_artifact_dir / "runtime_summary.json", runtime)


if __name__ == "__main__":
    main()
