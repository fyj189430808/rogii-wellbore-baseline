"""轻量入口：只补齐确定性 F05a 的结果审计，不训练或重新推理。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.finalize_f05a_deterministic_audit import (  # noqa: F401
    run_audit,
    validate_saved_model_feature_names,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / "F05a_deterministic_candidates_v1",
    )
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        default=(
            CLEAN_ROOT / "artifacts" / "F05a_deterministic_candidate_cache_v1"
        ),
    )
    parser.add_argument(
        "--b00-dir",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / "B00_simple_lgbm_v1",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    summary = run_audit(
        artifact_dir=args.artifact_dir,
        candidate_cache_dir=args.candidate_cache_dir,
        b00_dir=args.b00_dir,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(
        f"F05a={summary['micro_rmse']:.6f}, "
        f"B00={summary['b00_micro_rmse']:.6f}, "
        f"差值={summary['delta_vs_b00']:+.6f} ft"
    )


if __name__ == "__main__":
    main()
