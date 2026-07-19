"""构建 P3 标签无关的固定 116 井 shadow holdout。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_shadow_holdout import (  # noqa: E402
    LEGAL_METADATA_COLUMNS,
    collect_legal_well_metadata,
    select_balanced_shadow,
)


def _resolve_workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _balance_report(metadata: pd.DataFrame, shadow: pd.DataFrame) -> dict[str, object]:
    """保存折覆盖和允许数值字段在全集/影子集中的均值，便于审计选择。"""

    numeric_columns = [
        "hidden_row_count",
        "md_span",
        "hidden_gr_observed_rate",
        "x_median",
        "y_median",
        "azimuth_deg",
    ]
    return {
        "metadata_wells": int(len(metadata)),
        "shadow_wells": int(len(shadow)),
        "fold_counts": {
            str(fold): int(count)
            for fold, count in shadow.groupby("fold")["well_id"].count().sort_index().items()
        },
        "legal_numeric_means": {
            column: {
                "all_wells": float(metadata[column].mean()),
                "shadow_wells": float(shadow[column].mean()),
            }
            for column in numeric_columns
        },
    }


def build_shadow_holdout(
    raw_train_dir: Path,
    registry_path: Path,
    output_dir: Path,
    number_of_shadow_wells: int,
    salt: str,
    config: dict[str, object],
) -> dict[str, object]:
    """写入 Task 2 规定的全部 shadow 产物，不接触任何训练标签。"""

    started = time.perf_counter()
    registry = pd.read_csv(registry_path, usecols=["well_id", "fold"])
    metadata = collect_legal_well_metadata(raw_train_dir, registry)
    shadow = select_balanced_shadow(metadata, number_of_shadow_wells, salt)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata.csv", index=False, lineterminator="\n")
    shadow.to_csv(output_dir / "shadow_holdout.csv", index=False, lineterminator="\n")
    report = _balance_report(metadata, shadow)
    (output_dir / "balance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = {
        "raw_train_dir": str(raw_train_dir.resolve()),
        "registry_path": str(registry_path.resolve()),
        "metadata_columns": list(LEGAL_METADATA_COLUMNS),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "conclusion.md").write_text(
        "事实：shadow 井仅由合法井级元数据确定，且覆盖五个冻结 fold。\n\n"
        "推断：该集合可作为 P3 诊断的隔离保留井。\n\n"
        "仍未验证：不评价该集合上的任何目标或模型误差。\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the label-free P3 shadow holdout")
    parser.add_argument(
        "--config",
        type=Path,
        default=CLEAN_ROOT / "configs" / "p3_shadow_holdout_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / "P3_shadow_holdout_v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = build_shadow_holdout(
        raw_train_dir=_resolve_workspace_path(str(config["raw_train_dir"])),
        registry_path=_resolve_workspace_path(str(config["fold_registry"])),
        output_dir=args.output_dir,
        number_of_shadow_wells=int(config["number_of_shadow_wells"]),
        salt=str(config["salt"]),
        config=config,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
