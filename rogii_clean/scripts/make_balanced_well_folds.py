"""生成二阶段按完整井分组、按隐藏评价行数平衡的固定五折。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


# 当前脚本的父目录是干净项目根目录 rogii_clean。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 原始比赛数据位于 rogii_clean 的上一级工作区中。
WORKSPACE_ROOT = CLEAN_ROOT.parent

# 允许脚本直接导入 rogii_clean/src 下的简单模块。
sys.path.insert(0, str(CLEAN_ROOT))

from src.fold_split import build_balanced_well_fold_registry  # noqa: E402


# 这个名称同时用于 CSV、metadata 和后续实验配置，避免二阶段 fold 混用。
REGISTRY_VERSION = "balanced_well_5fold_v1"


def sha256_file(path: Path) -> str:
    """读取文件原始字节并返回 SHA-256，用于冻结注册表内容。"""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fold_summary(registry_df: pd.DataFrame) -> list[dict[str, object]]:
    """把一井一行注册表汇总成每折井数和隐藏评价行数。"""

    summary_df = (
        registry_df.groupby("fold", as_index=False)
        .agg(
            wells=("well_id", "count"),
            groups=("pad_id", "nunique"),
            hidden_rows=("hidden_rows", "sum"),
        )
        .sort_values("fold")
        .reset_index(drop=True)
    )
    total_hidden_rows = int(summary_df["hidden_rows"].sum())
    summary_df["hidden_row_fraction"] = (
        summary_df["hidden_rows"] / total_hidden_rows
    )
    return summary_df.to_dict(orient="records")


def write_registry_artifacts(
    registry_df: pd.DataFrame,
    output_path: Path,
    train_dir: Path,
    n_splits: int,
) -> dict[str, object]:
    """写注册表 CSV 和同名 metadata JSON，并返回实际保存的 metadata。"""

    output_columns = [
        "well_id",
        "pad_id",
        "fold",
        "representative_x",
        "representative_y",
        "total_rows",
        "visible_rows",
        "hidden_rows",
    ]
    missing_columns = [
        column_name
        for column_name in output_columns
        if column_name not in registry_df.columns
    ]
    if missing_columns:
        raise ValueError(f"注册表缺少列：{missing_columns}")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry_df[output_columns].to_csv(
        output_path,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )

    metadata = {
        "config_version": REGISTRY_VERSION,
        "validation_unit": "complete_well",
        "grouping": "none",
        "pad_id_semantics": "well_id_compatibility_alias",
        "train_dir": str(train_dir.resolve()),
        "n_splits": int(n_splits),
        "wells": int(registry_df["well_id"].nunique()),
        "groups": int(registry_df["pad_id"].nunique()),
        "hidden_rows": int(registry_df["hidden_rows"].sum()),
        "fold_summary": build_fold_summary(registry_df),
        "registry_sha256": sha256_file(output_path),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析原始训练目录、输出路径和固定折数。"""

    parser = argparse.ArgumentParser(description="生成二阶段按井平衡五折")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=WORKSPACE_ROOT / "input" / "data" / "raw" / "train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / "folds" / f"{REGISTRY_VERSION}.csv",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """从原始井文件生成注册表，并打印人工核对所需的完整摘要。"""

    args = parse_args(argv)
    train_dir = args.train_dir.resolve()
    output_path = args.output.resolve()
    registry_df = build_balanced_well_fold_registry(
        train_dir,
        n_splits=int(args.n_splits),
    )
    metadata = write_registry_artifacts(
        registry_df,
        output_path=output_path,
        train_dir=train_dir,
        n_splits=int(args.n_splits),
    )

    print("训练目录：", train_dir)
    print("按井 fold 注册表：", output_path)
    print("fold 元数据：", output_path.with_suffix(".meta.json"))
    print(
        "总体：",
        {
            "wells": metadata["wells"],
            "groups": metadata["groups"],
            "hidden_rows": metadata["hidden_rows"],
        },
    )
    for fold_row in metadata["fold_summary"]:
        print("fold：", fold_row)
    print("registry_sha256：", metadata["registry_sha256"])


if __name__ == "__main__":
    main()

