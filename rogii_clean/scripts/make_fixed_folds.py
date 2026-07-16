"""生成并验证固定的 spatial-pad 五折注册表。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


# 当前脚本位于 rogii_clean/scripts，因此父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 工作区根目录是 rogii_clean 的上一层。
WORKSPACE_ROOT = CLEAN_ROOT.parent

# 把干净项目根目录加入模块路径，允许导入 src。
sys.path.insert(0, str(CLEAN_ROOT))

# 导入已经拆开的固定 fold 构造函数。
from src.fold_split import build_fixed_fold_registry


# 计算文件 SHA-256，用于确认 fold 注册表没有被静默改写。
def sha256_file(path: Path) -> str:
    """输入文件路径，输出十六进制 SHA-256，无文件副作用。"""

    # 创建 SHA-256 累加器。
    digest = hashlib.sha256()

    # 以二进制只读模式打开文件。
    with path.open("rb") as file_handle:
        # 分块读取，避免一次把文件全部放入内存。
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            # 把当前数据块加入哈希。
            digest.update(chunk)

    # 返回稳定的十六进制字符串。
    return digest.hexdigest()


# 把注册表聚合成便于人工核对的 fold 摘要。
def build_fold_summary(registry_df: pd.DataFrame) -> list[dict]:
    """输入井级注册表，输出每折井数、pad数和隐藏行数。"""

    # 按 fold 聚合，保持 fold 0 到 4 的升序。
    summary_df = (
        registry_df.groupby("fold", as_index=False)
        .agg(
            wells=("well_id", "count"),
            pads=("pad_id", "nunique"),
            hidden_rows=("hidden_rows", "sum"),
        )
        .sort_values("fold")
    )

    # 总隐藏行用于计算每折评价行占比。
    total_hidden_rows = int(summary_df["hidden_rows"].sum())

    # 行占比只用于报告，不参与 fold 分配。
    summary_df["hidden_row_fraction"] = summary_df["hidden_rows"] / total_hidden_rows

    # 转成普通字典，便于保存 JSON。
    return summary_df.to_dict(orient="records")


# 解析命令行参数，默认值全部指向当前工作区。
def parse_args() -> argparse.Namespace:
    """返回命令行参数，不读取数据。"""

    # 创建简洁的命令行解析器。
    parser = argparse.ArgumentParser(description="生成固定 spatial-pad 五折")

    # 默认读取工作区原始训练目录。
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=WORKSPACE_ROOT / "input" / "data" / "raw" / "train",
    )

    # 默认输出到干净项目的 folds artifact 目录。
    parser.add_argument(
        "--output",
        type=Path,
        default=CLEAN_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv",
    )

    # 连边半径冻结为 1000 个原始 XY 单位。
    parser.add_argument("--radius", type=float, default=1000.0)

    # 主验证冻结为五折。
    parser.add_argument("--n-splits", type=int, default=5)

    # 返回解析后的参数。
    return parser.parse_args()


# 主函数只负责调用纯函数、写 artifact 和打印检查结果。
def main() -> None:
    """生成 CSV 和 metadata JSON；不训练模型。"""

    # 读取命令行参数。
    args = parse_args()

    # 把训练目录解析成绝对路径，便于日志核对。
    train_dir = args.train_dir.resolve()

    # 把输出路径解析成绝对路径。
    output_path = args.output.resolve()

    # 构造一井一行的固定 fold 注册表。
    registry_df = build_fixed_fold_registry(
        train_dir=train_dir,
        radius=float(args.radius),
        n_splits=int(args.n_splits),
    )

    # 创建 artifact 目录；不会删除或覆盖其他目录。
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 固定列顺序，避免 pandas 版本变化导致无意义差异。
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

    # 以稳定小数格式写 CSV；同样输入应得到同样字节。
    registry_df[output_columns].to_csv(
        output_path,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )

    # metadata 与 CSV 同名，只把后缀替换为 .meta.json。
    metadata_path = output_path.with_suffix(".meta.json")

    # 汇总固定 fold 的关键事实和文件指纹。
    metadata = {
        "config_version": "spatial_pad_1000_v1",
        "train_dir": str(train_dir),
        "radius_xy_units": float(args.radius),
        "n_splits": int(args.n_splits),
        "wells": int(registry_df["well_id"].nunique()),
        "pads": int(registry_df["pad_id"].nunique()),
        "hidden_rows": int(registry_df["hidden_rows"].sum()),
        "fold_summary": build_fold_summary(registry_df),
        "registry_sha256": sha256_file(output_path),
    }

    # 用 UTF-8 和排序键写 metadata，便于版本比较。
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 打印输入目录，防止误用另一份数据。
    print("训练目录：", train_dir)

    # 打印固定注册表路径。
    print("fold 注册表：", output_path)

    # 打印 metadata 路径。
    print("fold 元数据：", metadata_path)

    # 打印总体规模。
    print(
        "总体：",
        {
            "wells": metadata["wells"],
            "pads": metadata["pads"],
            "hidden_rows": metadata["hidden_rows"],
        },
    )

    # 逐折打印井数、pad 数和评价行数。
    for fold_row in metadata["fold_summary"]:
        # 每一行日志对应一个固定 outer fold。
        print("fold：", fold_row)

    # 打印 SHA-256，后续读取缓存前必须核对。
    print("registry_sha256：", metadata["registry_sha256"])


# 只有直接运行脚本时才生成文件。
if __name__ == "__main__":
    # 调用主函数。
    main()
