"""按冻结口径评分一份 OOF 预测文件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# 当前脚本的父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 把干净项目根目录加入模块搜索路径。
sys.path.insert(0, str(CLEAN_ROOT))

# 导入统一评分函数。
from src.metrics import (
    build_per_well_metrics,
    paired_well_bootstrap,
    summarize_by_fold,
    summarize_per_well_metrics,
)


# 读取 CSV 或 Parquet，不根据内容猜其他格式。
def read_prediction_file(path: Path) -> pd.DataFrame:
    """输入预测文件路径，输出 DataFrame。"""

    # Parquet 使用 pandas 的列式读取。
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    # CSV 使用标准逗号分隔读取。
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    # 其他后缀不允许静默处理。
    raise ValueError("预测文件必须是 .csv 或 .parquet")


# 解析统一评分参数。
def parse_args() -> argparse.Namespace:
    """返回命令行参数。"""

    # 创建命令行解析器。
    parser = argparse.ArgumentParser(description="统一评分 ROGII OOF 预测")

    # 预测文件必须由用户显式提供。
    parser.add_argument("--predictions", type=Path, required=True)

    # 真实 TVT 列默认命名为 target_tvt。
    parser.add_argument("--target-column", default="target_tvt")

    # 模型预测列默认命名为 pred_tvt。
    parser.add_argument("--prediction-column", default="pred_tvt")

    # carry-forward 基线列默认命名为 carry_tvt。
    parser.add_argument("--baseline-column", default="carry_tvt")

    # 默认把报告写到预测文件同目录。
    parser.add_argument("--output-dir", type=Path, default=None)

    # 返回解析后的参数。
    return parser.parse_args()


# 主函数只评分，不训练或修改预测。
def main() -> None:
    """读取预测、校验、评分并保存统一报告。"""

    # 读取命令行参数。
    args = parse_args()

    # 解析预测文件绝对路径。
    prediction_path = args.predictions.resolve()

    # 输出目录默认使用预测文件所在目录。
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else prediction_path.parent
    )

    # 创建报告目录，不删除已有文件。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取隐藏评价行预测表。
    prediction_df = read_prediction_file(prediction_path)

    # 构造一井一行指标。
    per_well_df = build_per_well_metrics(
        prediction_df=prediction_df,
        target_column=str(args.target_column),
        prediction_column=str(args.prediction_column),
        baseline_column=str(args.baseline_column),
    )

    # 汇总完整五折主指标。
    overall_summary = summarize_per_well_metrics(per_well_df)

    # 汇总每个 outer fold。
    fold_summary = summarize_by_fold(per_well_df)

    # 有基线时执行固定 2000 次井级配对 bootstrap。
    bootstrap_summary = (
        paired_well_bootstrap(per_well_df, n_resamples=2000, seed=42)
        if "baseline_sse" in per_well_df.columns
        else None
    )

    # 合并成一个 JSON 报告。
    metrics = {
        "prediction_file": str(prediction_path),
        "overall": overall_summary,
        "folds": fold_summary,
        "paired_well_bootstrap": bootstrap_summary,
    }

    # 固定逐井输出文件名。
    per_well_path = output_dir / "per_well.csv"

    # 保存逐井指标。
    per_well_df.to_csv(per_well_path, index=False, lineterminator="\n")

    # 固定汇总输出文件名。
    metrics_path = output_dir / "metrics.json"

    # 保存可读 JSON。
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # 打印主指标，便于运行日志核对。
    print("总体指标：", json.dumps(overall_summary, ensure_ascii=False, indent=2))

    # 打印每折指标。
    print("每折指标：", json.dumps(fold_summary, ensure_ascii=False, indent=2))

    # 打印逐井文件位置。
    print("逐井指标：", per_well_path)

    # 打印汇总文件位置。
    print("汇总指标：", metrics_path)


# 只有直接运行脚本时才评分。
if __name__ == "__main__":
    # 调用主函数。
    main()
