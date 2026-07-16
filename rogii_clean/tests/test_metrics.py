"""统一指标的最小数值测试。"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# 测试文件的父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 把干净项目根目录加入模块路径。
sys.path.insert(0, str(CLEAN_ROOT))

# 导入待测试的统一指标函数。
from src.metrics import (
    build_per_well_metrics,
    paired_well_bootstrap,
    rmse,
    summarize_per_well_metrics,
)


# 检查 RMSE、micro、macro 和 bootstrap 的方向。
def test_metrics_with_two_wells() -> None:
    """构造两口小井，验证统一评分公式。"""

    # 两口井、四行的固定预测表。
    prediction_df = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "fold": [0, 0, 1, 1],
            "target_tvt": [0.0, 2.0, 10.0, 14.0],
            "pred_tvt": [0.0, 1.0, 11.0, 13.0],
            "carry_tvt": [0.0, 0.0, 10.0, 10.0],
        }
    )

    # 模型四行误差是 0、1、-1、1，所以 micro RMSE 为 sqrt(3/4)。
    expected_micro_rmse = float(np.sqrt(3.0 / 4.0))

    # 直接 RMSE 函数应得到同一结果。
    assert np.isclose(
        rmse(prediction_df["target_tvt"], prediction_df["pred_tvt"]),
        expected_micro_rmse,
    )

    # 建立逐井指标。
    per_well_df = build_per_well_metrics(prediction_df)

    # 汇总完整指标。
    summary = summarize_per_well_metrics(per_well_df)

    # micro 必须符合逐行公式。
    assert np.isclose(summary["micro_rmse"], expected_micro_rmse)

    # 模型应优于这个人为构造的 carry 基线。
    assert summary["micro_rmse_delta_vs_baseline"] < 0.0

    # 固定 seed 的井级 bootstrap 应大概率判断模型更好。
    bootstrap = paired_well_bootstrap(per_well_df, n_resamples=200, seed=42)

    # 改善概率应高于一半。
    assert bootstrap["probability_better"] > 0.5
