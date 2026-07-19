from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


# clean 项目根目录和冻结运行器路径。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))
RUNNER_PATH = CLEAN_ROOT / "scripts" / "diagnose_p2_f01_continuous_gr_path.py"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_f01_continuous_gr_offset_path_v1.json"


def _load_runner():
    """按文件路径加载运行器，便于测试内部汇总和程序正对照。"""

    spec = importlib.util.spec_from_file_location("p2_f01_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 P2-F01 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_freezes_continuous_path_design() -> None:
    """确认首版没有静默换中心、尺度、网格或连续性约束。"""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["center_path"] == "last_visible_tvt_plus_pf_ancc_delta"
    assert config["block_width_ft"] == 50.0
    assert config["smoothing_widths_ft"] == [5.0, 11.0, 21.0, 51.0, 101.0]
    assert config["allowed_offset_changes_ft"] == [-4.0, -2.0, 0.0, 2.0, 4.0]
    assert config["maximum_change_acceleration_ft"] == 2.0


def test_program_controls_pass_before_real_data() -> None:
    """合成路径必须恢复到一个网格内，全部缺失时必须严格输出零 offset。"""

    runner = _load_runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    controls = runner.run_program_controls(config)

    assert controls["synthetic_path_rmse_ft"] <= 2.0
    assert controls["all_missing_max_abs_offset_ft"] == pytest.approx(0.0)


def test_summary_applies_all_preregistered_gates() -> None:
    """用手工构造的五折好结果检查全部诊断晋级条件。"""

    runner = _load_runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    per_well = pd.DataFrame(
        {
            "well_id": [f"w{fold}" for fold in range(5)],
            "fold": [0, 1, 2, 3, 4],
            "hidden_rows": [100] * 5,
            "sse_continuous": [100.0] * 5,
            "sse_negative": [400.0] * 5,
            "sse_independent": [400.0] * 5,
            "sse_pf": [225.0] * 5,
            "sse_carry": [625.0] * 5,
            "sse_p2b00": [144.0] * 5,
            "continuous_rmse": [1.0] * 5,
            "negative_rmse": [2.0] * 5,
            "independent_rmse": [2.0] * 5,
            "pf_rmse": [1.5] * 5,
            "carry_rmse": [2.5] * 5,
            "p2b00_rmse": [1.2] * 5,
            "true_offset_in_grid_rows": [100] * 5,
        }
    )
    prefix = {
        "rows": 500,
        "sse_injected_center": 5000.0,
        "sse_recovered": 500.0,
        "sse_shifted": 8000.0,
    }
    program_controls = {
        "synthetic_path_rmse_ft": 1.0,
        "all_missing_max_abs_offset_ft": 0.0,
    }

    summary = runner.build_summary(
        per_well_df=per_well,
        prefix_totals=prefix,
        program_controls=program_controls,
        config=config,
        wall_seconds=5.0,
    )

    assert summary["micro_rmse"]["continuous"] == pytest.approx(1.0)
    assert all(summary["checks"].values())
    assert summary["continuous_gr_path_supported"] is True
