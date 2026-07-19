"""P2-F01b runner 的冻结配置、阶段选择和判定门槛测试。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


# 允许测试直接导入 scripts 下的新诊断脚本。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p2_f01b_path_domain_smoothing import (  # noqa: E402
    evaluate_stage_checks,
    run_program_controls,
    select_registry,
    validate_config,
)


def _load_config() -> dict:
    """读取已经预注册的 F01b 唯一配置。"""

    config_path = CLEAN_ROOT / "configs" / "p2_f01b_path_domain_smoothing_v1.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def test_config_rejects_a_silent_return_to_tvt_domain_smoothing() -> None:
    """runner 必须拒绝把唯一修改偷偷换回旧的 TVT 轴平滑。"""

    config = _load_config()
    validate_config(config)

    changed_config = copy.deepcopy(config)
    changed_config["score_domain"] = "smooth_typewell_on_tvt_before_sampling"
    with pytest.raises(ValueError, match="score_domain"):
        validate_config(changed_config)


def test_program_controls_cover_synthetic_translation_and_missing_gr() -> None:
    """真实井运行前，路径、平移等变和全缺失三项程序检查必须全部通过。"""

    config = _load_config()
    controls = run_program_controls(config)
    conditions = config["success_conditions"]

    assert controls["synthetic_path_rmse_ft"] <= conditions[
        "maximum_synthetic_path_rmse_ft"
    ]
    assert controls["translation_max_ncc_difference"] <= conditions[
        "maximum_translation_ncc_difference"
    ]
    assert controls["translation_finite_mask_equal"] is True
    assert controls["translation_pair_counts_equal"] is True
    assert controls["all_missing_max_abs_offset_ft"] <= conditions[
        "maximum_all_missing_offset_ft"
    ]


def test_registry_selection_is_fixed_for_smoke_fold0_and_all() -> None:
    """三个运行阶段必须使用配置指定的井，不能根据结果临时换井。"""

    config = _load_config()
    registry = pd.DataFrame(
        {
            "well_id": [
                "000d7d20",
                "00bbac68",
                "00e12e8b",
                "fold0_extra",
                "fold3_extra",
            ],
            "fold": [0, 1, 2, 0, 3],
        }
    )

    smoke = select_registry(registry, "smoke", config)
    fold0 = select_registry(registry, "fold0", config)
    all_wells = select_registry(registry, "all", config)

    assert smoke["well_id"].tolist() == config["smoke_well_ids"]
    assert fold0["well_id"].tolist() == ["000d7d20", "fold0_extra"]
    assert all_wells["well_id"].tolist() == registry["well_id"].tolist()


def test_smoke_gate_does_not_require_four_folds() -> None:
    """三井 smoke 只卡程序和前缀信号，不能套用不可能达到的 4/5 折门槛。"""

    config = _load_config()
    metrics = {
        "synthetic_path_rmse_ft": 1.0,
        "translation_max_ncc_difference": 0.0,
        "translation_finite_mask_equal": True,
        "translation_pair_counts_equal": True,
        "all_missing_max_abs_offset_ft": 0.0,
        "prefix_improvement_vs_shifted_gr_ft": 2.1,
        "improvement_vs_pf_ft": -10.0,
        "improvement_vs_shifted_gr_ft": -10.0,
        "improvement_vs_independent_blocks_ft": -10.0,
        "well_win_rate_vs_pf": 0.0,
        "p90_degradation_vs_pf_ft": 20.0,
        "true_offset_grid_coverage": 0.0,
        "folds_better_than_pf": 0,
        "maximum_single_fold_degradation_vs_pf_ft": 20.0,
    }

    checks = evaluate_stage_checks("smoke", metrics, config)
    assert checks["stage_supported"] is True
    assert "full_fold_consistency_pass" not in checks


def test_fold0_and_full_apply_their_preregistered_performance_gates() -> None:
    """fold0 与全量阶段必须分别执行实验卡中已经写死的性能门槛。"""

    config = _load_config()
    passing_metrics = {
        "synthetic_path_rmse_ft": 1.0,
        "translation_max_ncc_difference": 0.0,
        "translation_finite_mask_equal": True,
        "translation_pair_counts_equal": True,
        "all_missing_max_abs_offset_ft": 0.0,
        "prefix_improvement_vs_shifted_gr_ft": 2.1,
        "improvement_vs_pf_ft": 0.6,
        "improvement_vs_shifted_gr_ft": 0.6,
        "improvement_vs_independent_blocks_ft": 0.6,
        "well_win_rate_vs_pf": 0.56,
        "p90_degradation_vs_pf_ft": 0.4,
        "true_offset_grid_coverage": 0.91,
        "folds_better_than_pf": 4,
        "maximum_single_fold_degradation_vs_pf_ft": 0.4,
    }

    assert evaluate_stage_checks("fold0", passing_metrics, config)[
        "stage_supported"
    ] is True
    assert evaluate_stage_checks("all", passing_metrics, config)[
        "stage_supported"
    ] is True

    failing_full = dict(passing_metrics)
    failing_full["folds_better_than_pf"] = 3
    full_checks = evaluate_stage_checks("all", failing_full, config)
    assert full_checks["stage_supported"] is False
    assert full_checks["full_fold_consistency_pass"] is False

