"""P2-M01 runner 的冻结配置、合法 marker 表和晋级门槛测试。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


# 允许测试直接导入 scripts 下的诊断 runner。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.diagnose_p2_m01_marker_normalized_coordinate import (  # noqa: E402
    evaluate_success_checks,
    run_program_controls,
    select_validation_registry,
    validate_config,
    validate_legal_marker_table,
)


def _load_config() -> dict:
    """读取 P2-M01 已预注册的唯一配置。"""

    path = CLEAN_ROOT / "configs" / "p2_m01_marker_normalized_coordinate_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_config_rejects_changing_template_or_marker_definition() -> None:
    """runner 必须拒绝结果出现后更换指纹长度或增加稀有 marker。"""

    config = _load_config()
    validate_config(config)

    changed_tail = copy.deepcopy(config)
    changed_tail["template_tail_rows"] = 100
    with pytest.raises(ValueError, match="template_tail_rows"):
        validate_config(changed_tail)

    changed_markers = copy.deepcopy(config)
    changed_markers["marker_names"].append("MNSS")
    with pytest.raises(ValueError, match="marker_names"):
        validate_config(changed_markers)


def test_program_controls_cover_crop_recovery_q_roundtrip_and_self_exclusion() -> None:
    """真实 fold 运行前，模板裁剪、marker 转移和 q 公式必须先通过。"""

    config = _load_config()
    controls = run_program_controls(config)
    conditions = config["success_conditions"]

    assert controls["cropped_template_fingerprint_equal"] is True
    assert controls["query_well_excluded_from_donors"] is True
    assert controls["maximum_crop_recovery_marker_error_ft"] <= conditions[
        "maximum_crop_recovery_marker_error_ft"
    ]
    assert controls["maximum_q_roundtrip_error"] <= conditions[
        "maximum_q_roundtrip_error"
    ]


def test_validation_registry_is_fixed_to_fold_zero() -> None:
    """M01 v1 只能读取预注册 fold 0，不能看结果后换折。"""

    config = _load_config()
    registry = pd.DataFrame(
        {
            "well_id": ["a", "b", "c", "d"],
            "fold": [0, 1, 0, 4],
        }
    )
    selected = select_validation_registry(registry, config)
    assert selected["well_id"].tolist() == ["a", "c"]


def test_legal_marker_table_rejects_oracle_columns_and_wrong_fold() -> None:
    """合法 marker 缓存不得混入验证 Geology、隐藏 TVT 或 oracle 排名。"""

    legal = pd.DataFrame(
        {
            "well_id": ["a"],
            "fold": [0],
            "template_fingerprint": ["hash"],
            "marker_name": ["ANCC"],
            "marker_tvt": [11000.0],
            "marker_usable": [True],
            "marker_ambiguous": [False],
            "donor_well_count": [3],
            "donor_value_count": [3],
            "donor_marker_range_ft": [0.0],
        }
    )
    validate_legal_marker_table(legal, expected_fold=0)

    leaked = legal.copy()
    leaked["true_zone"] = 2
    with pytest.raises(ValueError, match="禁止列"):
        validate_legal_marker_table(leaked, expected_fold=0)

    wrong_fold = legal.copy()
    wrong_fold["fold"] = 1
    with pytest.raises(ValueError, match="fold"):
        validate_legal_marker_table(wrong_fold, expected_fold=0)


def test_all_preregistered_rank_gates_are_required() -> None:
    """缺少任一覆盖、marker、oracle 或合法排名门槛时都不能进入 q-state 路径。"""

    config = _load_config()
    passing_metrics = {
        "template_match_well_coverage": 0.95,
        "auditable_block_coverage": 0.95,
        "withheld_marker_p90_error_ft": 0.0,
        "ambiguous_well_fraction": 0.0,
        "cross_zone_fraction_among_wrong_top1": 0.60,
        "oracle_zone_normalized_rank_improvement": 0.12,
        "oracle_zone_top5_gain": 0.12,
        "oracle_zone_bootstrap_ci_upper": -0.01,
        "pf_true_zone_agreement": 0.85,
        "legal_zone_normalized_rank_improvement": 0.07,
        "legal_zone_bootstrap_ci_upper": -0.01,
        "legal_gain_vs_wrong_marker_gate": 0.04,
        "legal_gain_vs_random_count_gate": 0.04,
        "maximum_q_roundtrip_error": 0.0,
        "maximum_crop_recovery_marker_error_ft": 0.0,
    }
    checks = evaluate_success_checks(passing_metrics, config)
    assert checks["marker_zone_rank_supported"] is True

    failing = dict(passing_metrics)
    failing["pf_true_zone_agreement"] = 0.79
    failed_checks = evaluate_success_checks(failing, config)
    assert failed_checks["marker_zone_rank_supported"] is False
    assert failed_checks["pf_true_zone_agreement_pass"] is False

