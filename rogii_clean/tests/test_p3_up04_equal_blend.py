import numpy as np
import json

from scripts.run_p3_up04_global_local_equal_blend import read_locked_candidate
from src.p3_up04_equal_blend import equal_blend_paths


def test_equal_blend_is_exact_arithmetic_mean() -> None:
    global_path = np.array([100.0, 101.0, 105.0, 110.0])
    local_path = np.array([102.0, 99.0, 107.0, 106.0])

    blended = equal_blend_paths(global_path, local_path)

    np.testing.assert_allclose(blended, np.array([101.0, 100.0, 106.0, 108.0]))


def test_equal_blend_rejects_mismatched_rows() -> None:
    try:
        equal_blend_paths(np.array([1.0, 2.0]), np.array([3.0]))
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError("两条路径行数不同时必须报错")


def test_locked_candidate_can_come_from_dedicated_selection_record(tmp_path) -> None:
    """UP01 主指标没有 selected 字段时，应读取当时独立保存的锁定记录。"""

    metrics_path = tmp_path / "metrics.json"
    provenance_path = tmp_path / "selection_provenance.json"
    metrics_path.write_text(json.dumps({"experiment_id": "UP01"}), encoding="utf-8")
    provenance_path.write_text(
        json.dumps({"selected_candidate": "degree2_blend50", "selection_folds": [1, 2]}),
        encoding="utf-8",
    )

    selected, folds = read_locked_candidate(
        metrics_path,
        selected_field="selected_by_folds12",
        selection_provenance_path=provenance_path,
    )

    assert selected == "degree2_blend50"
    assert folds == [1, 2]
