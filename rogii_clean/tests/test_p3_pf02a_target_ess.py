"""P3-PF02a：固定目标 ESS 离散选路的最小测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.p3_pf02a_target_ess import build_target_ess_path, choose_scale_for_target_ess


def test_choose_scale_uses_only_nearest_ess_and_prefers_safer_larger_scale_on_tie() -> None:
    ess_by_scale = {3.0: 72.0, 5.0: 88.0, 8.0: 104.0, 12.0: 120.0}

    selected_scale, selected_ess = choose_scale_for_target_ess(
        ess_by_scale=ess_by_scale,
        target_ess=96.0,
    )

    assert selected_scale == 8.0
    assert selected_ess == 104.0


def test_build_target_ess_path_copies_the_selected_frozen_path() -> None:
    frozen_paths = pd.DataFrame(
        {
            "well_id": ["well_a", "well_a"],
            "row_index": [10, 11],
            "last_visible_tvt": [1000.0, 1000.0],
            "pf128_scale_3_delta": [1.0, 2.0],
            "pf128_scale_5_delta": [3.0, 4.0],
            "pf128_scale_8_delta": [5.0, 6.0],
            "pf128_scale_12_delta": [7.0, 8.0],
        }
    )
    ess_by_scale = {3.0: 40.0, 5.0: 80.0, 8.0: 97.0, 12.0: 110.0}

    result = build_target_ess_path(
        frozen_paths=frozen_paths,
        ess_by_scale=ess_by_scale,
        target_ess=96.0,
    )

    np.testing.assert_array_equal(result["row_index"].to_numpy(), np.array([10, 11]))
    np.testing.assert_allclose(result["pf128_target_ess_delta"], np.array([5.0, 6.0]))
    assert result["selected_scale"].unique().tolist() == [8.0]
    assert result["selected_ess"].unique().tolist() == [97.0]

