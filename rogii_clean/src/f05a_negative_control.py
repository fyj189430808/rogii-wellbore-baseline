"""F05a 的井内错位负对照。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def circular_shift_candidates_within_well(
    feature_table: pd.DataFrame,
    candidate_columns: list[str],
) -> pd.DataFrame:
    """把候选轨迹在每口井内部循环平移半段，保留分布但破坏逐行对应。"""

    shifted_table = feature_table.copy()
    for _, row_indices in shifted_table.groupby("well_id", sort=False).groups.items():
        row_indices = np.asarray(row_indices)
        row_count = len(row_indices)
        if row_count <= 1:
            continue
        shift_rows = max(1, row_count // 2)
        original_values = shifted_table.loc[row_indices, candidate_columns].to_numpy()
        shifted_values = np.roll(original_values, shift=shift_rows, axis=0)
        shifted_table.loc[row_indices, candidate_columns] = shifted_values
    return shifted_table
