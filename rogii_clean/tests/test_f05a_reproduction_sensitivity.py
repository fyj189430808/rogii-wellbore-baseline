import numpy as np
import pandas as pd

from scripts.evaluate_f05a_reproduction_smoke_predictions import _keys_equal


def test_keys_equal_ignores_integer_storage_width():
    left = pd.DataFrame(
        {"well_id": ["a", "a"], "row_index": np.array([3, 4], dtype=np.int32)}
    )
    right = pd.DataFrame(
        {"well_id": ["a", "a"], "row_index": np.array([3, 4], dtype=np.int64)}
    )

    assert _keys_equal(left, right)
