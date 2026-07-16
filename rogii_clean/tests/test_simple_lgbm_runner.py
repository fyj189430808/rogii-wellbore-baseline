"""最简 LGBM runner 的两个核心数学测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 把 rogii_clean 加入模块路径，便于直接导入脚本中的纯函数。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_simple_lgbm_cv import fold_indices, restore_absolute_tvt


def test_fold_indices_separates_train_and_validation() -> None:
    """验证指定 fold 的行只进入验证集，不会同时进入训练集。"""

    feature_table = pd.DataFrame({"fold": [0, 0, 1, 1, 2]})
    train_indices, validation_indices = fold_indices(feature_table, validation_fold=1)

    assert train_indices.tolist() == [0, 1, 4]
    assert validation_indices.tolist() == [2, 3]
    assert set(train_indices).isdisjoint(set(validation_indices))


def test_restore_absolute_tvt_adds_visible_anchor() -> None:
    """验证模型输出的相对 TVT 会逐行加回本井可见末值。"""

    anchor = np.array([100.0, 100.0, 200.0])
    predicted_delta = np.array([1.5, -2.0, 3.0])

    prediction = restore_absolute_tvt(anchor, predicted_delta)

    np.testing.assert_allclose(prediction, [101.5, 98.0, 203.0])
