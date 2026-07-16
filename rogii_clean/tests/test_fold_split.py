"""固定 fold 的最小一致性测试。"""

from pathlib import Path
import sys

import pandas as pd


# 测试文件的父目录是 rogii_clean。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 把干净项目根加入模块路径。
sys.path.insert(0, str(CLEAN_ROOT))

# 导入 fold 构造函数。
from src.fold_split import build_fixed_fold_registry


# 工作区训练目录。
TRAIN_DIR = CLEAN_ROOT.parent / "input" / "data" / "raw" / "train"


# 验证固定 fold 的核心不变量。
def test_spatial_pad_fold_invariants() -> None:
    """重新构造 fold 并检查井、pad、折和评价行。"""

    # 使用冻结的 1000-unit 半径和五折。
    registry_df = build_fixed_fold_registry(TRAIN_DIR, radius=1000.0, n_splits=5)

    # 数据应有 773 口唯一井。
    assert registry_df["well_id"].nunique() == 773

    # 一井一行。
    assert len(registry_df) == 773

    # 总自然隐藏评价行必须固定。
    assert int(registry_df["hidden_rows"].sum()) == 3_783_989

    # 连通 pad 数必须固定。
    assert registry_df["pad_id"].nunique() == 290

    # 同一 pad 不能跨 fold。
    assert int(registry_df.groupby("pad_id")["fold"].nunique().max()) == 1

    # 必须正好五折。
    assert set(registry_df["fold"].unique()) == {0, 1, 2, 3, 4}

    # 每折隐藏行数必须与冻结设计一致。
    expected_rows = {
        0: 757_050,
        1: 756_990,
        2: 756_649,
        3: 757_061,
        4: 756_239,
    }

    # 计算实际每折隐藏行。
    actual_rows = (
        registry_df.groupby("fold")["hidden_rows"].sum().astype(int).to_dict()
    )

    # 精确相等才能通过。
    assert actual_rows == expected_rows
