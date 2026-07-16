"""防止高分 Notebook 意外恢复同井标签通道。"""

import json
from pathlib import Path


# 测试文件上两级是工作区根目录。
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# 高分 Notebook 的固定路径。
NOTEBOOK_PATH = WORKSPACE_ROOT / "rogii-dual-track-prefix-calibrated-geosteering.ipynb"


# 检查所有关键开关和条件仍保持 private-safe。
def test_notebook_same_well_paths_are_disabled() -> None:
    """解析 Notebook 源码并检查同井覆盖、候选和 KNN 自排除。"""

    # 按 UTF-8 读取 Notebook JSON。
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    # 将指定单元格源码拼成普通字符串。
    cell_6 = "".join(notebook["cells"][6]["source"])

    # cell 6 必须关闭 cell 26 的同井 physical。
    assert "RUN_SAME_WELL_CONTACT_OVERRIDE = False" in cell_6

    # cell 6 必须关闭 overlap 探测。
    assert "RUN_OVERLAP_DRY_RUN_PROBE = False" in cell_6

    # cell 6 必须关闭 cell 48 的 guarded override。
    assert "RUN_GUARDED_OVERLAP_OVERRIDE = False" in cell_6

    # 读取 cell 14 的第一条特征轨道。
    cell_14 = "".join(notebook["cells"][14]["source"])

    # 第一条轨道必须始终把查询 well_id 传给空间 imputer。
    assert "swid=wid" in cell_14

    # 读取 cell 26 的 physical/selector 分支。
    cell_26 = "".join(notebook["cells"][26]["source"])

    # 同井 physical 必须受默认 False 的显式开关保护。
    assert "RUN_SAME_WELL_CONTACT_OVERRIDE', False" in cell_26

    # 读取 cell 40 的第二条特征轨道。
    cell_40 = "".join(notebook["cells"][40]["source"])

    # 第二条轨道也必须始终排除查询井。
    assert "swid = wid" in cell_40

    # 读取 cell 48 的 guarded override。
    cell_48 = "".join(notebook["cells"][48]["source"])

    # 即使 cell 6 未执行，cell 48 的默认值也必须是 False。
    assert "RUN_GUARDED_OVERLAP_OVERRIDE', False" in cell_48

    # 读取 cell 50 的可见前缀候选。
    cell_50 = "".join(notebook["cells"][50]["source"])

    # contact 候选只能在显式打开旧开关时加入。
    assert "if _GOLD_CONTACT_OVERRIDE:" in cell_50

    # formation 候选查询必须排除当前井。
    assert "fi.impute(xy, self_wid=wid)" in cell_50

    # dense 候选查询也必须排除当前井。
    assert "di.impute(xy, self_wid=wid)" in cell_50
