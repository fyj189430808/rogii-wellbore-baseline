"""构造固定的 median-XY 1000-unit connected-pad 五折。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# 读取一口井的最小统计量，不把整口井长期保存在内存中。
def read_well_summary(horizontal_path: Path) -> dict:
    """输入一个训练 horizontal CSV，输出井级坐标和评价行统计，无文件副作用。"""

    # 只读取 fold 构造需要的三列，避免加载 TVT、GR 和地层标签。
    horizontal_df = pd.read_csv(horizontal_path, usecols=["X", "Y", "TVT_input"])

    # 从文件名提取八位井号，例如 000d7d20。
    well_id = horizontal_path.name.split("__", 1)[0]

    # 整条轨迹 X 的中位数作为该井的代表 X 坐标，单位沿用原始数据坐标单位。
    representative_x = float(np.nanmedian(horizontal_df["X"].to_numpy(dtype=np.float64)))

    # 整条轨迹 Y 的中位数作为该井的代表 Y 坐标，单位沿用原始数据坐标单位。
    representative_y = float(np.nanmedian(horizontal_df["Y"].to_numpy(dtype=np.float64)))

    # TVT_input 有值的行是可见前缀行。
    visible_rows = int(horizontal_df["TVT_input"].notna().sum())

    # TVT_input 为空的行是固定自然评价行。
    hidden_rows = int(horizontal_df["TVT_input"].isna().sum())

    # 总行数用于检查 visible_rows + hidden_rows 是否完整覆盖一口井。
    total_rows = int(len(horizontal_df))

    # 返回一行井级记录，后续一口井只会分配到一个 pad 和一个 fold。
    return {
        "well_id": well_id,
        "representative_x": representative_x,
        "representative_y": representative_y,
        "total_rows": total_rows,
        "visible_rows": visible_rows,
        "hidden_rows": hidden_rows,
    }


# 扫描全部训练井并建立稳定的井级表。
def build_well_summary_table(train_dir: Path) -> pd.DataFrame:
    """输入训练目录，输出按 well_id 排序的 773 行井级表。"""

    # 按文件名排序，保证不同机器上的遍历顺序一致。
    horizontal_paths = sorted(train_dir.glob("*__horizontal_well.csv"))

    # 没有找到训练文件时立即报错，避免生成空 fold。
    if not horizontal_paths:
        raise FileNotFoundError(f"没有找到训练 horizontal 文件：{train_dir}")

    # 逐井读取最小统计量，列表长度应等于训练井数量。
    summary_rows = [read_well_summary(path) for path in horizontal_paths]

    # 把井级字典转成表格，每行代表一口完整井。
    well_summary_df = pd.DataFrame(summary_rows)

    # 井号必须唯一，否则同一口井可能被错误拆到多个 fold。
    if well_summary_df["well_id"].duplicated().any():
        raise ValueError("训练目录中存在重复 well_id")

    # 每口井的可见行和隐藏行之和必须等于总行数。
    row_check = well_summary_df["visible_rows"] + well_summary_df["hidden_rows"]

    # 行数不一致说明 TVT_input 掩码或读取逻辑存在问题。
    if not row_check.equals(well_summary_df["total_rows"]):
        raise ValueError("visible_rows + hidden_rows 与 total_rows 不一致")

    # 代表坐标必须是有限值，否则空间连边没有定义。
    coordinate_values = well_summary_df[["representative_x", "representative_y"]].to_numpy(dtype=np.float64)

    # 任一坐标缺失都停止，而不是静默把井分成单独 pad。
    if not np.isfinite(coordinate_values).all():
        raise ValueError("存在非有限的代表 XY 坐标")

    # 返回按 well_id 排序并重置行号的稳定井级表。
    return well_summary_df.sort_values("well_id").reset_index(drop=True)


# 根据代表点距离构造 single-linkage 连通分量。
def add_connected_pad_ids(well_summary_df: pd.DataFrame, radius: float) -> pd.DataFrame:
    """输入井级表和连边半径，输出新增 pad_id 的井级表。"""

    # 复制输入，避免函数悄悄修改调用者持有的 DataFrame。
    output_df = well_summary_df.copy()

    # 坐标矩阵 shape 为 [井数量, 2]。
    coordinates = output_df[["representative_x", "representative_y"]].to_numpy(dtype=np.float64)

    # KDTree 用于高效寻找半径内井对，不需要构造 773×773 的完整距离矩阵。
    coordinate_tree = cKDTree(coordinates)

    # query_pairs 返回距离不超过 radius 的无向边；排序后保证 union 顺序稳定。
    nearby_pairs = sorted(coordinate_tree.query_pairs(r=float(radius)))

    # parent[i] 保存并查集父节点，初始时每口井是自己的连通分量。
    parent = np.arange(len(output_df), dtype=np.int64)

    # find_root 返回节点的最终根，并执行路径压缩。
    def find_root(index: int) -> int:
        # 沿父节点向上走到根节点。
        while parent[index] != index:
            # 路径压缩让后续查询直接跳过一层。
            parent[index] = parent[parent[index]]
            # 继续移动到新的父节点。
            index = int(parent[index])
        # 返回这个连通分量的根索引。
        return int(index)

    # union_pair 把一条空间近邻边的两个连通分量合并。
    def union_pair(left_index: int, right_index: int) -> None:
        # 找到左端点所在分量的根。
        left_root = find_root(left_index)
        # 找到右端点所在分量的根。
        right_root = find_root(right_index)
        # 两端已经连通时不需要重复合并。
        if left_root == right_root:
            return
        # 总是把较大根索引接到较小根索引，保证结果与边遍历细节无关。
        smaller_root = min(left_root, right_root)
        # 较大根索引将不再是根。
        larger_root = max(left_root, right_root)
        # 完成两个连通分量的合并。
        parent[larger_root] = smaller_root

    # 逐条加入 1000-unit 内的空间边。
    for left_index, right_index in nearby_pairs:
        # 合并这两个代表点所属的连通分量。
        union_pair(int(left_index), int(right_index))

    # 为每口井计算最终根索引，shape 为 [井数量]。
    roots = np.asarray([find_root(index) for index in range(len(output_df))], dtype=np.int64)

    # 按根索引收集同一个连通 pad 中的 well_id。
    root_to_wells: dict[int, list[str]] = {}

    # 同时遍历根和井号，建立稳定的 pad 成员表。
    for root, well_id in zip(roots, output_df["well_id"].astype(str), strict=True):
        # 第一次遇到根时创建空列表，然后加入当前井号。
        root_to_wells.setdefault(int(root), []).append(well_id)

    # pad_id 取该连通分量中字典序最小的井号。
    root_to_pad_id = {root: min(well_ids) for root, well_ids in root_to_wells.items()}

    # 将每口井的根索引映射成稳定、可读的 pad_id。
    output_df["pad_id"] = [root_to_pad_id[int(root)] for root in roots]

    # 返回带 pad_id 的井级表。
    return output_df


# 将完整 pad 确定性分配到五折，并优先平衡隐藏评价行。
def add_balanced_fold_ids(pad_df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """输入带 pad_id 的井级表，输出新增 fold 的固定注册表。"""

    # 聚合每个 pad 的隐藏行数和井数，pad 是不可再拆分的最小分组。
    pad_summary_df = (
        pad_df.groupby("pad_id", as_index=False)
        .agg(hidden_rows=("hidden_rows", "sum"), well_count=("well_id", "count"))
    )

    # 先放最大的 pad；相同大小时按井数和 pad_id 保证确定性。
    pad_summary_df = pad_summary_df.sort_values(
        ["hidden_rows", "well_count", "pad_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # 每个 fold 的状态依次记录隐藏行数、井数和 pad 数量。
    fold_states = [
        {"hidden_rows": 0, "well_count": 0, "pad_count": 0}
        for _ in range(int(n_splits))
    ]

    # 保存 pad_id 到 fold 的唯一映射。
    pad_to_fold: dict[str, int] = {}

    # 从最大 pad 到最小 pad 依次分配。
    for pad_row in pad_summary_df.itertuples(index=False):
        # 选择当前 (隐藏行, 井数, pad数, fold号) 字典序最小的 fold。
        selected_fold = min(
            range(int(n_splits)),
            key=lambda fold_id: (
                fold_states[fold_id]["hidden_rows"],
                fold_states[fold_id]["well_count"],
                fold_states[fold_id]["pad_count"],
                fold_id,
            ),
        )

        # 记录这个 pad 的固定 fold。
        pad_to_fold[str(pad_row.pad_id)] = int(selected_fold)

        # 把 pad 的隐藏行数累加到所选 fold。
        fold_states[selected_fold]["hidden_rows"] += int(pad_row.hidden_rows)

        # 把 pad 的井数累加到所选 fold。
        fold_states[selected_fold]["well_count"] += int(pad_row.well_count)

        # 所选 fold 的 pad 数量增加一。
        fold_states[selected_fold]["pad_count"] += 1

    # 复制输入，避免修改上一步的 pad 表。
    output_df = pad_df.copy()

    # 根据 pad_id 映射固定 fold。
    output_df["fold"] = output_df["pad_id"].map(pad_to_fold).astype(np.int64)

    # 同一个 pad 只能出现一个 fold。
    pad_fold_counts = output_df.groupby("pad_id")["fold"].nunique()

    # 任何 pad 被拆分都说明实现错误。
    if int(pad_fold_counts.max()) != 1:
        raise ValueError("同一个 spatial pad 被拆到了多个 fold")

    # fold 编号必须完整覆盖 0 到 n_splits-1。
    expected_folds = set(range(int(n_splits)))

    # 实际 fold 集合用于完整性检查。
    actual_folds = set(output_df["fold"].astype(int).unique().tolist())

    # 缺折或多出 fold 都停止。
    if actual_folds != expected_folds:
        raise ValueError(f"fold 集合错误：{actual_folds}")

    # 返回按 well_id 排序的固定注册表。
    return output_df.sort_values("well_id").reset_index(drop=True)


# 对外提供一个简单入口，按固定顺序完成读取、pad 和 fold。
def build_fixed_fold_registry(
    train_dir: Path,
    radius: float = 1000.0,
    n_splits: int = 5,
) -> pd.DataFrame:
    """输入训练目录和冻结参数，输出一井一行的固定 fold 注册表。"""

    # 第一步读取全部井的代表坐标和自然评价行数量。
    well_summary_df = build_well_summary_table(train_dir)

    # 第二步按代表点半径连通分量生成不可拆分的 spatial pad。
    pad_df = add_connected_pad_ids(well_summary_df, radius=radius)

    # 第三步按隐藏行和井数确定性平衡到固定五折。
    fold_registry_df = add_balanced_fold_ids(pad_df, n_splits=n_splits)

    # 返回最终注册表，不在核心函数中写文件。
    return fold_registry_df
