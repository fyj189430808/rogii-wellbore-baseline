"""P2-M01 的模板指纹、marker 转移和归一化坐标基础函数。

这个文件只实现确定性的数据变换，不读取隐藏 TVT，也不训练任何模型。
函数刻意保持简单，便于在诊断脚本和单元测试中逐步核对数据血缘。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def typewell_template_fingerprint(
    typewell: pd.DataFrame,
    tail_rows: int = 500,
) -> str:
    """返回 Typewell 深部尾段 ``(TVT, GR)`` 精确数值的 SHA-256。

    输入是一个 Typewell 表。函数先删除 TVT 或 GR 不是有限数值的行，
    再按 TVT 从浅到深稳定排序，最后只取最深的 ``tail_rows`` 行。哈希
    的原始内容是按 ``TVT, GR`` 列顺序连续存放的 float64 字节。

    这样只裁掉浅部行的两个 Typewell 会得到相同指纹；深部任意一个真实
    数值发生变化时，指纹也会变化。
    """

    # 指纹定义只允许正整数尾段长度，避免 ``-500`` 等含义不清的切片。
    if isinstance(tail_rows, bool) or int(tail_rows) != tail_rows or tail_rows <= 0:
        raise ValueError("tail_rows 必须是正整数")

    # 明确检查原始字段，避免列名拼错后生成一个看似正常的空指纹。
    required_columns = {"TVT", "GR"}
    missing_columns = required_columns.difference(typewell.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Typewell 缺少必要列：{missing_text}")

    # 非数值文本按缺失处理；转换后的两个数组形状均为 [原始行数]。
    tvt_values = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(
        dtype=np.float64,
    )
    gr_values = pd.to_numeric(typewell["GR"], errors="coerce").to_numpy(
        dtype=np.float64,
    )

    # 只有 TVT 和 GR 同时有限的行才属于模板指纹，NaN/inf 不参与匹配。
    finite_mask = np.isfinite(tvt_values) & np.isfinite(gr_values)
    if not np.any(finite_mask):
        raise ValueError("Typewell 没有可用于指纹的有限 TVT-GR 行")

    finite_tvt = tvt_values[finite_mask]
    finite_gr = gr_values[finite_mask]

    # mergesort 是稳定排序；如果存在相同 TVT，原文件中的先后顺序保持不变。
    sorted_positions = np.argsort(finite_tvt, kind="mergesort")
    sorted_tvt = finite_tvt[sorted_positions]
    sorted_gr = finite_gr[sorted_positions]

    # 取排序后的深部尾段。行数不足 500 时使用全部有限行。
    number_to_keep = min(int(tail_rows), len(sorted_tvt))
    tail_tvt = sorted_tvt[-number_to_keep:]
    tail_gr = sorted_gr[-number_to_keep:]

    # 列顺序固定为 TVT、GR，二维数组形状为 [尾段行数, 2]。
    fingerprint_rows = np.column_stack((tail_tvt, tail_gr))

    # 显式使用小端 float64，使同一份数据在不同机器字节序上仍有相同指纹。
    exact_float64_rows = np.ascontiguousarray(fingerprint_rows, dtype="<f8")
    return hashlib.sha256(exact_float64_rows.tobytes(order="C")).hexdigest()


def aggregate_outer_train_markers(
    query_well_id: str,
    query_fold: int,
    query_template_fingerprint: str,
    donor_markers: pd.DataFrame,
    marker_names: Sequence[str],
    maximum_marker_range_ft: float,
) -> pd.DataFrame:
    """仅用合法 outer-train 同模板井，为查询井汇总 marker TVT。

    查询井自身、与查询井处于同一 fold 的井、不同模板井以及没有观察到
    真实边界的 marker 都会先被排除。每个 donor 井对同一 marker 最多贡献
    一个值，之后取 donor 中位数。donor 极差超过允许值时，该 marker 被
    标成歧义并禁止后续正式使用。

    返回表始终包含 ``marker_names`` 中的全部 marker；没有 donor 的 marker
    会保留一行，但 ``marker_tvt`` 为 NaN 且 ``marker_usable`` 为 False。
    """

    # 极差阈值的单位是 ft，必须是非负有限数值。
    if not np.isfinite(maximum_marker_range_ft) or maximum_marker_range_ft < 0:
        raise ValueError("maximum_marker_range_ft 必须是非负有限数值")

    # 重复 marker 名会让输出行含义不唯一，因此直接拒绝。
    ordered_marker_names = [str(name) for name in marker_names]
    if len(set(ordered_marker_names)) != len(ordered_marker_names):
        raise ValueError("marker_names 不能包含重复名称")

    # 这些字段共同保证 donor 的井、fold、模板和边界来源都能被审计。
    required_columns = {
        "well_id",
        "fold",
        "template_fingerprint",
        "marker_name",
        "marker_tvt",
        "marker_boundary_observed",
    }
    missing_columns = required_columns.difference(donor_markers.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"donor_markers 缺少必要列：{missing_text}")

    # 四个条件必须同时满足：不是查询井、不是查询 fold、模板相同、边界可见。
    legal_donor_mask = donor_markers["well_id"].astype(str).ne(str(query_well_id))
    legal_donor_mask &= donor_markers["fold"].ne(query_fold)
    legal_donor_mask &= donor_markers["template_fingerprint"].astype(str).eq(
        str(query_template_fingerprint),
    )
    legal_donor_mask &= donor_markers["marker_boundary_observed"].eq(True)

    # 只复制后续计算需要的列，避免 oracle 或其他无关列进入合法结果。
    legal_donors = donor_markers.loc[
        legal_donor_mask,
        ["well_id", "marker_name", "marker_tvt"],
    ].copy()
    legal_donors["marker_name"] = legal_donors["marker_name"].astype(str)
    legal_donors["marker_tvt"] = pd.to_numeric(
        legal_donors["marker_tvt"],
        errors="coerce",
    )
    legal_donors = legal_donors.loc[
        np.isfinite(legal_donors["marker_tvt"].to_numpy(dtype=np.float64))
    ]

    output_rows: list[dict[str, Any]] = []

    # 按预先冻结的 marker 顺序逐个汇总，保证输出顺序可复现。
    for marker_name in ordered_marker_names:
        marker_rows = legal_donors.loc[legal_donors["marker_name"].eq(marker_name)]
        raw_value_count = int(len(marker_rows))

        if raw_value_count > 0:
            # 同一 donor 若意外出现重复行，先在井内取中位数，不能让它获得额外权重。
            one_value_per_donor = marker_rows.groupby(
                "well_id",
                sort=True,
            )["marker_tvt"].median()
            donor_values = one_value_per_donor.to_numpy(dtype=np.float64)
            donor_well_count = int(len(donor_values))
            marker_tvt = float(np.median(donor_values))
            marker_range_ft = float(np.max(donor_values) - np.min(donor_values))
            marker_ambiguous = bool(marker_range_ft > maximum_marker_range_ft)
            marker_usable = not marker_ambiguous
        else:
            # 没有合法 donor 时保留占位行，便于调用方识别具体缺失的 marker。
            donor_well_count = 0
            marker_tvt = float("nan")
            marker_range_ft = float("nan")
            marker_ambiguous = False
            marker_usable = False

        output_rows.append(
            {
                "well_id": str(query_well_id),
                "fold": int(query_fold),
                "template_fingerprint": str(query_template_fingerprint),
                "marker_name": marker_name,
                "marker_tvt": marker_tvt,
                "marker_usable": bool(marker_usable),
                "marker_ambiguous": bool(marker_ambiguous),
                "donor_well_count": donor_well_count,
                "donor_value_count": raw_value_count,
                "donor_marker_range_ft": marker_range_ft,
            }
        )

    # 即使 marker_names 为空也返回字段完整的空表，避免下游列访问失败。
    output_columns = [
        "well_id",
        "fold",
        "template_fingerprint",
        "marker_name",
        "marker_tvt",
        "marker_usable",
        "marker_ambiguous",
        "donor_well_count",
        "donor_value_count",
        "donor_marker_range_ft",
    ]
    return pd.DataFrame(output_rows, columns=output_columns)


def _validate_marker_positions(marker_positions: np.ndarray) -> np.ndarray:
    """检查 marker 是一维、有限且严格递增，并返回 float64 数组。"""

    markers = np.asarray(marker_positions, dtype=np.float64)
    if markers.ndim != 1:
        raise ValueError("marker_positions 必须是一维数组")
    if len(markers) < 2:
        raise ValueError("至少需要两个 marker 才能定义区间坐标")
    if not np.isfinite(markers).all():
        raise ValueError("marker_positions 必须全部是有限数值")
    if not np.all(np.diff(markers) > 0.0):
        raise ValueError("marker_positions 必须严格递增")
    return markers


def tvt_to_marker_coordinate(
    tvt: np.ndarray | Sequence[float],
    marker_positions: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把绝对 TVT 映射为分段线性的 marker 坐标 ``q``。

    区间采用左闭右开规则：``m[k] <= TVT < m[k+1]``。marker 本身的
    q 坐标是整数 k，区间内部按相对厚度线性插值。首 marker 以外和末
    marker 及其以外均不外推，分别返回 NaN、zone=-1、valid=False。
    """

    markers = _validate_marker_positions(np.asarray(marker_positions))
    tvt_values = np.asarray(tvt, dtype=np.float64)

    # 输出保持输入 TVT 的原 shape，便于逐行写回原始井表。
    marker_coordinate = np.full(tvt_values.shape, np.nan, dtype=np.float64)
    zone_index = np.full(tvt_values.shape, -1, dtype=np.int64)

    # 末 marker 不属于任何区间，因此上界必须严格小于 markers[-1]。
    valid_mask = (
        np.isfinite(tvt_values)
        & (tvt_values >= markers[0])
        & (tvt_values < markers[-1])
    )

    if np.any(valid_mask):
        valid_tvt = tvt_values[valid_mask]

        # side="right" 让恰好等于内部 marker 的点进入它右侧的新区间。
        valid_zones = np.searchsorted(markers, valid_tvt, side="right") - 1
        lower_markers = markers[valid_zones]
        upper_markers = markers[valid_zones + 1]
        interval_thickness = upper_markers - lower_markers

        valid_q = valid_zones + (valid_tvt - lower_markers) / interval_thickness
        marker_coordinate[valid_mask] = valid_q
        zone_index[valid_mask] = valid_zones

    return marker_coordinate, zone_index, valid_mask


def marker_coordinate_to_tvt(
    marker_coordinate: np.ndarray | Sequence[float],
    marker_positions: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """把 marker 坐标 ``q`` 逆变换回绝对 TVT，不在定义域外推。"""

    markers = _validate_marker_positions(np.asarray(marker_positions))
    q_values = np.asarray(marker_coordinate, dtype=np.float64)
    recovered_tvt = np.full(q_values.shape, np.nan, dtype=np.float64)

    # 有 n 个 marker 时只定义 n-1 个区间，所以合法 q 范围是 [0, n-1)。
    maximum_q = float(len(markers) - 1)
    valid_mask = np.isfinite(q_values) & (q_values >= 0.0) & (q_values < maximum_q)

    if np.any(valid_mask):
        valid_q = q_values[valid_mask]
        valid_zones = np.floor(valid_q).astype(np.int64)
        fractional_position = valid_q - valid_zones
        lower_markers = markers[valid_zones]
        interval_thickness = markers[valid_zones + 1] - lower_markers
        recovered_tvt[valid_mask] = lower_markers + fractional_position * interval_thickness

    return recovered_tvt, valid_mask


def normalized_gated_rank(
    costs: np.ndarray | Sequence[float],
    true_candidate_position: int,
    eligible_mask: np.ndarray | Sequence[bool],
) -> dict[str, Any]:
    """计算真实候选在门控候选中的排名和归一化排名。

    cost 越小排名越靠前。若门控排除了真实候选，或真实候选 cost 不是
    有限值，``normalized_rank`` 固定记为 1（最差），不能通过丢行美化结果。
    单候选且命中时归一化排名记为 0。
    """

    cost_values = np.asarray(costs, dtype=np.float64)
    gate_mask = np.asarray(eligible_mask, dtype=bool)
    if cost_values.ndim != 1 or gate_mask.ndim != 1:
        raise ValueError("costs 和 eligible_mask 必须是一维数组")
    if cost_values.shape != gate_mask.shape:
        raise ValueError("costs 和 eligible_mask 的长度必须相同")
    if isinstance(true_candidate_position, bool):
        raise ValueError("true_candidate_position 必须是整数位置")
    true_position = int(true_candidate_position)
    if true_position != true_candidate_position:
        raise ValueError("true_candidate_position 必须是整数位置")
    if true_position < 0 or true_position >= len(cost_values):
        raise IndexError("true_candidate_position 超出候选范围")

    # 非有限 cost 不能参与有效排名；候选数按实际可比较的候选计算。
    comparable_mask = gate_mask & np.isfinite(cost_values)
    candidate_count = int(np.count_nonzero(comparable_mask))
    true_included = bool(comparable_mask[true_position])

    if not true_included:
        # rank 记作门外的下一名，同时按预注册规则把百分位固定成最差 1。
        return {
            "true_included": False,
            "candidate_count": candidate_count,
            "rank": candidate_count + 1,
            "normalized_rank": 1.0,
        }

    true_cost = cost_values[true_position]

    # 只统计严格更低的 cost；相同 cost 共享最好名次，避免按数组顺序偏置。
    rank = 1 + int(np.count_nonzero(comparable_mask & (cost_values < true_cost)))
    if candidate_count == 1:
        normalized_rank = 0.0
    else:
        normalized_rank = float((rank - 1) / (candidate_count - 1))

    return {
        "true_included": True,
        "candidate_count": candidate_count,
        "rank": rank,
        "normalized_rank": normalized_rank,
    }


def deterministic_contiguous_gate(
    number_of_candidates: int,
    gate_size: int,
    key: str,
    seed: int,
) -> np.ndarray:
    """用 ``key`` 和 ``seed`` 的 SHA-256 固定选择一段连续候选。

    返回形状为 ``[number_of_candidates]`` 的布尔数组。选择不依赖 Python
    进程的随机哈希，因此换机器、换进程或恢复运行时都会得到相同结果。
    """

    if isinstance(number_of_candidates, bool) or int(number_of_candidates) != number_of_candidates:
        raise ValueError("number_of_candidates 必须是整数")
    if isinstance(gate_size, bool) or int(gate_size) != gate_size:
        raise ValueError("gate_size 必须是整数")

    candidate_count = int(number_of_candidates)
    selected_count = int(gate_size)
    if candidate_count < 0:
        raise ValueError("number_of_candidates 不能为负数")
    if selected_count < 0 or selected_count > candidate_count:
        raise ValueError("gate_size 必须位于 0 到候选总数之间")

    gate = np.zeros(candidate_count, dtype=bool)
    if selected_count == 0:
        return gate

    # 长度固定的 seed 字节加上带长度前缀的 key，可避免字符串拼接歧义。
    key_bytes = str(key).encode("utf-8")
    seed_bytes = int(seed).to_bytes(16, byteorder="little", signed=True)
    key_length_bytes = len(key_bytes).to_bytes(8, byteorder="little", signed=False)
    digest = hashlib.sha256(key_length_bytes + key_bytes + seed_bytes).digest()

    # 可选起点共有 N-K+1 个；完整选择全部候选时唯一合法起点就是 0。
    number_of_starts = candidate_count - selected_count + 1
    start_position = int.from_bytes(digest[:8], byteorder="little") % number_of_starts
    gate[start_position : start_position + selected_count] = True
    return gate
