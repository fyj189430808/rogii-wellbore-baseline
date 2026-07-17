"""RF03-D0：在合法可见前缀上比较真实 TVT 与固定错位的 typewell GR。"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 所有 offset 必须共用完全相同的水平井行，0 ft 是正对照，其余是固定错位。
ALIGNMENT_OFFSETS_FT = (-20.0, -10.0, 0.0, 10.0, 20.0)


def clean_typewell_arrays(typewell_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """清洗 typewell TVT/GR，重复 TVT 取 GR 中位数，并按 TVT 严格递增返回。"""

    # 只复制本诊断需要的两列，不修改调用者传入的原始 DataFrame。
    cleaned = typewell_df.loc[:, ["TVT", "GR"]].copy()

    # 非数值文本转换为 NaN，随后与原始缺失值一起删除。
    cleaned["TVT"] = pd.to_numeric(cleaned["TVT"], errors="coerce")
    cleaned["GR"] = pd.to_numeric(cleaned["GR"], errors="coerce")
    cleaned = cleaned.loc[
        np.isfinite(cleaned["TVT"].to_numpy(dtype=np.float64))
        & np.isfinite(cleaned["GR"].to_numpy(dtype=np.float64))
    ]

    # 同一 TVT 若有重复采样，使用中位 GR，避免 np.interp 的重复横坐标歧义。
    cleaned = cleaned.groupby("TVT", as_index=False)["GR"].median()
    cleaned = cleaned.sort_values("TVT").reset_index(drop=True)

    # 至少两个不同 TVT 点才能形成可插值的参考曲线。
    if len(cleaned) < 2:
        raise ValueError("typewell 至少需要两个有限且不同的 TVT/GR 点")

    # 两个输出 shape 都为 [清洗后 typewell 行数]。
    typewell_tvt = cleaned["TVT"].to_numpy(dtype=np.float64)
    typewell_gr = cleaned["GR"].to_numpy(dtype=np.float64)
    return typewell_tvt, typewell_gr


def pearson_ncc(first_values: np.ndarray, second_values: np.ndarray) -> float:
    """返回两个等长序列的 Pearson normalized cross-correlation。"""

    # 两个数组都转成 float64，确保中心化和范数计算稳定。
    first = np.asarray(first_values, dtype=np.float64)
    second = np.asarray(second_values, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("NCC 两个输入的 shape 不一致")

    # 公共 mask 仍做防御性有限值检查，任何 offset 都只在同一行集合评分。
    finite_mask = np.isfinite(first) & np.isfinite(second)
    if int(finite_mask.sum()) < 2:
        return float("nan")

    first_centered = first[finite_mask] - float(np.mean(first[finite_mask]))
    second_centered = second[finite_mask] - float(np.mean(second[finite_mask]))
    denominator = float(
        np.sqrt(
            np.sum(first_centered * first_centered)
            * np.sum(second_centered * second_centered)
        )
    )
    if denominator <= 1e-12:
        return float("nan")

    return float(np.sum(first_centered * second_centered) / denominator)


def fit_affine_reference(
    horizontal_gr: np.ndarray,
    reference_gr: np.ndarray,
) -> tuple[float, float]:
    """拟合 horizontal_GR = slope×reference_GR + intercept，返回两个标量。"""

    # 本函数只在调用者已经建立的公共有限行上拟合，不跨井共享参数。
    horizontal = np.asarray(horizontal_gr, dtype=np.float64)
    reference = np.asarray(reference_gr, dtype=np.float64)
    if horizontal.shape != reference.shape:
        raise ValueError("仿射校准两个输入的 shape 不一致")

    # 参考 GR 近似常数时斜率不可识别，退化为只校准均值偏移。
    if float(np.std(reference)) <= 1e-12:
        intercept = float(np.mean(horizontal) - np.mean(reference))
        return 1.0, intercept

    # 一次多项式的第一个系数是 slope，第二个是 intercept。
    slope, intercept = np.polyfit(reference, horizontal, 1)
    return float(slope), float(intercept)


def score_prefix_offsets(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    well_id: str,
    fold: int,
    scope: str,
    tail_window_ft: float | None,
    minimum_points: int,
) -> pd.DataFrame:
    """在一个可见前缀 scope 上为五个固定 TVT offset 计算 NCC 与仿射后中位AE。"""

    # typewell 两个数组 shape 相同，并已处理 NaN、重复 TVT 和排序。
    typewell_tvt, typewell_gr = clean_typewell_arrays(typewell_df)

    # 这里只读取推理时合法的 MD、GR 和 TVT_input，不访问水平井隐藏 TVT。
    md_values = horizontal_df["MD"].to_numpy(dtype=np.float64)
    horizontal_gr = horizontal_df["GR"].to_numpy(dtype=np.float64)
    visible_tvt = horizontal_df["TVT_input"].to_numpy(dtype=np.float64)

    # visible_mask 的 shape 为 [整井行数]，只选 TVT_input 与 GR 同时有限的可见行。
    visible_mask = (
        np.isfinite(md_values)
        & np.isfinite(horizontal_gr)
        & np.isfinite(visible_tvt)
    )

    # tail scope 只保留离最后可见 MD 指定距离内的行，更贴近自然隐藏段起点。
    if tail_window_ft is not None and visible_mask.any():
        last_visible_md = float(np.max(md_values[visible_mask]))
        visible_mask &= md_values >= last_visible_md - float(tail_window_ft)

    # 所有 offset 共用 typewell 支持范围内的同一批行，避免样本量差异制造假优势。
    minimum_offset = float(min(ALIGNMENT_OFFSETS_FT))
    maximum_offset = float(max(ALIGNMENT_OFFSETS_FT))
    visible_mask &= visible_tvt + minimum_offset >= float(typewell_tvt[0])
    visible_mask &= visible_tvt + maximum_offset <= float(typewell_tvt[-1])

    # 两个评分输入的 shape 都为 [公共可见行数]。
    common_tvt = visible_tvt[visible_mask]
    common_horizontal_gr = horizontal_gr[visible_mask]
    number_of_points = int(len(common_tvt))

    # 每个 offset 输出一行，便于后续按井、scope、fold 做确定性汇总。
    score_rows: list[dict[str, float | int | str]] = []
    for offset_ft in ALIGNMENT_OFFSETS_FT:
        raw_ncc = float("nan")
        affine_slope = float("nan")
        affine_intercept = float("nan")
        affine_median_ae = float("nan")

        # 支持量不足时仍保存该井和 n_points，但不伪造分数。
        if number_of_points >= int(minimum_points):
            candidate_tvt = common_tvt + float(offset_ft)
            reference_gr = np.interp(candidate_tvt, typewell_tvt, typewell_gr)
            raw_ncc = pearson_ncc(common_horizontal_gr, reference_gr)
            affine_slope, affine_intercept = fit_affine_reference(
                common_horizontal_gr,
                reference_gr,
            )
            calibrated_reference = affine_slope * reference_gr + affine_intercept
            affine_median_ae = float(
                np.median(np.abs(common_horizontal_gr - calibrated_reference))
            )

        score_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "scope": str(scope),
                "offset_ft": float(offset_ft),
                "n_points": number_of_points,
                "raw_ncc": raw_ncc,
                "affine_slope": affine_slope,
                "affine_intercept": affine_intercept,
                "affine_median_ae": affine_median_ae,
            }
        )

    return pd.DataFrame(score_rows)


def build_prefix_alignment_margins(offset_scores: pd.DataFrame) -> pd.DataFrame:
    """把每井五个 offset 分数汇总为 0 ft 相对最佳错位的两个 margin。"""

    # 每个 group 对应同一口井、fold 和前缀 scope 的五个预注册 offset。
    margin_rows: list[dict[str, float | int | bool | str]] = []
    group_columns = ["well_id", "fold", "scope"]
    for group_key, group_df in offset_scores.groupby(group_columns, sort=True):
        well_id, fold, scope = group_key
        zero_rows = group_df.loc[group_df["offset_ft"] == 0.0]
        wrong_rows = group_df.loc[group_df["offset_ft"] != 0.0]
        if len(zero_rows) != 1 or len(wrong_rows) != 4:
            raise ValueError("每个前缀 scope 必须包含一个0ft和四个错位分数")

        zero_row = zero_rows.iloc[0]
        zero_ncc = float(zero_row["raw_ncc"])
        zero_mae = float(zero_row["affine_median_ae"])
        best_wrong_ncc = float(wrong_rows["raw_ncc"].max())
        best_wrong_mae = float(wrong_rows["affine_median_ae"].min())
        ncc_margin = zero_ncc - best_wrong_ncc
        mae_margin = best_wrong_mae - zero_mae

        margin_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "scope": str(scope),
                "n_points": int(zero_row["n_points"]),
                "zero_raw_ncc": zero_ncc,
                "best_wrong_raw_ncc": best_wrong_ncc,
                "ncc_margin_vs_best_wrong": ncc_margin,
                "zero_affine_median_ae": zero_mae,
                "best_wrong_affine_median_ae": best_wrong_mae,
                "mae_margin_vs_best_wrong": mae_margin,
                "zero_ncc_is_best": bool(np.isfinite(ncc_margin) and ncc_margin >= 0.0),
                "zero_mae_is_best": bool(np.isfinite(mae_margin) and mae_margin >= 0.0),
            }
        )

    return pd.DataFrame(margin_rows)
