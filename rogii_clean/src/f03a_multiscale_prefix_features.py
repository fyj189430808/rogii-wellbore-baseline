"""F03a v2：只用可见前缀衡量当前水平井与 Typewell 的匹配可靠性。"""

from __future__ import annotations

import numpy as np
import pandas as pd


SMOOTH_SCALES_FT = (2, 5, 10, 20)

F03A_MULTISCALE_FEATURE_COLUMNS = [
    "f03a_prefix_raw_ncc",
    "f03a_prefix_2ft_ncc",
    "f03a_prefix_5ft_ncc",
    "f03a_prefix_10ft_ncc",
    "f03a_prefix_20ft_ncc",
    "f03a_prefix_affine_median_ae",
    "f03a_prefix_derivative_ncc",
    "f03a_prefix_valid_pair_count",
    "f03a_prefix_valid_pair_fraction",
]


def _pearson_ncc(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个共同有效序列的 Pearson 相关系数。"""

    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    if int(finite.sum()) < 2:
        return float("nan")

    first_centered = first_values[finite] - np.mean(first_values[finite])
    second_centered = second_values[finite] - np.mean(second_values[finite])
    denominator = np.sqrt(
        np.sum(first_centered * first_centered)
        * np.sum(second_centered * second_centered)
    )
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(first_centered * second_centered) / denominator)


def _clean_typewell(typewell_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """清理 Typewell 的 TVT/GR，并把重复 TVT 的 GR 合并为中位数。"""

    missing = {"TVT", "GR"} - set(typewell_df.columns)
    if missing:
        raise ValueError(f"Typewell 缺少列：{sorted(missing)}")

    cleaned = typewell_df[["TVT", "GR"]].copy()
    cleaned["TVT"] = pd.to_numeric(cleaned["TVT"], errors="coerce")
    cleaned["GR"] = pd.to_numeric(cleaned["GR"], errors="coerce")
    cleaned = cleaned.loc[
        np.isfinite(cleaned["TVT"].to_numpy(dtype=np.float64))
        & np.isfinite(cleaned["GR"].to_numpy(dtype=np.float64))
    ]
    cleaned = cleaned.groupby("TVT", as_index=False)["GR"].median()
    cleaned = cleaned.sort_values("TVT")
    if len(cleaned) < 2:
        raise ValueError("Typewell 至少需要两个有限且不同的 TVT/GR 点")
    return (
        cleaned["TVT"].to_numpy(dtype=np.float64),
        cleaned["GR"].to_numpy(dtype=np.float64),
    )


def _collapse_duplicate_md(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    reference_gr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同一 MD 的共同有效配对取中位数，返回严格递增的 MD。"""

    pairs = pd.DataFrame(
        {"md": md, "horizontal_gr": horizontal_gr, "reference_gr": reference_gr}
    )
    pairs = pairs.groupby("md", as_index=False)[
        ["horizontal_gr", "reference_gr"]
    ].median()
    pairs = pairs.sort_values("md")
    return (
        pairs["md"].to_numpy(dtype=np.float64),
        pairs["horizontal_gr"].to_numpy(dtype=np.float64),
        pairs["reference_gr"].to_numpy(dtype=np.float64),
    )


def _centered_smooth_ncc(
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    reference_gr: np.ndarray,
    scale_ft: int,
) -> float:
    """先沿 MD 重采样到 1 ft，再用固定物理宽度的中心窗口计算 NCC。"""

    grid_start = float(np.ceil(md[0]))
    grid_stop = float(np.floor(md[-1]))
    if grid_stop <= grid_start:
        return float("nan")

    md_grid = np.arange(grid_start, grid_stop + 0.5, 1.0, dtype=np.float64)
    horizontal_grid = np.interp(md_grid, md, horizontal_gr)
    reference_grid = np.interp(md_grid, md, reference_gr)

    # 例如 10 ft 使用中心点左右各 5 ft，共 11 个 1-ft 样本。
    radius_points = int(np.floor(float(scale_ft) / 2.0))
    window_points = radius_points * 2 + 1
    horizontal_smooth = pd.Series(horizontal_grid).rolling(
        window=window_points,
        center=True,
        min_periods=window_points,
    ).mean()
    reference_smooth = pd.Series(reference_grid).rolling(
        window=window_points,
        center=True,
        min_periods=window_points,
    ).mean()
    return _pearson_ncc(
        horizontal_smooth.to_numpy(dtype=np.float64),
        reference_smooth.to_numpy(dtype=np.float64),
    )


def _affine_median_absolute_error(
    horizontal_gr: np.ndarray,
    reference_gr: np.ndarray,
) -> float:
    """拟合 horizontal≈a×reference+b，并返回抗尖峰的中位绝对误差。"""

    if len(horizontal_gr) < 2:
        return float("nan")
    if float(np.std(reference_gr)) <= 1e-12:
        slope = 1.0
        intercept = float(np.median(horizontal_gr - reference_gr))
    else:
        design = np.column_stack([reference_gr, np.ones(len(reference_gr))])
        slope, intercept = np.linalg.lstsq(design, horizontal_gr, rcond=None)[0]
    calibrated = float(slope) * reference_gr + float(intercept)
    return float(np.median(np.abs(horizontal_gr - calibrated)))


def build_multiscale_prefix_features(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
) -> dict[str, float]:
    """返回一口井的 9 个可见前缀特征，不读取水平井隐藏 TVT。"""

    missing = {"MD", "GR", "TVT_input"} - set(horizontal_df.columns)
    if missing:
        raise ValueError(f"水平井缺少列：{sorted(missing)}")

    md_all = pd.to_numeric(horizontal_df["MD"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    horizontal_gr_all = pd.to_numeric(
        horizontal_df["GR"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    visible_tvt_all = pd.to_numeric(
        horizontal_df["TVT_input"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    visible_mask = np.isfinite(visible_tvt_all)
    visible_count = int(visible_mask.sum())

    typewell_tvt, typewell_gr = _clean_typewell(typewell_df)
    matched_mask = (
        visible_mask
        & np.isfinite(md_all)
        & np.isfinite(horizontal_gr_all)
        & (visible_tvt_all >= typewell_tvt[0])
        & (visible_tvt_all <= typewell_tvt[-1])
    )
    matched_count = int(matched_mask.sum())
    matched_fraction = (
        float(matched_count / visible_count) if visible_count > 0 else float("nan")
    )

    feature_values = {
        feature_name: float("nan")
        for feature_name in F03A_MULTISCALE_FEATURE_COLUMNS
    }
    feature_values["f03a_prefix_valid_pair_count"] = float(matched_count)
    feature_values["f03a_prefix_valid_pair_fraction"] = matched_fraction
    if matched_count < 2:
        return feature_values

    matched_md = md_all[matched_mask]
    matched_horizontal_gr = horizontal_gr_all[matched_mask]
    matched_reference_gr = np.interp(
        visible_tvt_all[matched_mask],
        typewell_tvt,
        typewell_gr,
    )
    matched_md, matched_horizontal_gr, matched_reference_gr = _collapse_duplicate_md(
        matched_md,
        matched_horizontal_gr,
        matched_reference_gr,
    )

    feature_values["f03a_prefix_raw_ncc"] = _pearson_ncc(
        matched_horizontal_gr,
        matched_reference_gr,
    )
    for scale_ft in SMOOTH_SCALES_FT:
        feature_values[f"f03a_prefix_{scale_ft}ft_ncc"] = _centered_smooth_ncc(
            matched_md,
            matched_horizontal_gr,
            matched_reference_gr,
            scale_ft,
        )
    feature_values["f03a_prefix_affine_median_ae"] = _affine_median_absolute_error(
        matched_horizontal_gr,
        matched_reference_gr,
    )

    md_steps = np.diff(matched_md)
    adjacent = md_steps > 0.0
    horizontal_derivative = np.diff(matched_horizontal_gr)[adjacent] / md_steps[adjacent]
    reference_derivative = np.diff(matched_reference_gr)[adjacent] / md_steps[adjacent]
    feature_values["f03a_prefix_derivative_ncc"] = _pearson_ncc(
        horizontal_derivative,
        reference_derivative,
    )
    return feature_values
