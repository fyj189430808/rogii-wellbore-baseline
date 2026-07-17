"""审计训练井 surface 列是否由 U=TVT+Z 与 Typewell marker 派生。

主比较采用 Typewell 中每个 Geology 标签首次出现位置的 TVT。由于 Typewell
通常按 0.5 ft 采样，脚本同时计算“前一采样点与首个标签采样点的中点”作为
边界定义敏感性检查。该脚本只读取训练数据，不训练模型，也不生成正式特征。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_RAW_DIR = WORKSPACE_ROOT / "input" / "data" / "raw" / "train"
DEFAULT_REGISTRY = PROJECT_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "F07R_surface_lineage_audit_v1"

SURFACE_NAMES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
ADJACENT_SURFACE_PAIRS = tuple(zip(SURFACE_NAMES[:-1], SURFACE_NAMES[1:]))

# Typewell 的 TVT 网格通常为 0.5 ft。以下门槛只判断派生关系是否得到强支持，
# 不用于模型训练或特征选择。
MIN_COMPARISON_COVERAGE = 0.95
MAX_P90_WITHIN_WELL_STD_FT = 0.02
MAX_P90_OFFSET_MARKER_ERROR_FT = 0.50
MAX_P90_GAP_ERROR_FT = 0.75


def _finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    """返回一维有限浮点值，供聚合统计使用。"""

    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(np.float64)
    return numeric[np.isfinite(numeric)]


def summarize_values(values: pd.Series | np.ndarray) -> dict[str, Any]:
    """汇总普通数值的 count、median、P90 和 max；空集合返回 None。"""

    finite = _finite_values(values)
    if finite.size == 0:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "max": float(np.max(finite)),
    }


def summarize_absolute(values: pd.Series | np.ndarray) -> dict[str, Any]:
    """汇总绝对误差的 count、median、P90 和 max。"""

    finite = _finite_values(values)
    return summarize_values(np.abs(finite))


def normalize_geology(values: pd.Series) -> pd.Series:
    """统一 Geology 标签的大小写和空白，空字符串按缺失处理。"""

    normalized = values.astype("string").str.strip().str.upper()
    return normalized.mask(normalized.eq(""), pd.NA)


def extract_typewell_markers(typewell_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """提取六个 marker 的首标签 TVT、中点边界 TVT、步长和标签块数量。"""

    empty_marker = {
        "marker_first_sample_tvt": np.nan,
        "marker_midpoint_tvt": np.nan,
        "marker_previous_tvt": np.nan,
        "marker_sample_step_ft": np.nan,
        "marker_block_count": 0,
        "marker_boundary_observed": False,
        "marker_left_censored": False,
    }
    markers = {surface: dict(empty_marker) for surface in SURFACE_NAMES}
    if "TVT" not in typewell_df.columns or "Geology" not in typewell_df.columns:
        return markers

    ordered = typewell_df[["TVT", "Geology"]].copy()
    ordered["TVT"] = pd.to_numeric(ordered["TVT"], errors="coerce")
    ordered = ordered.loc[ordered["TVT"].notna()].sort_values("TVT", kind="stable")
    ordered = ordered.reset_index(drop=True)
    geology = normalize_geology(ordered["Geology"])
    tvt = ordered["TVT"].to_numpy(np.float64)

    for surface in SURFACE_NAMES:
        matches = geology.eq(surface).fillna(False).to_numpy(dtype=bool)
        positions = np.flatnonzero(matches)
        if positions.size == 0:
            continue

        first_position = int(positions[0])
        first_tvt = float(tvt[first_position])
        previous_tvt = np.nan
        midpoint_tvt = np.nan
        sample_step = np.nan
        boundary_observed = False
        if first_position > 0 and np.isfinite(tvt[first_position - 1]):
            previous_tvt = float(tvt[first_position - 1])
            sample_step = first_tvt - previous_tvt
            midpoint_tvt = (first_tvt + previous_tvt) / 2.0
            boundary_observed = True

        starts = matches & ~np.r_[False, matches[:-1]]
        markers[surface] = {
            "marker_first_sample_tvt": first_tvt,
            "marker_midpoint_tvt": float(midpoint_tvt),
            "marker_previous_tvt": previous_tvt,
            "marker_sample_step_ft": sample_step,
            "marker_block_count": int(np.count_nonzero(starts)),
            "marker_boundary_observed": boundary_observed,
            "marker_left_censored": first_position == 0,
        }
    return markers


def audit_one_well(
    well_id: str,
    horizontal_path: Path,
    typewell_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """审计一口井，返回逐 surface、逐相邻 gap 和井级覆盖记录。"""

    horizontal_header = pd.read_csv(horizontal_path, nrows=0).columns.tolist()
    horizontal_usecols = [
        column for column in ("TVT", "Z", *SURFACE_NAMES) if column in horizontal_header
    ]
    horizontal_df = pd.read_csv(horizontal_path, usecols=horizontal_usecols)

    typewell_header = pd.read_csv(typewell_path, nrows=0).columns.tolist()
    typewell_usecols = [
        column for column in ("TVT", "Geology") if column in typewell_header
    ]
    typewell_df = pd.read_csv(typewell_path, usecols=typewell_usecols)
    markers = extract_typewell_markers(typewell_df)

    has_tvt_and_z = "TVT" in horizontal_df.columns and "Z" in horizontal_df.columns
    if has_tvt_and_z:
        tvt = pd.to_numeric(horizontal_df["TVT"], errors="coerce").to_numpy(np.float64)
        z = pd.to_numeric(horizontal_df["Z"], errors="coerce").to_numpy(np.float64)
        structural_u = tvt + z
    else:
        structural_u = np.full(len(horizontal_df), np.nan, dtype=np.float64)

    surface_rows: list[dict[str, Any]] = []
    surface_arrays: dict[str, np.ndarray] = {}
    for surface in SURFACE_NAMES:
        surface_present = surface in horizontal_df.columns
        if surface_present:
            surface_values = pd.to_numeric(
                horizontal_df[surface], errors="coerce"
            ).to_numpy(np.float64)
        else:
            surface_values = np.full(len(horizontal_df), np.nan, dtype=np.float64)
        surface_arrays[surface] = surface_values

        valid = np.isfinite(structural_u) & np.isfinite(surface_values)
        offset_values = structural_u[valid] - surface_values[valid]
        if offset_values.size:
            offset_median = float(np.median(offset_values))
            offset_std = float(np.std(offset_values, ddof=0))
            centered_max_abs = float(np.max(np.abs(offset_values - offset_median)))
        else:
            offset_median = np.nan
            offset_std = np.nan
            centered_max_abs = np.nan

        marker = markers[surface]
        first_marker = float(marker["marker_first_sample_tvt"])
        midpoint_marker = float(marker["marker_midpoint_tvt"])
        surface_rows.append(
            {
                "well_id": well_id,
                "surface": surface,
                "horizontal_rows": int(len(horizontal_df)),
                "surface_column_present": bool(surface_present),
                "surface_valid_rows": int(np.count_nonzero(valid)),
                "typewell_geology_column_present": "Geology" in typewell_df.columns,
                "marker_present": bool(np.isfinite(first_marker)),
                "marker_boundary_observed": bool(marker["marker_boundary_observed"]),
                "marker_left_censored": bool(marker["marker_left_censored"]),
                "marker_block_count": int(marker["marker_block_count"]),
                "marker_previous_tvt": marker["marker_previous_tvt"],
                "marker_first_sample_tvt": first_marker,
                "marker_midpoint_tvt": midpoint_marker,
                "marker_sample_step_ft": marker["marker_sample_step_ft"],
                "median_u_minus_surface": offset_median,
                "within_well_std_u_minus_surface": offset_std,
                "centered_max_abs_u_minus_surface": centered_max_abs,
                "offset_minus_first_marker": (
                    offset_median - first_marker
                    if marker["marker_boundary_observed"]
                    else np.nan
                ),
                "offset_minus_midpoint_marker": (
                    offset_median - midpoint_marker
                    if marker["marker_boundary_observed"]
                    else np.nan
                ),
            }
        )

    surface_by_name = {row["surface"]: row for row in surface_rows}
    gap_rows: list[dict[str, Any]] = []
    for upper_surface, lower_surface in ADJACENT_SURFACE_PAIRS:
        upper_values = surface_arrays[upper_surface]
        lower_values = surface_arrays[lower_surface]
        valid = np.isfinite(upper_values) & np.isfinite(lower_values)
        surface_gap_values = upper_values[valid] - lower_values[valid]
        if surface_gap_values.size:
            surface_gap_median = float(np.median(surface_gap_values))
            surface_gap_std = float(np.std(surface_gap_values, ddof=0))
        else:
            surface_gap_median = np.nan
            surface_gap_std = np.nan

        upper_row = surface_by_name[upper_surface]
        lower_row = surface_by_name[lower_surface]
        both_boundaries_observed = bool(
            upper_row["marker_boundary_observed"]
            and lower_row["marker_boundary_observed"]
        )
        if both_boundaries_observed:
            marker_gap_first = (
                lower_row["marker_first_sample_tvt"]
                - upper_row["marker_first_sample_tvt"]
            )
            marker_gap_midpoint = (
                lower_row["marker_midpoint_tvt"] - upper_row["marker_midpoint_tvt"]
            )
        else:
            marker_gap_first = np.nan
            marker_gap_midpoint = np.nan
        gap_rows.append(
            {
                "well_id": well_id,
                "surface_pair": f"{upper_surface}->{lower_surface}",
                "upper_surface": upper_surface,
                "lower_surface": lower_surface,
                "valid_rows": int(np.count_nonzero(valid)),
                "surface_gap_median": surface_gap_median,
                "surface_gap_within_well_std": surface_gap_std,
                "marker_gap_first_sample": marker_gap_first,
                "marker_gap_midpoint": marker_gap_midpoint,
                "gap_error_first_sample": surface_gap_median - marker_gap_first,
                "gap_error_midpoint": surface_gap_median - marker_gap_midpoint,
            }
        )

    geology_non_null_rows = 0
    geology_unique_labels: list[str] = []
    if "Geology" in typewell_df.columns:
        normalized = normalize_geology(typewell_df["Geology"])
        geology_non_null_rows = int(normalized.notna().sum())
        geology_unique_labels = sorted(normalized.dropna().unique().tolist())
    well_row = {
        "well_id": well_id,
        "horizontal_rows": int(len(horizontal_df)),
        "typewell_rows": int(len(typewell_df)),
        "has_horizontal_tvt_z": bool(has_tvt_and_z),
        "surface_column_count": int(sum(name in horizontal_df.columns for name in SURFACE_NAMES)),
        "typewell_geology_column_present": "Geology" in typewell_df.columns,
        "geology_non_null_rows": geology_non_null_rows,
        "geology_unique_labels": geology_unique_labels,
        "marker_count": int(
            sum(np.isfinite(markers[name]["marker_first_sample_tvt"]) for name in SURFACE_NAMES)
        ),
    }
    return surface_rows, gap_rows, well_row


def _aggregate_surface_records(surface_df: pd.DataFrame) -> dict[str, Any]:
    """按 surface 汇总覆盖率、marker 误差和井内恒等式误差。"""

    result: dict[str, Any] = {}
    for surface in SURFACE_NAMES:
        group = surface_df.loc[surface_df["surface"].eq(surface)].copy()
        comparable = group["median_u_minus_surface"].notna() & group[
            "offset_minus_first_marker"
        ].notna()
        result[surface] = {
            "well_count": int(len(group)),
            "surface_column_present_count": int(group["surface_column_present"].sum()),
            "surface_valid_count": int(group["median_u_minus_surface"].notna().sum()),
            "marker_present_count": int(group["marker_present"].sum()),
            "marker_boundary_observed_count": int(
                group["marker_boundary_observed"].sum()
            ),
            "marker_left_censored_count": int(group["marker_left_censored"].sum()),
            "comparison_count": int(comparable.sum()),
            "comparison_coverage": float(comparable.mean()) if len(group) else 0.0,
            "abs_median_u_minus_surface_minus_first_marker_ft": summarize_absolute(
                group.loc[comparable, "offset_minus_first_marker"]
            ),
            "abs_median_u_minus_surface_minus_midpoint_marker_ft": summarize_absolute(
                group.loc[comparable, "offset_minus_midpoint_marker"]
            ),
            "within_well_std_u_minus_surface_ft": summarize_values(
                group["within_well_std_u_minus_surface"]
            ),
            "centered_max_abs_u_minus_surface_ft": summarize_values(
                group["centered_max_abs_u_minus_surface"]
            ),
            "marker_sample_step_ft": summarize_values(group["marker_sample_step_ft"]),
            "wells_with_multiple_marker_blocks": int(
                group["marker_block_count"].gt(1).sum()
            ),
        }
    return result


def _aggregate_gap_records(gap_df: pd.DataFrame) -> dict[str, Any]:
    """按相邻 surface 对汇总 gap 与 Typewell marker gap 的误差。"""

    result: dict[str, Any] = {}
    for upper_surface, lower_surface in ADJACENT_SURFACE_PAIRS:
        pair = f"{upper_surface}->{lower_surface}"
        group = gap_df.loc[gap_df["surface_pair"].eq(pair)].copy()
        comparable = group["surface_gap_median"].notna() & group[
            "marker_gap_first_sample"
        ].notna()
        result[pair] = {
            "well_count": int(len(group)),
            "comparison_count": int(comparable.sum()),
            "comparison_coverage": float(comparable.mean()) if len(group) else 0.0,
            "abs_surface_gap_minus_first_marker_gap_ft": summarize_absolute(
                group.loc[comparable, "gap_error_first_sample"]
            ),
            "abs_surface_gap_minus_midpoint_marker_gap_ft": summarize_absolute(
                group.loc[comparable, "gap_error_midpoint"]
            ),
            "surface_gap_within_well_std_ft": summarize_values(
                group["surface_gap_within_well_std"]
            ),
        }
    return result


def build_summary(
    registry_df: pd.DataFrame,
    processed_well_ids: list[str],
    surface_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    well_df: pd.DataFrame,
    elapsed_seconds: float,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    """生成机器可读汇总与 F07R 前置条件结论。"""

    surface_summary = _aggregate_surface_records(surface_df)
    gap_summary = _aggregate_gap_records(gap_df)
    expected_comparisons = len(processed_well_ids) * len(SURFACE_NAMES)
    actual_comparisons = int(
        (
            surface_df["median_u_minus_surface"].notna()
            & surface_df["offset_minus_first_marker"].notna()
        ).sum()
    )
    overall_coverage = (
        actual_comparisons / expected_comparisons if expected_comparisons else 0.0
    )

    worst_surface_offset_p90 = max(
        item["abs_median_u_minus_surface_minus_first_marker_ft"]["p90"]
        for item in surface_summary.values()
        if item["abs_median_u_minus_surface_minus_first_marker_ft"]["p90"] is not None
    )
    worst_surface_std_p90 = max(
        item["within_well_std_u_minus_surface_ft"]["p90"]
        for item in surface_summary.values()
        if item["within_well_std_u_minus_surface_ft"]["p90"] is not None
    )
    worst_gap_p90 = max(
        item["abs_surface_gap_minus_first_marker_gap_ft"]["p90"]
        for item in gap_summary.values()
        if item["abs_surface_gap_minus_first_marker_gap_ft"]["p90"] is not None
    )

    relationship_supported = bool(
        overall_coverage >= MIN_COMPARISON_COVERAGE
        and worst_surface_std_p90 <= MAX_P90_WITHIN_WELL_STD_FT
        and worst_surface_offset_p90 <= MAX_P90_OFFSET_MARKER_ERROR_FT
        and worst_gap_p90 <= MAX_P90_GAP_ERROR_FT
    )
    full_registry_processed = bool(
        len(processed_well_ids) == len(registry_df)
        and set(processed_well_ids) == set(registry_df["well_id"].astype(str))
        and not errors
    )

    geology_column_count = int(well_df["typewell_geology_column_present"].sum())
    all_markers_count = int(well_df["marker_count"].eq(len(SURFACE_NAMES)).sum())
    comparison_mask = (
        surface_df["median_u_minus_surface"].notna()
        & surface_df["offset_minus_first_marker"].notna()
    )
    outlier_rows = surface_df.loc[
        comparison_mask & surface_df["offset_minus_first_marker"].abs().gt(1.0),
        ["well_id", "surface", "offset_minus_first_marker"],
    ].copy()
    outlier_rows = outlier_rows.sort_values(
        "offset_minus_first_marker", key=lambda values: values.abs(), ascending=False
    )
    summary = {
        "experiment_id": "F07R_surface_lineage_audit_v1",
        "purpose": "只审计 surface 数据血缘，不训练模型，不生成正式特征",
        "primary_marker_definition": "Typewell 中该 Geology 首次出现采样点的 TVT",
        "sensitivity_marker_definition": "首次标签采样点与前一 TVT 采样点的中点",
        "expected_registry_wells": int(len(registry_df)),
        "processed_wells": int(len(processed_well_ids)),
        "processing_error_count": int(len(errors)),
        "processing_errors": errors,
        "full_registry_processed": full_registry_processed,
        "elapsed_seconds": float(elapsed_seconds),
        "coverage": {
            "wells_with_typewell_geology_column": geology_column_count,
            "wells_with_all_six_markers": all_markers_count,
            "left_censored_marker_records": int(
                surface_df["marker_left_censored"].sum()
            ),
            "wells_with_all_six_surface_columns": int(
                well_df["surface_column_count"].eq(len(SURFACE_NAMES)).sum()
            ),
            "surface_marker_comparisons": actual_comparisons,
            "expected_surface_marker_comparisons": expected_comparisons,
            "surface_marker_comparison_ratio": float(overall_coverage),
            "absolute_marker_error_over_1ft_records": int(len(outlier_rows)),
            "absolute_marker_error_over_1ft_wells": int(
                outlier_rows["well_id"].nunique()
            ),
            "absolute_marker_error_over_1ft_details": outlier_rows.to_dict(
                orient="records"
            ),
        },
        "per_surface": surface_summary,
        "per_adjacent_surface_gap": gap_summary,
        "relationship_thresholds": {
            "minimum_comparison_coverage": MIN_COMPARISON_COVERAGE,
            "maximum_worst_surface_p90_within_well_std_ft": MAX_P90_WITHIN_WELL_STD_FT,
            "maximum_worst_surface_p90_offset_marker_error_ft": MAX_P90_OFFSET_MARKER_ERROR_FT,
            "maximum_worst_pair_p90_gap_error_ft": MAX_P90_GAP_ERROR_FT,
        },
        "relationship_checks": {
            "comparison_coverage": float(overall_coverage),
            "worst_surface_p90_within_well_std_ft": float(worst_surface_std_p90),
            "worst_surface_p90_offset_marker_error_ft": float(worst_surface_offset_p90),
            "worst_pair_p90_gap_error_ft": float(worst_gap_p90),
            "derived_relationship_supported": relationship_supported,
        },
        "f07r_lineage_prerequisite": {
            "audit_complete": full_registry_processed,
            "derived_relationship_supported": relationship_supported,
            "satisfied": bool(full_registry_processed),
            "note": (
                "surface 已完成全量血缘审计；F07R 正式特征仍应直接使用严格 outer-train 的 U=TVT+Z，"
                "不使用验证井 surface。"
                if full_registry_processed
                else "尚未完成全部 registry 井，不能视为满足 lineage 已审计前置条件。"
            ),
        },
    }
    return summary


def write_conclusion(summary: dict[str, Any], output_path: Path) -> None:
    """把关键事实和 F07R 前置条件写成简短中文结论。"""

    coverage = summary["coverage"]
    checks = summary["relationship_checks"]
    prerequisite = summary["f07r_lineage_prerequisite"]
    surface_lines = []
    for surface, item in summary["per_surface"].items():
        error = item["abs_median_u_minus_surface_minus_first_marker_ft"]
        within_std = item["within_well_std_u_minus_surface_ft"]
        surface_lines.append(
            f"| {surface} | {item['comparison_count']} | {error['median']:.6f} | "
            f"{error['p90']:.6f} | {error['max']:.6f} | {within_std['median']:.6f} | "
            f"{within_std['p90']:.6f} |"
        )

    gap_lines = []
    for pair, item in summary["per_adjacent_surface_gap"].items():
        error = item["abs_surface_gap_minus_first_marker_gap_ft"]
        gap_lines.append(
            f"| {pair} | {item['comparison_count']} | {error['median']:.6f} | "
            f"{error['p90']:.6f} | {error['max']:.6f} |"
        )

    supported_text = "支持" if checks["derived_relationship_supported"] else "不支持"
    prerequisite_text = "满足" if prerequisite["satisfied"] else "不满足"
    content = f"""# F07R surface lineage 审计结论

## 事实

- 已处理 {summary['processed_wells']} / {summary['expected_registry_wells']} 口 registry 训练井，处理错误 {summary['processing_error_count']} 口。
- {coverage['wells_with_typewell_geology_column']} 口 Typewell 有 `Geology` 列，{coverage['wells_with_all_six_markers']} 口包含全部六个 marker。
- 可比较的 surface-marker 记录为 {coverage['surface_marker_comparisons']} / {coverage['expected_surface_marker_comparisons']}，覆盖率 {coverage['surface_marker_comparison_ratio']:.4%}。
- 另有 {coverage['left_censored_marker_records']} 个 marker 标签从 Typewell 第一行就开始，真实上边界被文件截断，因此不把首行误当边界参与误差统计。
- 有 {coverage['absolute_marker_error_over_1ft_records']} 条记录（{coverage['absolute_marker_error_over_1ft_wells']} 口井）绝对 marker 偏差超过 1 ft；逐条记录保存在 `summary.json`，用于识别 Typewell 边界版本或垂向基准异常。
- 主要 marker 定义：Typewell 中该 Geology 首次出现采样点的 TVT；另保存相邻采样中点定义作敏感性检查。

| surface | 比较井数 | marker绝对误差中位数(ft) | P90(ft) | 最大值(ft) | 井内std中位数(ft) | 井内std P90(ft) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(surface_lines)}

| 相邻surface对 | 比较井数 | gap绝对误差中位数(ft) | P90(ft) | 最大值(ft) |
|---|---:|---:|---:|---:|
{chr(10).join(gap_lines)}

## 推断

- 全量结果对“`surface ≈ U - Typewell marker`，其中 `U=TVT+Z`”这一派生关系的判断：**{supported_text}**。
- 最差 surface 的 marker 误差 P90 为 {checks['worst_surface_p90_offset_marker_error_ft']:.6f} ft；最差 surface 的井内 std P90 为 {checks['worst_surface_p90_within_well_std_ft']:.6f} ft；最差相邻 gap 误差 P90 为 {checks['worst_pair_p90_gap_error_ft']:.6f} ft。
- 这说明六个 surface 更适合被解释为 TVT/Typewell marker 的派生标签，而不是六份相互独立的地质监督。

## 当前只能否定

- 不能把六个 surface 的近乎平行和 `U-surface` 近常数，直接解释成六个独立地质观测共同验证了真实地层结构。
- 本审计不判断邻井相对 ΔU 是否有效，也没有训练或评分任何模型。

## F07R 前置条件

- 路线图中的“surface lineage 已审计”前置条件：**{prerequisite_text}**。
- 正式 F07R 仍应只从严格 outer-train 井的 `U=TVT+Z` 构造邻井相对先验，不使用验证井 surface。
"""
    output_path.write_text(content, encoding="utf-8")


def run_audit(
    raw_dir: Path,
    registry_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """按固定 registry 运行审计并写出 CSV、JSON 和结论。"""

    started = time.perf_counter()
    registry_df = pd.read_csv(registry_path, dtype={"well_id": str})
    if registry_df["well_id"].duplicated().any():
        raise ValueError("固定 fold registry 含重复 well_id")
    selected_ids = registry_df["well_id"].astype(str).tolist()
    if limit is not None:
        selected_ids = selected_ids[: int(limit)]

    all_surface_rows: list[dict[str, Any]] = []
    all_gap_rows: list[dict[str, Any]] = []
    all_well_rows: list[dict[str, Any]] = []
    processed_ids: list[str] = []
    errors: list[dict[str, str]] = []
    for position, well_id in enumerate(selected_ids, start=1):
        horizontal_path = raw_dir / f"{well_id}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well_id}__typewell.csv"
        try:
            surface_rows, gap_rows, well_row = audit_one_well(
                well_id=well_id,
                horizontal_path=horizontal_path,
                typewell_path=typewell_path,
            )
            all_surface_rows.extend(surface_rows)
            all_gap_rows.extend(gap_rows)
            all_well_rows.append(well_row)
            processed_ids.append(well_id)
        except Exception as exc:  # 不中断全量覆盖统计，错误会进入 summary。
            errors.append({"well_id": well_id, "error": f"{type(exc).__name__}: {exc}"})
        if position % 100 == 0 or position == len(selected_ids):
            print(f"进度 {position}/{len(selected_ids)}，成功 {len(processed_ids)}，错误 {len(errors)}")

    surface_df = pd.DataFrame(all_surface_rows)
    gap_df = pd.DataFrame(all_gap_rows)
    well_df = pd.DataFrame(all_well_rows)
    elapsed_seconds = time.perf_counter() - started
    summary = build_summary(
        registry_df=registry_df,
        processed_well_ids=processed_ids,
        surface_df=surface_df,
        gap_df=gap_df,
        well_df=well_df,
        elapsed_seconds=elapsed_seconds,
        errors=errors,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    surface_df.to_csv(output_dir / "per_well_surface.csv", index=False)
    gap_df.to_csv(output_dir / "per_well_gap.csv", index=False)
    well_export = well_df.copy()
    well_export["geology_unique_labels"] = well_export["geology_unique_labels"].map(
        lambda values: "|".join(values)
    )
    well_export.to_csv(output_dir / "per_well_coverage.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_conclusion(summary, output_dir / "conclusion.md")
    return summary


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    summary = run_audit(
        raw_dir=args.raw_dir,
        registry_path=args.registry,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(json.dumps(summary["coverage"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["relationship_checks"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["f07r_lineage_prerequisite"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
