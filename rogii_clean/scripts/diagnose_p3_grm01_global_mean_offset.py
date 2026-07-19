"""运行 P3-GRM01 outer0：先合法选择常数偏移，落盘后才读取真值评分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_pf02_target_ess_lgbm_cv import load_development_registry  # noqa: E402
from src.p3_grm01_global_mean_offset import (  # noqa: E402
    CANDIDATE_OFFSETS_FT,
    circular_shift_finite_gr,
    nearest_grid_offset,
    rank_candidate_scores,
    score_legal_well,
)


EXPERIMENT_ID = "P3_GRM01_global_mean_offset_v1"
OUTER_SOURCE = (
    CLEAN_ROOT
    / "artifacts/P3_R01a_nested_linear_residual_v1/base_models/outer_0/base/fold_0/predictions.parquet"
)
OUTER_SOURCE_SHA256 = "cb0d1b778a2193507204fd6b104efbdb4fd44f20a4a96d3f9eff139759df9bad"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
RAW_TRAIN = PROJECT_ROOT / "input/data/raw/train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wells", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legal_outer_rows(well_ids: list[str]) -> pd.DataFrame:
    """物理只读取无标签列；此函数没有 target_tvt 列请求。"""

    observed_hash = file_sha256(OUTER_SOURCE)
    if observed_hash != OUTER_SOURCE_SHA256:
        raise ValueError(
            f"strict outer0 来源 SHA256 不匹配：{observed_hash}"
        )
    dataset = arrow_dataset.dataset(str(OUTER_SOURCE), format="parquet")
    legal_columns = ["well_id", "fold", "row_index", "md", "pred_tvt"]
    table = dataset.to_table(
        columns=legal_columns,
        filter=arrow_dataset.field("well_id").isin(well_ids),
    )
    frame = table.to_pandas()
    frame["well_id"] = frame["well_id"].astype(str)
    if "target_tvt" in frame.columns:
        raise RuntimeError("合法 outer0 读取意外包含 target_tvt")
    return frame.sort_values(["well_id", "row_index"]).reset_index(drop=True)


def validate_legal_outer_contract(
    legal_outer: pd.DataFrame,
    requested_wells: list[str],
    registry: pd.DataFrame,
    expected_rows: int,
) -> None:
    """严格核对井集合、fold、自然隐藏行数、键唯一性和基础路径有限性。"""

    actual_wells = set(legal_outer["well_id"].astype(str))
    if actual_wells != set(str(well_id) for well_id in requested_wells):
        raise ValueError("strict source 井集合与请求集合不完全相等")
    if len(legal_outer) != int(expected_rows):
        raise ValueError("strict source 行数与注册表自然隐藏行数不一致")
    if legal_outer.duplicated(["well_id", "row_index"]).any():
        raise ValueError("strict source 含重复 well_id/row_index 键")
    if not legal_outer["fold"].astype(int).eq(0).all():
        raise ValueError("strict source 并非全部属于 outer fold 0")
    finite = legal_outer[["md", "pred_tvt"]].to_numpy(dtype=np.float64)
    if not np.isfinite(finite).all():
        raise ValueError("strict source 的 md 或 pred_tvt 含 NaN/Inf")
    expected_counts = (
        registry.loc[registry["well_id"].astype(str).isin(actual_wells)]
        .set_index("well_id")["hidden_rows"]
        .astype(int)
        .to_dict()
    )
    actual_counts = legal_outer.groupby("well_id").size().astype(int).to_dict()
    if actual_counts != expected_counts:
        raise ValueError("strict source 每井行数与 registry hidden_rows 不一致")


def load_targets_after_legal_save(well_ids: list[str]) -> pd.DataFrame:
    """仅在合法选择已落盘后，独立读取评分所需 target_tvt。"""

    dataset = arrow_dataset.dataset(str(OUTER_SOURCE), format="parquet")
    table = dataset.to_table(
        columns=["well_id", "row_index", "target_tvt"],
        filter=arrow_dataset.field("well_id").isin(well_ids),
    )
    targets = table.to_pandas()
    targets["well_id"] = targets["well_id"].astype(str)
    return targets


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(error * error)))


def build_top_k_flags(
    rank: float,
    grid_covered: bool,
    evidence_sufficient: bool,
) -> dict[str, bool]:
    """只有网格覆盖且证据充分时，候选名次才可计入 Top-K。"""

    qualified = bool(grid_covered and evidence_sufficient and np.isfinite(rank))
    return {
        "qualified": qualified,
        "top1": bool(qualified and float(rank) <= 1.0),
        "top3": bool(qualified and float(rank) <= 3.0),
        "top5": bool(qualified and float(rank) <= 5.0),
    }


def summarize_path_distribution(
    target_tvt: np.ndarray,
    predicted_tvt: np.ndarray,
    well_ids: np.ndarray,
) -> dict[str, float]:
    """同时汇总逐行 micro 和按井等权的 macro/中位数/P90/最差井。"""

    target = np.asarray(target_tvt, dtype=np.float64)
    prediction = np.asarray(predicted_tvt, dtype=np.float64)
    wells = np.asarray(well_ids).astype(str)
    if not (target.shape == prediction.shape == wells.shape):
        raise ValueError("路径指标的 target、prediction、well_ids shape 不一致")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("路径指标输入含 NaN/Inf")
    well_rmse = []
    for well_id in np.unique(wells):
        mask = wells == well_id
        well_rmse.append(_rmse(target[mask], prediction[mask]))
    values = np.asarray(well_rmse, dtype=np.float64)
    return {
        "micro_rmse": _rmse(target, prediction),
        "macro_well_rmse": float(np.mean(values)),
        "median_well_rmse": float(np.median(values)),
        "p90_well_rmse": float(np.quantile(values, 0.90)),
        "worst_well_rmse": float(np.max(values)),
    }


def run_legal_stage(
    legal_outer: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用真实/循环错移 GR 分别评分；产物中禁止出现真值、RMSE 和名次。"""

    score_tables: list[pd.DataFrame] = []
    selection_rows: list[dict[str, float | int | bool | str]] = []
    well_ids = legal_outer["well_id"].drop_duplicates().tolist()
    for number, well_id in enumerate(well_ids, start=1):
        well_path = legal_outer.loc[legal_outer["well_id"].eq(well_id)].sort_values("row_index")
        horizontal = pd.read_csv(
            RAW_TRAIN / f"{well_id}__horizontal_well.csv",
            usecols=["MD", "GR", "TVT_input"],
        )
        typewell = pd.read_csv(
            RAW_TRAIN / f"{well_id}__typewell.csv",
            usecols=["TVT", "GR"],
        )
        hidden_positions = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        observed_positions = well_path["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(hidden_positions, observed_positions):
            raise ValueError(f"{well_id} strict outer0 与自然隐藏行键不一致")
        base_tvt = well_path["pred_tvt"].to_numpy(dtype=np.float64)
        true_result = score_legal_well(base_tvt, horizontal, typewell)
        hidden_gr = pd.to_numeric(
            horizontal.loc[horizontal["TVT_input"].isna(), "GR"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        shifted_result = score_legal_well(
            base_tvt,
            horizontal,
            typewell,
            hidden_gr_override=circular_shift_finite_gr(hidden_gr),
        )
        for control_name, result in (
            ("true_gr", true_result),
            ("circular_shift", shifted_result),
        ):
            scores = result.scores.copy()
            scores.insert(0, "control", control_name)
            scores.insert(0, "well_id", well_id)
            score_tables.append(scores)
            selection_rows.append(
                {
                    "well_id": well_id,
                    "control": control_name,
                    "selected_offset_ft": result.selected_offset_ft,
                    "evidence_sufficient": result.evidence_sufficient,
                    "common_points": result.common_points,
                    "valid_blocks": result.valid_blocks,
                    "prefix_pairs": result.prefix_pairs,
                    "affine_slope": result.affine_slope,
                    "affine_intercept": result.affine_intercept,
                    "robust_scale": result.robust_scale,
                    "prefix_fallback": result.prefix_fallback,
                }
            )
        print(f"合法 GR 偏移搜索 {number}/{len(well_ids)}：{well_id}", flush=True)

    candidate_scores = pd.concat(score_tables, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    forbidden_fragments = ("target", "true_m", "rmse", "rank")
    for frame_name, frame in (("候选得分", candidate_scores), ("合法选择", selections)):
        illegal = [
            column for column in frame.columns
            if any(fragment in column.lower() for fragment in forbidden_fragments)
        ]
        if illegal:
            raise RuntimeError(f"{frame_name}含非法评分列：{illegal}")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_scores.to_parquet(output_dir / "legal_candidate_scores.parquet", index=False)
    selections.to_csv(output_dir / "legal_selections.csv", index=False)
    return candidate_scores, selections


def score_after_legal_stage(
    legal_outer: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    selections: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    """合法文件存在后才读取真值，计算 oracle 排名和四条路径指标。"""

    if not (output_dir / "legal_candidate_scores.parquet").is_file():
        raise RuntimeError("合法候选得分尚未落盘，禁止读取真值")
    well_ids = legal_outer["well_id"].drop_duplicates().tolist()
    targets = load_targets_after_legal_save(well_ids)
    scored_rows = legal_outer.merge(
        targets,
        on=["well_id", "row_index"],
        how="inner",
        validate="one_to_one",
    )
    path_tables: list[pd.DataFrame] = []
    audit_rows: list[dict[str, float | int | bool | str]] = []
    for well_id, well in scored_rows.groupby("well_id", sort=False):
        well = well.sort_values("row_index").copy()
        base = well["pred_tvt"].to_numpy(dtype=np.float64)
        target = well["target_tvt"].to_numpy(dtype=np.float64)
        true_mean_residual = float(np.mean(target - base))
        oracle_offset = nearest_grid_offset(true_mean_residual)
        grid_covered = abs(true_mean_residual) <= 30.0
        selected = selections.loc[
            selections["well_id"].eq(well_id) & selections["control"].eq("true_gr")
        ].iloc[0]
        shifted = selections.loc[
            selections["well_id"].eq(well_id) & selections["control"].eq("circular_shift")
        ].iloc[0]
        true_ranked = rank_candidate_scores(
            candidate_scores.loc[
                candidate_scores["well_id"].eq(well_id)
                & candidate_scores["control"].eq("true_gr")
            ]
        )
        shifted_ranked = rank_candidate_scores(
            candidate_scores.loc[
                candidate_scores["well_id"].eq(well_id)
                & candidate_scores["control"].eq("circular_shift")
            ]
        )
        true_evidence = bool(selected["evidence_sufficient"])
        shifted_evidence = bool(shifted["evidence_sufficient"])
        true_rank = float("nan")
        shifted_rank = float("nan")
        if true_evidence and grid_covered:
            true_rank = float(
                true_ranked.loc[
                    true_ranked["offset_ft"].eq(oracle_offset), "rank"
                ].iloc[0]
            )
        if shifted_evidence and grid_covered:
            shifted_rank = float(
                shifted_ranked.loc[
                    shifted_ranked["offset_ft"].eq(oracle_offset), "rank"
                ].iloc[0]
            )
        selected_offset = float(selected["selected_offset_ft"])
        shifted_offset = float(shifted["selected_offset_ft"])
        true_flags = build_top_k_flags(true_rank, grid_covered, true_evidence)
        shifted_flags = build_top_k_flags(
            shifted_rank, grid_covered, shifted_evidence
        )
        well["base_tvt"] = base
        well["selected_tvt"] = base + selected_offset
        well["shifted_control_tvt"] = base + shifted_offset
        well["oracle_grid_tvt"] = base + oracle_offset
        path_tables.append(well)
        direction_eligible = abs(true_mean_residual) >= 2.0
        audit_rows.append(
            {
                "well_id": well_id,
                "true_mean_residual_ft": true_mean_residual,
                "oracle_offset_ft": oracle_offset,
                "true_gr_selected_offset_ft": selected_offset,
                "shifted_selected_offset_ft": shifted_offset,
                "true_m_rank": true_rank,
                "shifted_true_m_rank": shifted_rank,
                "true_top1": true_flags["top1"],
                "true_top3": true_flags["top3"],
                "true_top5": true_flags["top5"],
                "shifted_top1": shifted_flags["top1"],
                "shifted_top3": shifted_flags["top3"],
                "shifted_top5": shifted_flags["top5"],
                "grid_covered": grid_covered,
                "evidence_sufficient": true_evidence,
                "qualified": true_flags["qualified"],
                "shifted_qualified": shifted_flags["qualified"],
                "direction_eligible": direction_eligible,
                "true_direction_correct": bool(
                    direction_eligible and np.sign(selected_offset) == np.sign(true_mean_residual)
                ),
                "shifted_direction_correct": bool(
                    direction_eligible and np.sign(shifted_offset) == np.sign(true_mean_residual)
                ),
                "base_rmse": _rmse(target, base),
                "selected_rmse": _rmse(target, base + selected_offset),
                "shifted_rmse": _rmse(target, base + shifted_offset),
                "oracle_rmse": _rmse(target, base + oracle_offset),
            }
        )
    paths = pd.concat(path_tables, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    paths.to_parquet(output_dir / "scored_paths.parquet", index=False)
    audit.to_csv(output_dir / "oracle_per_well.csv", index=False)
    target = paths["target_tvt"].to_numpy(dtype=np.float64)
    eligible = audit["direction_eligible"]
    qualified = audit["qualified"]
    shifted_qualified = audit["shifted_qualified"]
    well_ids = paths["well_id"].to_numpy()
    path_distributions = {
        "base": summarize_path_distribution(target, paths["base_tvt"], well_ids),
        "selected": summarize_path_distribution(target, paths["selected_tvt"], well_ids),
        "shifted": summarize_path_distribution(target, paths["shifted_control_tvt"], well_ids),
        "oracle_grid": summarize_path_distribution(target, paths["oracle_grid_tvt"], well_ids),
    }
    base_rmse = path_distributions["base"]["micro_rmse"]
    selected_rmse = path_distributions["selected"]["micro_rmse"]
    shifted_rmse = path_distributions["shifted"]["micro_rmse"]
    oracle_rmse = path_distributions["oracle_grid"]["micro_rmse"]
    selected_improvement = base_rmse - selected_rmse
    shifted_improvement = base_rmse - shifted_rmse
    true_vs_shifted_improvement = shifted_rmse - selected_rmse
    true_top5 = float(audit["true_top5"].mean())
    shifted_top5 = float(audit["shifted_top5"].mean())
    true_direction = float(audit.loc[eligible, "true_direction_correct"].mean())
    shifted_direction = float(audit.loc[eligible, "shifted_direction_correct"].mean())
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(audit["well_id"].nunique()),
        "rows": int(len(paths)),
        "grid_coverage": float(audit["grid_covered"].mean()),
        "evidence_well_fraction": float(audit["evidence_sufficient"].mean()),
        "true_gr_top1": float(audit["true_top1"].mean()),
        "true_gr_top3": float(audit["true_top3"].mean()),
        "true_gr_top5": true_top5,
        "shifted_gr_top1": float(audit["shifted_top1"].mean()),
        "shifted_gr_top3": float(audit["shifted_top3"].mean()),
        "shifted_gr_top5": shifted_top5,
        "qualified_true_gr_top1": float(audit.loc[qualified, "true_top1"].mean()),
        "qualified_true_gr_top3": float(audit.loc[qualified, "true_top3"].mean()),
        "qualified_true_gr_top5": float(audit.loc[qualified, "true_top5"].mean()),
        "qualified_shifted_gr_top1": float(audit.loc[shifted_qualified, "shifted_top1"].mean()),
        "qualified_shifted_gr_top3": float(audit.loc[shifted_qualified, "shifted_top3"].mean()),
        "qualified_shifted_gr_top5": float(audit.loc[shifted_qualified, "shifted_top5"].mean()),
        "true_m_rank_mean": float(audit["true_m_rank"].mean()),
        "true_m_rank_median": float(audit["true_m_rank"].median()),
        "shifted_true_m_rank_mean": float(audit["shifted_true_m_rank"].mean()),
        "shifted_true_m_rank_median": float(audit["shifted_true_m_rank"].median()),
        "true_direction_rate": true_direction,
        "shifted_direction_rate": shifted_direction,
        "base_rmse": base_rmse,
        "selected_rmse": selected_rmse,
        "shifted_rmse": shifted_rmse,
        "oracle_grid_rmse": oracle_rmse,
        "selected_improvement_ft": selected_improvement,
        "shifted_improvement_ft": shifted_improvement,
        "true_vs_shifted_improvement_ft": true_vs_shifted_improvement,
        "selected_well_win_rate": float(
            (audit["selected_rmse"] < audit["base_rmse"]).mean()
        ),
        "shifted_well_win_rate": float(
            (audit["shifted_rmse"] < audit["base_rmse"]).mean()
        ),
        "path_distributions": path_distributions,
    }
    for path_name, distribution in path_distributions.items():
        for metric_name in (
            "macro_well_rmse",
            "median_well_rmse",
            "p90_well_rmse",
            "worst_well_rmse",
        ):
            metrics[f"{path_name}_{metric_name}"] = distribution[metric_name]
    gates = {
        "grid_coverage_at_least_90pct": metrics["grid_coverage"] >= 0.90,
        "evidence_wells_at_least_80pct": metrics["evidence_well_fraction"] >= 0.80,
        "all_well_true_top5_at_least_25pct": true_top5 >= 0.25,
        "true_top5_advantage_at_least_10pp": true_top5 - shifted_top5 >= 0.10,
        "true_direction_at_least_60pct": bool(np.isfinite(true_direction) and true_direction >= 0.60),
        "direction_advantage_at_least_10pp": bool(
            np.isfinite(true_direction)
            and np.isfinite(shifted_direction)
            and true_direction - shifted_direction >= 0.10
        ),
        "selected_path_improves_at_least_0_10ft": selected_improvement >= 0.10,
        "shifted_path_improves_at_most_0_02ft": shifted_improvement <= 0.02,
        "true_path_beats_shifted_at_least_0_10ft": true_vs_shifted_improvement >= 0.10,
    }
    metrics["gates"] = {key: bool(value) for key, value in gates.items()}
    metrics["overall_pass"] = bool(all(gates.values()))
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    start = time.perf_counter()
    registry, shadow_ids = load_development_registry(
        CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv",
        CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    )
    outer_wells = sorted(registry.loc[registry["fold"].eq(0), "well_id"].astype(str))
    if set(outer_wells).intersection(shadow_ids) or len(outer_wells) != 131:
        raise RuntimeError("outer0 开发井集合不是无影子的 131 口井")
    if args.max_wells is not None:
        if args.max_wells <= 0:
            raise ValueError("--max-wells 必须为正整数")
        outer_wells = outer_wells[: args.max_wells]
    legal_outer = load_legal_outer_rows(outer_wells)
    expected_rows = int(
        registry.loc[registry["well_id"].astype(str).isin(outer_wells), "hidden_rows"].sum()
    )
    if args.max_wells is None and expected_rows != 651881:
        raise RuntimeError("完整 outer0 固定评价行数不是 651881")
    validate_legal_outer_contract(
        legal_outer,
        outer_wells,
        registry,
        expected_rows=expected_rows,
    )
    candidate_scores, selections = run_legal_stage(legal_outer, output_dir)
    metrics = score_after_legal_stage(
        legal_outer, candidate_scores, selections, output_dir
    )
    (output_dir / "runtime.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - start,
                "outer_source_sha256": OUTER_SOURCE_SHA256,
                "max_wells": args.max_wells,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
