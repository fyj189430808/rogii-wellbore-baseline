"""运行 P3-N01m：用严格 inner OOF 邻井平均残差修正 outer0 常数偏差。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
from scipy.spatial import cKDTree


CLEAN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_p3_n01a_neighbor_residual_audit import (  # noqa: E402
    _bbox,
    _bbox_distance,
    _typewell_tail_fingerprint,
)
from scripts.run_p3_pf02_target_ess_lgbm_cv import load_development_registry  # noqa: E402
from scripts.run_p3_r01b_mean_residual_ridge import (  # noqa: E402
    INNER_FOLDS,
    OUTER_FOLD,
    OUTER_FEATURE_COLUMNS,
    R01A_DIR,
    SOURCE_PREDICTION_PATHS,
    SOURCE_PREDICTION_SHA256,
    fit_mean_residual_target,
    read_outer_feature_predictions,
    verify_prediction_sources,
)
from src.p3_n01a_neighbor_residual_audit import (  # noqa: E402
    MAX_NEIGHBORS,
    MAX_DISTANCE_FT,
    geometry_is_eligible,
    neighbor_weight,
)
from src.p3_n01m_neighbor_mean_residual import (  # noqa: E402
    SHUFFLE_SEED,
    aggregate_neighbor_mean,
    build_within_fold_derangement,
    compute_cached_trajectory_geometry,
    select_top_neighbors,
    summarize_source_signal,
    trajectory_direction,
)
from src.p3_r01a_linear_residual import paired_well_bootstrap_delta  # noqa: E402


EXPERIMENT_ID = "P3_N01m_neighbor_mean_residual_v1"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
RAW_TRAIN = WORKSPACE_ROOT / "input/data/raw/train"
EXPECTED_SOURCE_WELLS = 526
EXPECTED_OUTER_WELLS = 131
EXPECTED_OUTER_ROWS = 651_881
TYPEWELL_TAIL_SPAN_FT = 500.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42

LEGAL_NEIGHBOR_COLUMNS = [
    "well_id",
    "fold",
    "neighbor_offset_ft",
    "shuffled_offset_ft",
    "global_offset_ft",
    "neighbor_count",
    "total_weight",
    "eta",
    "weighted_median_offset_ft",
    "shuffled_weighted_median_offset_ft",
    "nearest_neighbor_distance_ft",
    "source_well_ids_json",
    "source_weights_json",
]

SOURCE_PAIR_SIGNAL_COLUMNS = [
    "target_source_well_id",
    "neighbor_source_well_id",
    "distance_ft",
    "azimuth_difference_deg",
    "parallel_overlap_ft",
    "same_typewell_tail",
    "weight",
    "target_m",
    "neighbor_m",
    "shuffled_neighbor_m",
    "real_absdiff",
    "shuffled_absdiff",
]


@dataclass(frozen=True)
class SourceRecord:
    """一口严格 inner OOF 源井的标签目标和无标签空间信息。"""

    well_id: str
    fold: int
    hidden_rows: int
    hidden_xy: np.ndarray
    bbox: tuple[float, float, float, float]
    mean_residual: float
    shuffled_residual: float
    typewell_tail_fingerprint: str
    spatial_index: cKDTree
    direction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r01a-dir", type=Path, default=R01A_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-wells", type=int, default=None)
    return parser.parse_args()


def _resolve_output_dir(base_output: Path, max_wells: int | None) -> Path:
    """smoke 强制写入正式产物目录下的独立子目录。"""

    if max_wells is None:
        return base_output
    if int(max_wells) <= 0:
        raise ValueError("--max-wells 必须为正整数")
    return base_output / f"_smoke_max_wells_{int(max_wells)}"


def _run_scope_config(
    max_wells: int | None,
    actual_wells: int,
    actual_rows: int,
) -> dict[str, object]:
    """明确区分冻结正式合同与本次实际运行范围。"""

    return {
        "frozen": {
            "outer_wells": EXPECTED_OUTER_WELLS,
            "outer_rows": EXPECTED_OUTER_ROWS,
        },
        "actual": {
            "mode": "full" if max_wells is None else "smoke",
            "max_wells": max_wells,
            "outer_wells": int(actual_wells),
            "outer_rows": int(actual_rows),
        },
    }


def _write_json(path: Path, value: dict[str, object] | list[object]) -> None:
    def json_safe(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): json_safe(content) for key, content in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(content) for content in item]
        if isinstance(item, (np.floating, float)):
            numeric = float(item)
            return numeric if np.isfinite(numeric) else None
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_prediction_rows(
    name: str,
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    expected_wells: set[str],
    require_target: bool,
) -> None:
    """核对严格预测的井、fold、行键、有限值和每井自然隐藏行数。"""

    required = {"well_id", "fold", "row_index", "md", "pred_tvt"}
    if require_target:
        required.add("target_tvt")
    if missing := required.difference(predictions.columns):
        raise RuntimeError(f"{name} 缺列：{sorted(missing)}")
    predictions["well_id"] = predictions["well_id"].astype(str)
    if set(predictions["well_id"]) != expected_wells:
        raise RuntimeError(f"{name} 没有精确覆盖冻结井集合")
    if predictions.duplicated(["well_id", "row_index"]).any():
        raise RuntimeError(f"{name} 含重复自然隐藏行键")
    finite_columns = ["md", "pred_tvt", *( ["target_tvt"] if require_target else [])]
    if not np.isfinite(predictions[finite_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"{name} 路径或标签含 NaN/Inf")
    fold_by_well = registry.set_index("well_id")["fold"].astype(int).to_dict()
    observed_fold = predictions.groupby("well_id")["fold"].first().astype(int).to_dict()
    if any(observed_fold[well_id] != fold_by_well[well_id] for well_id in expected_wells):
        raise RuntimeError(f"{name} fold 与开发注册表不一致")
    expected_counts = (
        registry.loc[registry["well_id"].isin(expected_wells)]
        .set_index("well_id")["hidden_rows"]
        .astype(int)
        .to_dict()
    )
    actual_counts = predictions.groupby("well_id").size().astype(int).to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"{name} 每井行数与自然隐藏段不一致")


def _load_strict_predictions(
    source_dir: Path,
    registry: pd.DataFrame,
    shadow_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], set[str]]:
    """验证五份 SHA，并分别读取 526 源井标签与 outer0 无标签列。"""

    verify_prediction_sources(
        source_dir,
        source_paths=SOURCE_PREDICTION_PATHS,
        expected_hashes=SOURCE_PREDICTION_SHA256,
    )
    source_wells = set(
        registry.loc[registry["fold"].astype(int).isin(INNER_FOLDS), "well_id"]
    )
    outer_wells = set(
        registry.loc[registry["fold"].astype(int).eq(OUTER_FOLD), "well_id"]
    )
    if len(source_wells) != EXPECTED_SOURCE_WELLS or len(outer_wells) != EXPECTED_OUTER_WELLS:
        raise RuntimeError("开发井不是冻结的 526 source / 131 outer0")
    if source_wells & outer_wells or (source_wells | outer_wells) & shadow_ids:
        raise RuntimeError("source/outer0 相交或包含影子井")
    inner_parts: list[pd.DataFrame] = []
    for fold in INNER_FOLDS:
        path = source_dir / SOURCE_PREDICTION_PATHS[f"inner{fold}"]
        part = pd.read_parquet(path)
        if set(part["fold"].astype(int)) != {int(fold)}:
            raise RuntimeError(f"inner{fold} 文件出现其他 fold")
        inner_parts.append(part)
    inner = pd.concat(inner_parts, ignore_index=True)
    outer_path = source_dir / SOURCE_PREDICTION_PATHS["outer0"]
    outer_legal = read_outer_feature_predictions(outer_path)
    _validate_prediction_rows("inner OOF", inner, registry, source_wells, True)
    _validate_prediction_rows("outer0 legal", outer_legal, registry, outer_wells, False)
    if len(outer_legal) != EXPECTED_OUTER_ROWS or not outer_legal["fold"].astype(int).eq(0).all():
        raise RuntimeError("outer0 legal 不是固定 131 井/651881 行/fold0")
    return inner, outer_legal, source_wells, outer_wells


def _read_selected_outer_scoring_predictions(
    path: Path,
    selected_feature_predictions: pd.DataFrame,
    selected_wells: list[str],
) -> pd.DataFrame:
    """合法文件落盘后，用 Arrow 谓词只读取所选 outer 井及评分列。"""

    requested = sorted({str(well_id) for well_id in selected_wells})
    if not requested:
        raise ValueError("至少需要选择一口 outer 井")
    dataset = pds.dataset(str(path), format="parquet")
    required = [*OUTER_FEATURE_COLUMNS, "target_tvt"]
    if missing := set(required).difference(dataset.schema.names):
        raise RuntimeError(f"outer0 评分预测缺列：{sorted(missing)}")
    well_type = dataset.schema.field("well_id").type
    try:
        filter_values = pa.array(requested, type=well_type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError) as error:
        raise RuntimeError(f"无法把所选井号转换为 parquet well_id 类型 {well_type}") from error
    table = dataset.to_table(
        columns=required,
        filter=pds.field("well_id").isin(filter_values),
    )
    scoring = table.to_pandas()
    feature = selected_feature_predictions[OUTER_FEATURE_COLUMNS].copy()
    feature["well_id"] = feature["well_id"].astype(str)
    scoring["well_id"] = scoring["well_id"].astype(str)
    if set(feature["well_id"]) != set(requested) or set(scoring["well_id"]) != set(requested):
        raise RuntimeError("Arrow 过滤结果没有精确覆盖所选 outer 井")
    keys = ["well_id", "row_index"]
    if feature.duplicated(keys).any() or scoring.duplicated(keys).any():
        raise RuntimeError("所选 outer0 特征/评分预测行键重复")
    audit = feature.merge(
        scoring[[*keys, "fold", "md", "pred_tvt"]],
        on=keys,
        how="outer",
        suffixes=("_feature", "_score"),
        indicator=True,
        validate="one_to_one",
    )
    if len(audit) != len(feature) or not audit["_merge"].eq("both").all():
        raise RuntimeError("所选 outer0 二次读取行键与特征阶段不一致")
    for name in ["fold", "md", "pred_tvt"]:
        left = audit[f"{name}_feature"].to_numpy(dtype=np.float64)
        right = audit[f"{name}_score"].to_numpy(dtype=np.float64)
        if not np.array_equal(left, right):
            raise RuntimeError(f"所选 outer0 二次读取的 {name} 与特征阶段不一致")
    if not np.isfinite(scoring[["md", "pred_tvt", "target_tvt"]].to_numpy()).all():
        raise RuntimeError("所选 outer0 评分路径或标签含 NaN/Inf")
    return scoring


def _load_hidden_xy(well_id: str, expected_row_index: np.ndarray) -> np.ndarray:
    """只读 X/Y/TVT_input，并精确对齐严格预测自然隐藏行键。"""

    horizontal = pd.read_csv(
        RAW_TRAIN / f"{well_id}__horizontal_well.csv",
        usecols=["X", "Y", "TVT_input"],
    )
    hidden_positions = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    if not np.array_equal(hidden_positions, np.asarray(expected_row_index, dtype=np.int64)):
        raise RuntimeError(f"{well_id} raw 自然隐藏行与严格预测不一致")
    xy = horizontal.iloc[hidden_positions][["X", "Y"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xy).all() or len(xy) < 2:
        raise RuntimeError(f"{well_id} 隐藏 XY 非法")
    return xy


def _load_source_records(
    inner: pd.DataFrame,
    source_targets: pd.DataFrame,
    derangement: pd.DataFrame,
) -> dict[str, SourceRecord]:
    """把 526 口 inner OOF 目标和轨迹整理成只读源记录。"""

    target_by_well = source_targets.set_index("well_id")["mean_residual"].to_dict()
    donor_by_well = derangement.set_index("source_well_id")["donor_well_id"].to_dict()
    records: dict[str, SourceRecord] = {}
    for number, (well_id, well) in enumerate(inner.groupby("well_id", sort=True), start=1):
        well = well.sort_values("row_index")
        xy = _load_hidden_xy(
            str(well_id), well["row_index"].to_numpy(dtype=np.int64)
        )
        donor = donor_by_well[str(well_id)]
        records[str(well_id)] = SourceRecord(
            well_id=str(well_id),
            fold=int(well["fold"].iloc[0]),
            hidden_rows=int(len(well)),
            hidden_xy=xy,
            bbox=_bbox(xy),
            mean_residual=float(target_by_well[str(well_id)]),
            shuffled_residual=float(target_by_well[donor]),
            typewell_tail_fingerprint=_typewell_tail_fingerprint(
                RAW_TRAIN / f"{well_id}__typewell.csv", TYPEWELL_TAIL_SPAN_FT
            ),
            spatial_index=cKDTree(xy),
            direction=trajectory_direction(xy),
        )
        if number % 100 == 0 or number == len(source_targets):
            print(f"source records {number}/{len(source_targets)}", flush=True)
    return records


def _all_eligible_neighbors(
    target_well_id: str,
    target_xy: np.ndarray,
    target_fingerprint: str,
    sources: dict[str, SourceRecord],
) -> pd.DataFrame:
    """返回通过 N01a 冻结几何门槛的全部邻井，不执行 top8 截断。"""

    target_bbox = _bbox(target_xy)
    target_direction = trajectory_direction(target_xy)
    rows: list[dict[str, object]] = []
    for source in sources.values():
        if source.well_id == str(target_well_id):
            continue
        if source.fold not in INNER_FOLDS:
            raise RuntimeError("目标井邻居来源出现 outer0 或非法 fold")
        if _bbox_distance(target_bbox, source.bbox) > MAX_DISTANCE_FT:
            continue
        geometry = compute_cached_trajectory_geometry(
            target_xy,
            target_direction,
            source.hidden_xy,
            source.direction,
            source.spatial_index,
        )
        if not geometry_is_eligible(geometry):
            continue
        same_typewell = source.typewell_tail_fingerprint == target_fingerprint
        weight = neighbor_weight(
            geometry.minimum_distance_ft,
            geometry.azimuth_difference_deg,
            geometry.parallel_overlap_ft,
            same_typewell,
        )
        rows.append(
            {
                "source_well_id": source.well_id,
                "source_fold": source.fold,
                "mean_residual": source.mean_residual,
                "shuffled_residual": source.shuffled_residual,
                "weight": weight,
                "distance_ft": geometry.minimum_distance_ft,
                "azimuth_difference_deg": geometry.azimuth_difference_deg,
                "parallel_overlap_ft": geometry.parallel_overlap_ft,
                "same_typewell_tail": same_typewell,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "source_well_id", "source_fold", "mean_residual",
                "shuffled_residual", "weight", "distance_ft",
                "azimuth_difference_deg", "parallel_overlap_ft",
                "same_typewell_tail",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("source_well_id", kind="mergesort")
        .reset_index(drop=True)
    )


def _candidate_neighbors(
    target_well_id: str,
    target_xy: np.ndarray,
    target_fingerprint: str,
    sources: dict[str, SourceRecord],
) -> pd.DataFrame:
    """正式预测只在全部合格邻井中按冻结顺序取 top8。"""

    eligible = _all_eligible_neighbors(
        target_well_id, target_xy, target_fingerprint, sources
    )
    return select_top_neighbors(eligible, target_well_id, MAX_NEIGHBORS)


def _build_source_pair_signal(sources: dict[str, SourceRecord]) -> pd.DataFrame:
    """每个无向 source 井对只查一次距离，再分别生成两个方向的记录。"""

    rows: list[dict[str, object]] = []
    source_ids = sorted(sources)

    def append_direction(
        target: SourceRecord,
        neighbor: SourceRecord,
        geometry,
    ) -> None:
        if not geometry_is_eligible(geometry):
            return
        same_typewell = (
            target.typewell_tail_fingerprint == neighbor.typewell_tail_fingerprint
        )
        weight = neighbor_weight(
            geometry.minimum_distance_ft,
            geometry.azimuth_difference_deg,
            geometry.parallel_overlap_ft,
            same_typewell,
        )
        rows.append(
            {
                "target_source_well_id": target.well_id,
                "neighbor_source_well_id": neighbor.well_id,
                "distance_ft": float(geometry.minimum_distance_ft),
                "azimuth_difference_deg": float(geometry.azimuth_difference_deg),
                "parallel_overlap_ft": float(geometry.parallel_overlap_ft),
                "same_typewell_tail": bool(same_typewell),
                "weight": float(weight),
                "target_m": target.mean_residual,
                "neighbor_m": neighbor.mean_residual,
                "shuffled_neighbor_m": neighbor.shuffled_residual,
                "real_absdiff": abs(target.mean_residual - neighbor.mean_residual),
                "shuffled_absdiff": abs(
                    target.mean_residual - neighbor.shuffled_residual
                ),
            }
        )

    for position, first_id in enumerate(source_ids):
        first = sources[first_id]
        if first.fold not in INNER_FOLDS:
            raise RuntimeError("source-source 信号出现 outer0 或非法 fold")
        for second_id in source_ids[position + 1 :]:
            second = sources[second_id]
            if second.fold not in INNER_FOLDS:
                raise RuntimeError("source-source 信号出现 outer0 或非法 fold")
            if _bbox_distance(first.bbox, second.bbox) > MAX_DISTANCE_FT:
                continue
            symmetric_distance = float(
                second.spatial_index.query(first.hidden_xy, k=1)[0].min()
            )
            first_to_second = compute_cached_trajectory_geometry(
                first.hidden_xy,
                first.direction,
                second.hidden_xy,
                second.direction,
                second.spatial_index,
                minimum_distance_ft=symmetric_distance,
            )
            second_to_first = compute_cached_trajectory_geometry(
                second.hidden_xy,
                second.direction,
                first.hidden_xy,
                first.direction,
                first.spatial_index,
                minimum_distance_ft=symmetric_distance,
            )
            append_direction(first, second, first_to_second)
            append_direction(second, first, second_to_first)
    return pd.DataFrame(rows, columns=SOURCE_PAIR_SIGNAL_COLUMNS)


def _build_legal_predictions(
    outer_legal: pd.DataFrame,
    selected_outer_wells: list[str],
    sources: dict[str, SourceRecord],
    global_offset: float,
) -> pd.DataFrame:
    """只用无标签 outer0 轨迹生成三种井级修正，不读取 target_tvt。"""

    rows: list[dict[str, object]] = []
    for number, well_id in enumerate(selected_outer_wells, start=1):
        well = outer_legal.loc[outer_legal["well_id"].astype(str).eq(well_id)].sort_values("row_index")
        xy = _load_hidden_xy(well_id, well["row_index"].to_numpy(dtype=np.int64))
        fingerprint = _typewell_tail_fingerprint(
            RAW_TRAIN / f"{well_id}__typewell.csv", TYPEWELL_TAIL_SPAN_FT
        )
        neighbors = _candidate_neighbors(well_id, xy, fingerprint, sources)
        real = aggregate_neighbor_mean(
            neighbors["mean_residual"].to_numpy(dtype=np.float64),
            neighbors["weight"].to_numpy(dtype=np.float64),
        )
        shuffled = aggregate_neighbor_mean(
            neighbors["shuffled_residual"].to_numpy(dtype=np.float64),
            neighbors["weight"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "well_id": well_id,
                "fold": 0,
                "neighbor_offset_ft": real.predicted_offset_ft,
                "shuffled_offset_ft": shuffled.predicted_offset_ft,
                "global_offset_ft": float(global_offset),
                "neighbor_count": real.neighbor_count,
                "total_weight": real.total_weight,
                "eta": real.eta,
                "weighted_median_offset_ft": real.weighted_median_residual,
                "shuffled_weighted_median_offset_ft": shuffled.weighted_median_residual,
                "nearest_neighbor_distance_ft": (
                    float(neighbors["distance_ft"].min()) if len(neighbors) else float("nan")
                ),
                "source_well_ids_json": json.dumps(
                    neighbors["source_well_id"].astype(str).tolist(), separators=(",", ":")
                ),
                "source_weights_json": json.dumps(
                    neighbors["weight"].astype(float).tolist(), separators=(",", ":")
                ),
            }
        )
        print(f"legal neighbors {number}/{len(selected_outer_wells)}: {well_id}", flush=True)
    legal = pd.DataFrame(rows, columns=LEGAL_NEIGHBOR_COLUMNS)
    forbidden = [
        column for column in legal.columns
        if any(fragment in column.lower() for fragment in ("target", "true", "rmse", "oracle"))
    ]
    if forbidden:
        raise RuntimeError(f"合法邻井预测含 outer 评分列：{forbidden}")
    return legal


def _rmse_from_sse(sse: float, rows: int) -> float:
    return float(np.sqrt(float(sse) / int(rows)))


def _score(
    outer_scoring: pd.DataFrame,
    legal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """合法结果落盘后才使用 outer0 target，生成路径、切片、oracle 和八门槛。"""

    paths = outer_scoring.merge(legal, on=["well_id", "fold"], validate="many_to_one")
    paths["base_tvt"] = paths["pred_tvt"].to_numpy(dtype=np.float64)
    for label, offset_column in (
        ("candidate", "neighbor_offset_ft"),
        ("shuffled", "shuffled_offset_ft"),
        ("global", "global_offset_ft"),
    ):
        paths[f"{label}_tvt"] = (
            paths["base_tvt"].to_numpy(dtype=np.float64)
            + paths[offset_column].to_numpy(dtype=np.float64)
        )
    rows: list[dict[str, object]] = []
    for well_id, well in paths.groupby("well_id", sort=False):
        target = well["target_tvt"].to_numpy(dtype=np.float64)
        base = well["base_tvt"].to_numpy(dtype=np.float64)
        true_offset = float(np.mean(target - base))
        values: dict[str, object] = {
            "well_id": str(well_id),
            "fold": 0,
            "rows": int(len(well)),
            "true_mean_residual": true_offset,
            "neighbor_offset_ft": float(well["neighbor_offset_ft"].iloc[0]),
            "shuffled_offset_ft": float(well["shuffled_offset_ft"].iloc[0]),
            "global_offset_ft": float(well["global_offset_ft"].iloc[0]),
            "neighbor_count": int(well["neighbor_count"].iloc[0]),
            "nearest_neighbor_distance_ft": float(well["nearest_neighbor_distance_ft"].iloc[0]),
        }
        paths.loc[well.index, "oracle_tvt"] = base + true_offset
        for label, column in (
            ("base", "base_tvt"),
            ("candidate", "candidate_tvt"),
            ("shuffled", "shuffled_tvt"),
            ("global", "global_tvt"),
            ("oracle", "oracle_tvt"),
        ):
            error = target - paths.loc[well.index, column].to_numpy(dtype=np.float64)
            values[f"{label}_sse"] = float(error @ error)
            values[f"{label}_rmse"] = float(np.sqrt(np.mean(error**2)))
        values["candidate_offset_ae"] = abs(float(values["neighbor_offset_ft"]) - true_offset)
        values["shuffled_offset_ae"] = abs(float(values["shuffled_offset_ft"]) - true_offset)
        values["global_offset_ae"] = abs(float(values["global_offset_ft"]) - true_offset)
        eligible_direction = abs(true_offset) >= 2.0
        values["direction_eligible"] = eligible_direction
        values["candidate_direction_correct"] = bool(
            eligible_direction and np.sign(values["neighbor_offset_ft"]) == np.sign(true_offset)
        )
        values["shuffled_direction_correct"] = bool(
            eligible_direction and np.sign(values["shuffled_offset_ft"]) == np.sign(true_offset)
        )
        rows.append(values)
    per_well = pd.DataFrame(rows)
    total_rows = int(per_well["rows"].sum())

    def pooled(label: str, subset: pd.DataFrame = per_well) -> float:
        return _rmse_from_sse(float(subset[f"{label}_sse"].sum()), int(subset["rows"].sum()))

    pooled_rmse = {label: pooled(label) for label in ["base", "candidate", "shuffled", "global", "oracle"]}
    direction = per_well["direction_eligible"]
    candidate_direction = float(per_well.loc[direction, "candidate_direction_correct"].mean())
    shuffled_direction = float(per_well.loc[direction, "shuffled_direction_correct"].mean())
    near = per_well["nearest_neighbor_distance_ft"].le(1000.0)
    near_candidate = pooled("candidate", per_well.loc[near]) if near.any() else float("nan")
    near_shuffled = pooled("shuffled", per_well.loc[near]) if near.any() else float("nan")
    bootstrap = paired_well_bootstrap_delta(per_well, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    removal_count = int(math.ceil(len(per_well) * 0.05))
    gains = per_well["base_sse"] - per_well["candidate_sse"]
    remaining = per_well.drop(index=gains.sort_values(ascending=False).head(removal_count).index)
    remaining_base = pooled("base", remaining)
    remaining_candidate = pooled("candidate", remaining)
    candidate_mae = float(per_well["candidate_offset_ae"].mean())
    shuffled_mae = float(per_well["shuffled_offset_ae"].mean())
    global_mae = float(per_well["global_offset_ae"].mean())
    supported_fraction = float(per_well["neighbor_count"].gt(0).mean())
    path_distributions: dict[str, dict[str, float]] = {}
    for label in ["base", "candidate", "shuffled", "global", "oracle"]:
        well_values = per_well[f"{label}_rmse"].to_numpy(dtype=np.float64)
        path_distributions[label] = {
            "micro_rmse": pooled_rmse[label],
            "macro_well_rmse": float(np.mean(well_values)),
            "median_well_rmse": float(np.median(well_values)),
            "p90_well_rmse": float(np.quantile(well_values, 0.90)),
            "worst_well_rmse": float(np.max(well_values)),
            "well_win_rate_vs_base": float(
                np.mean(per_well[f"{label}_rmse"] < per_well["base_rmse"])
            ),
        }
    slice_definitions = {
        "nearest_le_500ft": per_well["nearest_neighbor_distance_ft"].le(500.0),
        "nearest_le_1000ft": per_well["nearest_neighbor_distance_ft"].le(1000.0),
        "nearest_1000_to_2500ft": per_well["nearest_neighbor_distance_ft"].gt(1000.0)
        & per_well["nearest_neighbor_distance_ft"].le(2500.0),
        "unsupported": per_well["neighbor_count"].eq(0),
    }
    spatial_slices: dict[str, dict[str, float | int | None]] = {}
    for slice_name, mask in slice_definitions.items():
        subset = per_well.loc[mask]
        if subset.empty:
            spatial_slices[slice_name] = {"wells": 0, "rows": 0}
            continue
        slice_base = pooled("base", subset)
        slice_candidate = pooled("candidate", subset)
        slice_shuffled = pooled("shuffled", subset)
        spatial_slices[slice_name] = {
            "wells": int(len(subset)),
            "rows": int(subset["rows"].sum()),
            "base_rmse": slice_base,
            "candidate_rmse": slice_candidate,
            "shuffled_rmse": slice_shuffled,
            "candidate_improvement_ft": slice_base - slice_candidate,
            "candidate_vs_shuffled_improvement_ft": slice_shuffled - slice_candidate,
        }
    positive_gain = np.maximum(
        per_well["base_sse"].to_numpy(dtype=np.float64)
        - per_well["candidate_sse"].to_numpy(dtype=np.float64),
        0.0,
    )
    strongest_gain = float(
        np.sum(np.sort(positive_gain)[::-1][:removal_count])
    )
    positive_gain_total = float(np.sum(positive_gain))
    strongest_gain_share = (
        strongest_gain / positive_gain_total if positive_gain_total > 0.0 else 0.0
    )
    gates = {
        "neighbor_coverage_at_least_50pct": supported_fraction >= 0.50,
        "candidate_improvement_at_least_0_10ft": pooled_rmse["base"] - pooled_rmse["candidate"] >= 0.10,
        "candidate_beats_shuffled_at_least_0_20ft": pooled_rmse["shuffled"] - pooled_rmse["candidate"] >= 0.20,
        "candidate_mae_at_least_10pct_better_than_shuffled": candidate_mae <= 0.90 * shuffled_mae,
        "candidate_mae_below_global": candidate_mae < global_mae,
        "direction_advantage_at_least_10pp": bool(
            np.isfinite(candidate_direction)
            and np.isfinite(shuffled_direction)
            and candidate_direction - shuffled_direction >= 0.10
        ),
        "nearest_1000ft_candidate_beats_shuffled": bool(
            np.isfinite(near_candidate) and near_candidate < near_shuffled
        ),
        "bootstrap_and_leave_top5pct_pass": bool(
            bootstrap["ci95_high"] < 0.0 and remaining_candidate < remaining_base
        ),
    }
    metrics: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "wells": int(len(per_well)),
        "rows": total_rows,
        "pooled_rmse": pooled_rmse,
        "candidate_improvement_ft": pooled_rmse["base"] - pooled_rmse["candidate"],
        "candidate_vs_shuffled_improvement_ft": pooled_rmse["shuffled"] - pooled_rmse["candidate"],
        "supported_well_fraction": supported_fraction,
        "candidate_offset_mae": candidate_mae,
        "shuffled_offset_mae": shuffled_mae,
        "global_offset_mae": global_mae,
        "candidate_direction_rate": candidate_direction,
        "shuffled_direction_rate": shuffled_direction,
        "nearest_1000ft_wells": int(near.sum()),
        "nearest_1000ft_candidate_rmse": near_candidate,
        "nearest_1000ft_shuffled_rmse": near_shuffled,
        "paired_well_bootstrap": bootstrap,
        "top5pct_removed_wells": removal_count,
        "remaining_base_rmse": remaining_base,
        "remaining_candidate_rmse": remaining_candidate,
        "strongest_5pct_positive_gain_share": strongest_gain_share,
        "path_distributions": path_distributions,
        "spatial_signal_slices": spatial_slices,
        "gates": {key: bool(value) for key, value in gates.items()},
        "overall_pass": bool(all(gates.values())),
    }
    return paths, per_well, metrics


def _conclusion(metrics: dict[str, object]) -> str:
    return (
        f"数据直接证明的事实：N01m 使用 {metrics['wells']} 口 outer0 井，支持率为 "
        f"{metrics['supported_well_fraction']:.1%}，八项门槛整体"
        f"{'通过' if metrics['overall_pass'] else '未通过'}。\n\n"
        "基于事实的合理推断：结果只评价冻结的加权中位数与 kappa=1 收缩。\n\n"
        "仍然没有验证的猜测：其他距离、角度、邻井数、裁剪或学习型权重是否有效。\n\n"
        "当前实验只能否定的具体实现：2500ft/45度/500ft/top8/固定权重/加权中位数/kappa=1。\n\n"
        "下一步最便宜的验证：严格按八项门槛决定是否为 outer1 生成嵌套资产。\n"
    )


def main() -> None:
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir.resolve(), args.max_wells).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    registry, shadow_ids = load_development_registry(
        CLEAN_ROOT / "artifacts/folds/balanced_well_5fold_v1.csv",
        CLEAN_ROOT / "artifacts/P3_shadow_holdout_v1/shadow_holdout.csv",
    )
    registry = registry.copy()
    registry["well_id"] = registry["well_id"].astype(str)
    source_dir = args.r01a_dir.resolve()
    inner, outer_legal, source_wells, outer_wells = _load_strict_predictions(
        source_dir, registry, shadow_ids
    )
    source_targets = fit_mean_residual_target(inner)
    if len(source_targets) != EXPECTED_SOURCE_WELLS or set(source_targets["well_id"]) != source_wells:
        raise RuntimeError("严格 inner OOF 没有生成 526 口唯一源目标")
    derangement = build_within_fold_derangement(
        source_targets[["well_id", "fold"]], seed=SHUFFLE_SEED
    )
    records = _load_source_records(inner, source_targets, derangement)
    source_table = source_targets.merge(
        derangement, left_on=["well_id", "fold"], right_on=["source_well_id", "fold"],
        validate="one_to_one",
    )
    source_table["hidden_rows"] = source_table["well_id"].map(
        {well_id: record.hidden_rows for well_id, record in records.items()}
    )
    source_table.to_csv(output_dir / "source_mean_residuals.csv", index=False)
    pair_signal = _build_source_pair_signal(records)
    pair_signal.to_csv(output_dir / "source_pair_signal.csv", index=False)
    _write_json(
        output_dir / "source_signal_metrics.json",
        summarize_source_signal(pair_signal),
    )
    global_offset = float(
        np.average(
            source_table["mean_residual"].to_numpy(dtype=np.float64),
            weights=source_table["hidden_rows"].to_numpy(dtype=np.float64),
        )
    )
    selected_outer_wells = sorted(outer_wells)
    if args.max_wells is not None:
        selected_outer_wells = selected_outer_wells[: int(args.max_wells)]
    selected_outer_features = outer_legal.loc[
        outer_legal["well_id"].astype(str).isin(selected_outer_wells)
    ].copy()
    legal = _build_legal_predictions(
        outer_legal, selected_outer_wells, records, global_offset
    )
    legal.to_csv(output_dir / "legal_neighbor_predictions.csv", index=False)

    # 合法结果已落盘，以下才允许第二次读取 outer0 target_tvt。
    outer_path = source_dir / SOURCE_PREDICTION_PATHS["outer0"]
    outer_scoring = _read_selected_outer_scoring_predictions(
        outer_path,
        selected_outer_features,
        selected_outer_wells,
    )
    actual_wells = int(outer_scoring["well_id"].astype(str).nunique())
    actual_rows = int(len(outer_scoring))
    if args.max_wells is None and (
        actual_wells != EXPECTED_OUTER_WELLS or actual_rows != EXPECTED_OUTER_ROWS
    ):
        raise RuntimeError("正式运行没有精确评分冻结的 131 井/651881 行")
    paths, per_well, metrics = _score(outer_scoring, legal)
    if args.max_wells is None and (
        int(metrics["wells"]) != EXPECTED_OUTER_WELLS
        or int(metrics["rows"]) != EXPECTED_OUTER_ROWS
    ):
        raise RuntimeError("正式指标没有精确覆盖冻结的 131 井/651881 行")
    paths.to_parquet(output_dir / "predictions.parquet", index=False)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    _write_json(output_dir / "metrics.json", metrics)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "outer_fold": 0,
        "source_wells": 526,
        "outer_wells": 131,
        "outer_rows": 651881,
        "source_prediction_sha256": SOURCE_PREDICTION_SHA256,
        "shuffle_seed": SHUFFLE_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "run_scope": _run_scope_config(args.max_wells, actual_wells, actual_rows),
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "config.json", config)
    _write_json(
        output_dir / "feature_list.json",
        ["hidden_XY_trajectory", "typewell_tail_500ft_fingerprint", "source_inner_oof_mean_residual"],
    )
    _write_json(
        output_dir / "parameter_list.json",
        {
            "maximum_distance_ft": 2500.0,
            "maximum_azimuth_difference_deg": 45.0,
            "minimum_parallel_overlap_ft": 500.0,
            "maximum_neighbors": 8,
            "distance_decay_ft": 1000.0,
            "overlap_full_weight_ft": 1000.0,
            "same_typewell_multiplier": 1.5,
            "eta_prior_weight": 1.0,
        },
    )
    _write_json(
        output_dir / "runtime.json",
        {"seconds": time.perf_counter() - start, "max_wells": args.max_wells},
    )
    (output_dir / "conclusion.md").write_text(_conclusion(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
