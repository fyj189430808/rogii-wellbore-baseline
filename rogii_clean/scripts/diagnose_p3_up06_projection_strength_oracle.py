"""UP06：诊断每井二次投影强度是否是整段路径的主要剩余瓶颈。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_up01_robust_u_projection import robust_polynomial_projection  # noqa: E402
from src.p3_up06_projection_strength_oracle import (  # noqa: E402
    quadratic_sse_terms,
    sse_curve_from_terms,
)


EXPERIMENT_ID = "P3_UP06_projection_strength_oracle_audit_v1"
DEFAULT_CONFIG = CLEAN_ROOT / "configs" / "p3_up06_projection_strength_oracle_audit_v1.json"
DEFAULT_OUTPUT = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
KEYS = ["well_id", "fold", "row_index"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 UP06 投影强度 oracle 诊断")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    return path.resolve()


def shadow_well_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, usecols=lambda name: name in {"well_id", "is_shadow"})
    if "well_id" not in frame.columns:
        raise ValueError("shadow 登记表缺少 well_id")
    if "is_shadow" not in frame.columns:
        return set(frame["well_id"].astype(str))
    flags = frame["is_shadow"]
    if flags.dtype == bool:
        selected = flags.to_numpy()
    else:
        selected = flags.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
    return set(frame.loc[selected, "well_id"].astype(str))


def alpha_grid_from_config(config: dict[str, Any]) -> np.ndarray:
    settings = config["alpha_grid"]
    start = float(settings["start"])
    stop = float(settings["stop"])
    step = float(settings["step"])
    grid = np.round(np.arange(start, stop + step * 0.5, step), 10)
    if len(grid) != 26 or grid[0] != 0.0 or grid[-1] != 1.25:
        raise ValueError("UP06 alpha 网格必须固定为 [0,1.25]、步长 0.05")
    return grid


def load_legal_sources(config: dict[str, Any], excluded: set[str]) -> pd.DataFrame:
    up01_path = resolve_path(str(config["source_up01_legal_candidates"]))
    up01_manifest = read_json(resolve_path(str(config["source_up01_manifest"])))
    if up01_manifest.get("hidden_target_read") is not False:
        raise RuntimeError("UP01 候选血缘不合法")
    if file_sha256(up01_path) != str(up01_manifest["legal_candidates_sha256"]):
        raise RuntimeError("UP01 候选哈希与清单不一致")
    up01 = ds.dataset(up01_path, format="parquet").to_table(
        columns=KEYS + ["md", "p2_pred_tvt", "degree2_blend50_pred_tvt"]
    ).to_pandas()
    up01["well_id"] = up01["well_id"].astype(str)

    up03_parts: list[pd.DataFrame] = []
    for path_text in config["source_up03_legal_candidates"]:
        path = resolve_path(str(path_text))
        part = ds.dataset(path, format="parquet").to_table(
            columns=KEYS + ["candidate_pred_tvt"]
        ).to_pandas()
        part["well_id"] = part["well_id"].astype(str)
        up03_parts.append(part)
    up03 = pd.concat(up03_parts, ignore_index=True)

    for name, frame in (("UP01", up01), ("UP03", up03)):
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} 路径存在重复行键")
        if set(frame["well_id"]).intersection(excluded):
            raise RuntimeError(f"{name} 路径混入 shadow 井")
    merged = up01.merge(up03, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(up01) or len(merged) != len(up03):
        raise ValueError("UP01 与 UP03 行键不完全一致")
    merged = merged.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    return merged


def curvature_rms(md: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 3 or float(md[-1] - md[0]) <= 0.0:
        return 0.0
    if np.any(np.diff(md) <= 0.0):
        return float("nan")
    first_derivative = np.gradient(values, md)
    second_derivative = np.gradient(first_derivative, md)
    return float(np.sqrt(np.mean(second_derivative * second_derivative)))


def build_legal_candidates_and_features(
    sources: pd.DataFrame,
    raw_train_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_count = len(sources)
    z_all = np.empty(row_count, dtype=np.float64)
    p3_projected_all = np.empty(row_count, dtype=np.float64)
    up03_projected_all = np.empty(row_count, dtype=np.float64)
    feature_rows: list[dict[str, Any]] = []

    for well_number, (well_id, frame) in enumerate(sources.groupby("well_id", sort=False), 1):
        positions = frame.index.to_numpy(dtype=np.int64)
        horizontal_path = raw_train_dir / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path, usecols=["MD", "Z", "GR", "TVT_input"])
        row_indices = frame["row_index"].to_numpy(dtype=np.int64)
        selected = horizontal.iloc[row_indices]
        if selected["TVT_input"].notna().any():
            raise RuntimeError(f"{well_id} 含非自然隐藏行")
        md = frame["md"].to_numpy(dtype=np.float64)
        raw_md = selected["MD"].to_numpy(dtype=np.float64)
        if not np.allclose(md, raw_md, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"{well_id} 的 MD 与原始文件未对齐")
        z = selected["Z"].to_numpy(dtype=np.float64)
        gr_observed_rate = float(selected["GR"].notna().mean())
        z_all[positions] = z

        p3_base = frame["p2_pred_tvt"].to_numpy(dtype=np.float64)
        p3_blend50 = frame["degree2_blend50_pred_tvt"].to_numpy(dtype=np.float64)
        # UP01 已落盘的是 50% 融合，因此完整二次投影可精确反解为 2*blend50-base。
        p3_projected = 2.0 * p3_blend50 - p3_base
        p3_projected_all[positions] = p3_projected

        up03_base = frame["candidate_pred_tvt"].to_numpy(dtype=np.float64)
        up03_base_u = up03_base + z
        up03_projected_u = robust_polynomial_projection(md, up03_base_u, degree=2)
        up03_projected = up03_projected_u - z
        up03_projected_all[positions] = up03_projected

        common = {
            "well_id": str(well_id),
            "fold": int(frame["fold"].iloc[0]),
            "hidden_rows": int(len(frame)),
            "hidden_md_span_ft": float(md[-1] - md[0]) if len(md) > 1 else 0.0,
            "gr_observed_rate": gr_observed_rate,
        }
        for path_name, base_tvt, projected_tvt in (
            ("p3b00", p3_base, p3_projected),
            ("up03", up03_base, up03_projected),
        ):
            base_u = base_tvt + z
            projected_u = projected_tvt + z
            delta_u = projected_u - base_u
            row = dict(common)
            row.update(
                {
                    "path_name": path_name,
                    "projection_delta_rms_ft": float(np.sqrt(np.mean(delta_u * delta_u))),
                    "projection_delta_curvature_rms_per_ft2": curvature_rms(md, delta_u),
                    "projection_delta_endpoint_change_ft": float(delta_u[-1] - delta_u[0]),
                    "projection_delta_final_ft": float(delta_u[-1]),
                    "predicted_u_curvature_rms_per_ft2": curvature_rms(md, base_u),
                    "predicted_u_endpoint_change_ft": float(base_u[-1] - base_u[0]),
                }
            )
            feature_rows.append(row)
        if well_number % 50 == 0 or well_number == sources["well_id"].nunique():
            print(f"合法阶段：已处理 {well_number} / {sources['well_id'].nunique()} 口井", flush=True)

    legal = sources[KEYS + ["md"]].copy()
    legal["z"] = z_all
    legal["p3b00_base_tvt"] = sources["p2_pred_tvt"].to_numpy(dtype=np.float64)
    legal["p3b00_projected_tvt"] = p3_projected_all
    legal["up03_base_tvt"] = sources["candidate_pred_tvt"].to_numpy(dtype=np.float64)
    legal["up03_projected_tvt"] = up03_projected_all
    numeric_columns = legal.columns.difference(KEYS)
    if not np.isfinite(legal[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("合法候选包含 NaN 或无穷值")
    features = pd.DataFrame(feature_rows)
    if not np.isfinite(features.drop(columns=["well_id", "path_name"]).to_numpy(dtype=np.float64)).all():
        raise ValueError("合法井级量包含 NaN 或无穷值")
    return legal, features


def write_legal_stage(
    output: Path,
    config_path: Path,
    config: dict[str, Any],
    legal: pd.DataFrame,
    features: pd.DataFrame,
    source_hashes: dict[str, str],
) -> None:
    legal_path = output / "legal_candidates.parquet"
    temporary_legal = legal_path.with_suffix(".parquet.tmp")
    legal.to_parquet(temporary_legal, index=False)
    temporary_legal.replace(legal_path)
    feature_path = output / "legal_well_features.csv"
    temporary_feature = feature_path.with_suffix(".csv.tmp")
    features.to_csv(temporary_feature, index=False, lineterminator="\n")
    temporary_feature.replace(feature_path)
    shutil.copyfile(config_path, output / "config.json")
    write_json(
        output / "legal_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "legal_generation_complete": True,
            "hidden_target_read": False,
            "shadow_target_read": False,
            "shadow_overlap_wells": 0,
            "wells": int(legal["well_id"].nunique()),
            "rows": int(len(legal)),
            "folds": sorted(int(value) for value in legal["fold"].unique()),
            "paths": ["p3b00", "up03"],
            "alpha_grid": [float(value) for value in alpha_grid_from_config(config)],
            "legal_candidates_sha256": file_sha256(legal_path),
            "legal_well_features_sha256": file_sha256(feature_path),
            "source_hashes": source_hashes,
        },
    )


def load_development_targets(
    prediction_path: Path,
    excluded: set[str],
) -> pd.DataFrame:
    dataset = ds.dataset(prediction_path, format="parquet")
    row_filter = ~ds.field("well_id").isin(sorted(excluded))
    targets = dataset.to_table(columns=KEYS + ["target_tvt"], filter=row_filter).to_pandas()
    targets["well_id"] = targets["well_id"].astype(str)
    targets = targets.sort_values(["well_id", "row_index"], kind="stable").reset_index(drop=True)
    if set(targets["well_id"]).intersection(excluded):
        raise RuntimeError("目标读取混入 shadow 井")
    return targets


def spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    left_rank = left.loc[valid].rank(method="average")
    right_rank = right.loc[valid].rank(method="average")
    if float(left_rank.std()) == 0.0 or float(right_rank.std()) == 0.0:
        return float("nan")
    return float(left_rank.corr(right_rank, method="pearson"))


def score_oracle(
    legal: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    grid: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    if not legal[KEYS].equals(targets[KEYS]):
        raise ValueError("合法候选与目标行键不完全一致")
    truth_all = targets["target_tvt"].to_numpy(dtype=np.float64)
    output_frames: dict[str, list[dict[str, Any]]] = {
        "global_curve": [],
        "fold_curve": [],
        "fold_best": [],
        "per_well": [],
        "correlations": [],
    }
    metrics: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "model_training": False,
        "oracle_is_submittable": False,
        "shadow_target_read": False,
        "wells": int(legal["well_id"].nunique()),
        "rows": int(len(legal)),
        "alpha_grid": [float(value) for value in grid],
        "paths": {},
    }

    for path_name in ("p3b00", "up03"):
        base_column = f"{path_name}_base_tvt"
        projected_column = f"{path_name}_projected_tvt"
        global_terms = [0.0, 0.0, 0.0]
        fold_terms = {fold: [0.0, 0.0, 0.0, 0] for fold in range(5)}
        per_well_temporary: list[dict[str, Any]] = []

        for (well_id, fold), frame in legal.groupby(["well_id", "fold"], sort=True):
            positions = frame.index.to_numpy(dtype=np.int64)
            target = truth_all[positions]
            base = frame[base_column].to_numpy(dtype=np.float64)
            projected = frame[projected_column].to_numpy(dtype=np.float64)
            terms = quadratic_sse_terms(target, base, projected)
            sse_curve = sse_curve_from_terms(terms, grid)
            best_index = int(np.argmin(sse_curve))
            for index in range(3):
                global_terms[index] += terms[index]
                fold_terms[int(fold)][index] += terms[index]
            fold_terms[int(fold)][3] += len(frame)
            per_well_temporary.append(
                {
                    "well_id": str(well_id),
                    "fold": int(fold),
                    "path_name": path_name,
                    "rows": int(len(frame)),
                    "oracle_alpha": float(grid[best_index]),
                    "base_rmse": float(np.sqrt(terms[0] / len(frame))),
                    "oracle_rmse": float(np.sqrt(sse_curve[best_index] / len(frame))),
                    "oracle_sse": float(sse_curve[best_index]),
                    "base_sse": float(terms[0]),
                    "cross_term": float(terms[1]),
                    "direction_sse": float(terms[2]),
                }
            )

        global_sse_curve = sse_curve_from_terms(tuple(global_terms), grid)
        global_rmse_curve = np.sqrt(global_sse_curve / float(len(legal)))
        global_best_index = int(np.argmin(global_sse_curve))
        global_best_alpha = float(grid[global_best_index])
        for alpha, rmse_value in zip(grid, global_rmse_curve, strict=True):
            output_frames["global_curve"].append(
                {"path_name": path_name, "alpha": float(alpha), "micro_rmse": float(rmse_value)}
            )

        per_well = pd.DataFrame(per_well_temporary)
        fixed_sse = (
            per_well["base_sse"]
            - 2.0 * global_best_alpha * per_well["cross_term"]
            + global_best_alpha * global_best_alpha * per_well["direction_sse"]
        ).clip(lower=0.0)
        per_well["global_fixed_alpha"] = global_best_alpha
        per_well["global_fixed_rmse"] = np.sqrt(fixed_sse / per_well["rows"])
        per_well["fixed_minus_oracle_rmse_ft"] = (
            per_well["global_fixed_rmse"] - per_well["oracle_rmse"]
        )
        output_frames["per_well"].extend(per_well.to_dict(orient="records"))

        per_fold_best: list[dict[str, Any]] = []
        for fold in range(5):
            terms_with_rows = fold_terms[fold]
            fold_curve = sse_curve_from_terms(tuple(terms_with_rows[:3]), grid)
            fold_rmse_curve = np.sqrt(fold_curve / float(terms_with_rows[3]))
            best_index = int(np.argmin(fold_curve))
            per_fold_best.append(
                {
                    "fold": fold,
                    "best_alpha": float(grid[best_index]),
                    "best_micro_rmse": float(fold_rmse_curve[best_index]),
                    "rows": int(terms_with_rows[3]),
                }
            )
            output_frames["fold_best"].append(
                {
                    "path_name": path_name,
                    "fold": fold,
                    "best_alpha": float(grid[best_index]),
                    "best_micro_rmse": float(fold_rmse_curve[best_index]),
                    "rows": int(terms_with_rows[3]),
                }
            )
            for alpha, rmse_value in zip(grid, fold_rmse_curve, strict=True):
                output_frames["fold_curve"].append(
                    {
                        "path_name": path_name,
                        "fold": fold,
                        "alpha": float(alpha),
                        "micro_rmse": float(rmse_value),
                    }
                )

        oracle_micro_rmse = float(np.sqrt(per_well["oracle_sse"].sum() / per_well["rows"].sum()))
        global_best_rmse = float(global_rmse_curve[global_best_index])
        adaptive_potential = global_best_rmse - oracle_micro_rmse

        path_features = features.loc[features["path_name"] == path_name].merge(
            per_well[["well_id", "fold", "oracle_alpha"]],
            on=["well_id", "fold"],
            how="inner",
            validate="one_to_one",
        )
        legal_feature_names = [
            "hidden_rows",
            "hidden_md_span_ft",
            "gr_observed_rate",
            "projection_delta_rms_ft",
            "projection_delta_curvature_rms_per_ft2",
            "projection_delta_endpoint_change_ft",
            "projection_delta_final_ft",
            "predicted_u_curvature_rms_per_ft2",
            "predicted_u_endpoint_change_ft",
        ]
        correlation_rows: list[dict[str, Any]] = []
        for feature_name in legal_feature_names:
            pooled_rho = spearman_without_scipy(path_features["oracle_alpha"], path_features[feature_name])
            fold_rhos: list[float] = []
            for fold in range(5):
                fold_frame = path_features.loc[path_features["fold"] == fold]
                fold_rhos.append(
                    spearman_without_scipy(fold_frame["oracle_alpha"], fold_frame[feature_name])
                )
            finite_fold_rhos = [value for value in fold_rhos if np.isfinite(value)]
            if np.isfinite(pooled_rho) and pooled_rho != 0.0:
                same_direction = sum(np.sign(value) == np.sign(pooled_rho) for value in finite_fold_rhos)
            else:
                same_direction = 0
            correlation_rows.append(
                {
                    "path_name": path_name,
                    "feature": feature_name,
                    "pooled_spearman": pooled_rho,
                    **{f"fold_{fold}_spearman": fold_rhos[fold] for fold in range(5)},
                    "folds_same_direction": int(same_direction),
                }
            )
        output_frames["correlations"].extend(correlation_rows)
        correlation_frame = pd.DataFrame(correlation_rows)
        id_rule = config["identifiability_rule"]
        identifiable_rows = correlation_frame.loc[
            correlation_frame["pooled_spearman"].abs()
            >= float(id_rule["minimum_absolute_pooled_spearman"])
        ]
        identifiable_rows = identifiable_rows.loc[
            identifiable_rows["folds_same_direction"]
            >= int(id_rule["minimum_folds_same_direction"])
        ]

        alpha_counts = per_well["oracle_alpha"].value_counts().sort_index()
        metrics["paths"][path_name] = {
            "base_alpha0_micro_rmse": float(global_rmse_curve[0]),
            "full_projection_alpha1_micro_rmse": float(
                global_rmse_curve[int(np.where(np.isclose(grid, 1.0))[0][0])]
            ),
            "global_best_alpha": global_best_alpha,
            "global_best_fixed_micro_rmse": global_best_rmse,
            "per_well_oracle_micro_rmse": oracle_micro_rmse,
            "fixed_minus_per_well_oracle_ft": adaptive_potential,
            "has_at_least_0_30ft_adaptive_potential": bool(
                adaptive_potential >= float(config["potential_threshold_ft"])
            ),
            "per_fold_best": per_fold_best,
            "oracle_alpha_distribution": {
                f"{float(alpha):.2f}": int(count) for alpha, count in alpha_counts.items()
            },
            "oracle_alpha_at_lower_boundary_fraction": float((per_well["oracle_alpha"] == grid[0]).mean()),
            "oracle_alpha_at_upper_boundary_fraction": float((per_well["oracle_alpha"] == grid[-1]).mean()),
            "max_absolute_legal_feature_spearman": float(
                correlation_frame["pooled_spearman"].abs().max()
            ),
            "identifiable_by_preregistered_rule": bool(len(identifiable_rows) > 0),
            "identifiable_features": identifiable_rows["feature"].tolist(),
        }

    return metrics, {name: pd.DataFrame(rows) for name, rows in output_frames.items()}


def conclusion_text(metrics: dict[str, Any]) -> str:
    lines = [
        "# P3-UP06 全井二次投影强度 oracle 诊断",
        "",
        "> 本实验使用隐藏真值逐井选择 alpha，只是诊断上限，绝对不能提交，也不能作为正式特征。",
        "",
        "数据直接证明的事实：",
        "",
        f"- 严格排除 116 口 shadow 后，只评估了 {metrics['wells']} 口开发井、{metrics['rows']:,} 行。",
    ]
    for path_name, values in metrics["paths"].items():
        lines.extend(
            [
                f"- {path_name}：alpha=0 为 {values['base_alpha0_micro_rmse']:.6f} ft；"
                f"全局最佳固定 alpha={values['global_best_alpha']:.2f} 时为 "
                f"{values['global_best_fixed_micro_rmse']:.6f} ft；逐井 oracle 为 "
                f"{values['per_well_oracle_micro_rmse']:.6f} ft。",
                f"- {path_name}：固定 alpha 到逐井 oracle 还差 "
                f"{values['fixed_minus_per_well_oracle_ft']:.6f} ft；"
                f"是否达到 0.30 ft 潜力：{'是' if values['has_at_least_0_30ft_adaptive_potential'] else '否'}。",
                f"- {path_name}：合法井级量最大 |Spearman|="
                f"{values['max_absolute_legal_feature_spearman']:.4f}；"
                f"按预登记规则是否可辨识：{'是' if values['identifiable_by_preregistered_rule'] else '否'}。",
            ]
        )
    lines.extend(
        [
            "",
            "基于事实的合理推断：",
            "",
            "- 只有同时存在至少 0.30 ft 的逐井自适应上限，并且 oracle alpha 与合法井级量呈跨折一致相关，才值得进入严格 OOF 的强度学习。",
            "",
            "仍然没有验证的猜测：",
            "",
            "- 本诊断没有证明一个正式模型能达到 oracle，也没有搜索网格外的 alpha。",
            "",
            "当前实验只能否定的具体实现：",
            "",
            "- 若潜力不足，只能否定当前固定 Huber 二次投影方向上的逐井强度学习；不能否定其他路径表示。",
            "",
            "下一步最便宜的验证：",
            "",
            "- 若潜力和可辨识性同时成立，再做严格 OOF 的小型 alpha 模型；否则不训练，转向新的路径方向。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("配置实验编号不匹配")
    if config.get("diagnostic_only") is not True or config.get("model_training") is not False:
        raise ValueError("UP06 只允许诊断，禁止训练模型")

    output = args.output_dir.resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "oracle" / "metrics.json").exists() and not args.force:
        print(f"UP06 已完成：{output / 'oracle' / 'metrics.json'}")
        return

    start = time.perf_counter()
    grid = alpha_grid_from_config(config)
    shadow_path = resolve_path(str(config["shadow_registry"]))
    excluded = shadow_well_ids(shadow_path)
    if len(excluded) != 116:
        raise ValueError("shadow 井数不是冻结的 116")

    (output / "experiment_card.md").write_text(
        "# P3-UP06 实验卡\n\n"
        "唯一假设：全井二次投影剩余误差主要来自每口井所需投影强度不同。\n\n"
        "合法输入：P3B00、UP01 已落盘二次投影关系、UP03、MD/Z/GR 与自然隐藏掩码。\n\n"
        "oracle：alpha 固定网格 [0,1.25]、步长 0.05；逐井使用隐藏真值选 alpha，绝不可提交。\n\n"
        "判断标准：全局固定 alpha 到逐井 oracle 至少还有 0.30 ft，且至少一个合法量的 "
        "|Spearman|>=0.15 并在至少 4/5 折同方向，才值得训练严格 OOF alpha 模型。\n\n"
        "模型训练：无。影子集目标：不读取。registry/current_state：不修改。\n",
        encoding="utf-8",
    )
    print("阶段 A：只读合法路径、MD/Z/GR，生成投影和合法井级量；此时不读取 target_tvt。", flush=True)
    sources = load_legal_sources(config, excluded)
    if sources["well_id"].nunique() != int(config["development_wells"]):
        raise ValueError("开发井数不是冻结的 657")
    raw_train_dir = resolve_path(str(config["raw_train_dir"]))
    legal, features = build_legal_candidates_and_features(sources, raw_train_dir)
    # 合法阶段绝不触碰含 target_tvt 的 predictions.parquet，连整文件哈希也延后到 oracle 阶段。
    source_paths = [
        resolve_path(str(config["source_up01_legal_candidates"])),
        *[resolve_path(str(value)) for value in config["source_up03_legal_candidates"]],
    ]
    source_hashes = {str(path): file_sha256(path) for path in source_paths}
    write_legal_stage(output, config_path, config, legal, features, source_hashes)
    print("阶段 A 完成：合法候选和合法井级量已落盘。现在才允许进入独立 oracle 目录读取开发真值。", flush=True)

    print("阶段 B：读取 657 口开发井真值，只计算 alpha 上限和相关性，不训练模型。", flush=True)
    prediction_path = resolve_path(str(config["source_predictions"]))
    targets = load_development_targets(prediction_path, excluded)
    metrics, frames = score_oracle(legal, features, targets, grid, config)
    oracle_dir = output / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    frames["global_curve"].to_csv(oracle_dir / "global_alpha_curve.csv", index=False)
    frames["fold_curve"].to_csv(oracle_dir / "per_fold_alpha_curve.csv", index=False)
    frames["fold_best"].to_csv(oracle_dir / "per_fold_best_alpha.csv", index=False)
    frames["per_well"].to_csv(oracle_dir / "per_well_oracle.csv", index=False)
    frames["correlations"].to_csv(oracle_dir / "oracle_alpha_legal_feature_spearman.csv", index=False)
    write_json(oracle_dir / "metrics.json", metrics)
    (output / "conclusion.md").write_text(conclusion_text(metrics), encoding="utf-8")
    write_json(
        output / "runtime.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "elapsed_seconds": float(time.perf_counter() - start),
            "model_training": False,
            "legal_stage_completed_before_target_read": True,
            "shadow_target_read": False,
            "oracle_output_directory": str(oracle_dir),
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
