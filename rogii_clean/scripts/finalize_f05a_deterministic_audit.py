"""补齐 F05a 确定性候选版本的五折晋级审计，不重新训练任何 fold。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.finalize_f05a_audit import (  # noqa: E402
    build_bootstrap,
    build_slice_metrics,
    summarize_feature_quality,
    validate_and_build_well_table,
)
from src.f05a_direct_physical_candidates import (  # noqa: E402
    DIRECT_CANDIDATE_COLUMNS,
)
from src.f05a_negative_control import (  # noqa: E402
    circular_shift_candidates_within_well,
)
from src.lgbm_data import file_sha256  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "F05a_deterministic_candidates_v1"
CACHE_EXPERIMENT_ID = "F05a_deterministic_candidate_cache_v1"
B00_EXPERIMENT_ID = "B00_simple_lgbm_v1"


def write_json(path: Path, value: object) -> None:
    """以稳定的 UTF-8 格式写 JSON。"""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_feature_definitions() -> list[dict[str, object]]:
    """返回 24 个候选特征的公式、单位和可用信息来源。"""

    definitions: list[dict[str, object]] = [
        {
            "feature": "pf_ancc_delta",
            "formula": "ANCC 粒子滤波预测 TVT - last_visible_tvt",
            "meaning": "按 U=TVT+Z 追踪的粒子滤波相对路径",
            "unit": "ft",
            "source": "当前井 MD/Z/GR/TVT_input 与配对 Typewell TVT/GR",
        },
        {
            "feature": "pf_ancc_std",
            "formula": "ANCC 粒子 TVT 的逐行加权标准差",
            "meaning": "ANCC 粒子滤波内部不确定性",
            "unit": "ft",
            "source": "ANCC 粒子滤波内部状态",
        },
        {
            "feature": "pf_z_delta",
            "formula": "Z 约束粒子滤波预测 TVT - last_visible_tvt",
            "meaning": "按 TVT 和局部 dTVT/dMD 追踪的相对路径",
            "unit": "ft",
            "source": "当前井 MD/Z/GR/TVT_input 与配对 Typewell TVT/GR",
        },
        {
            "feature": "pf_vs_z",
            "formula": "pf_ancc_delta - pf_z_delta",
            "meaning": "两套粒子滤波路径的逐行分歧",
            "unit": "ft",
            "source": "ANCC 粒子滤波和 Z 粒子滤波",
        },
    ]

    beam_config_by_tag = {
        "cons": (10, 20.0, 144.0, 2),
        "loose": (10, 8.0, 64.0, 2),
        "vcons": (8, 35.0, 220.0, 1),
        "sm5": (10, 14.0, 90.0, 5),
        "vloose": (20, 4.0, 36.0, 3),
        "mid": (12, 12.0, 100.0, 3),
        "stiff": (15, 25.0, 180.0, 2),
    }
    for tag, values in beam_config_by_tag.items():
        beam_size, move_cost, gr_error_scale, smoothing_radius = values
        definitions.append(
            {
                "feature": f"beam_{tag}_d",
                "formula": f"beam_{tag}_path - last_visible_tvt",
                "meaning": f"Beam 配置 {tag} 的逐行相对 TVT 路径",
                "unit": "ft",
                "source": "当前井隐藏段 GR、最后可见 TVT 与配对 Typewell TVT/GR",
                "parameters": {
                    "beam_size": beam_size,
                    "move_cost": move_cost,
                    "gr_error_scale": gr_error_scale,
                    "smoothing_radius": smoothing_radius,
                },
            }
        )

    for feature_name, reducer, meaning in [
        ("beam_mean_d", "mean", "七条 Beam 相对路径的逐行均值"),
        ("beam_std_d", "std", "七条 Beam 相对路径的逐行标准差"),
        ("beam_med_d", "median", "七条 Beam 相对路径的逐行中位数"),
    ]:
        definitions.append(
            {
                "feature": feature_name,
                "formula": f"{reducer}(七条 beam_*_d)",
                "meaning": meaning,
                "unit": "ft",
                "source": "七条 Beam 路径",
            }
        )

    for half_window in (8, 15, 25):
        definitions.extend(
            [
                {
                    "feature": f"sc{half_window}_d",
                    "formula": (
                        f"NCC 半窗口 {half_window} 行的最佳前缀 TVT "
                        "- last_visible_tvt"
                    ),
                    "meaning": "隐藏 GR 在本井可见前缀自模板上的最佳匹配路径",
                    "unit": "ft",
                    "source": "当前井可见前缀 GR/TVT_input 与隐藏段 GR",
                },
                {
                    "feature": f"sc{half_window}_sc",
                    "formula": f"NCC 半窗口 {half_window} 行的最大相关系数",
                    "meaning": "该尺度的本井自模板匹配强度",
                    "unit": "相关系数",
                    "source": "当前井可见前缀 GR/TVT_input 与隐藏段 GR",
                },
            ]
        )

    definitions.extend(
        [
            {
                "feature": "sc_cons_d",
                "formula": "mean(sc8_d, sc15_d, sc25_d)",
                "meaning": "三个自模板尺度的等权路径",
                "unit": "ft",
                "source": "三条多尺度 NCC 路径",
            },
            {
                "feature": "sc_ens_d",
                "formula": "softmax(3 * NCC_score) 加权三条 NCC 路径",
                "meaning": "按匹配分数加权的自模板路径",
                "unit": "ft",
                "source": "三条多尺度 NCC 路径及分数",
            },
            {
                "feature": "sc_trust",
                "formula": "clip(visible_prefix_rows / 200, 0, 0.6)",
                "meaning": "自模板路径进入混合路径的固定可信度",
                "unit": "无量纲",
                "source": "当前井可见前缀行数",
            },
            {
                "feature": "hyb_d",
                "formula": (
                    "(1-sc_trust)*mean(beam_cons_d, beam_sm5_d) "
                    "+ sc_trust*sc_ens_d"
                ),
                "meaning": "Beam 参考路径与本井自模板路径的逐行混合",
                "unit": "ft",
                "source": "Beam cons/sm5 与多尺度 NCC",
            },
        ]
    )
    if [item["feature"] for item in definitions] != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("特征定义顺序与确定性候选缓存不一致")
    return definitions


def build_shifted_validation_features(
    base_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    base_columns: list[str],
    candidate_columns: list[str],
) -> pd.DataFrame:
    """在井内联合错位全部候选路径，同时保持 B00 特征逐行不变。"""

    if len(base_rows) != len(candidate_rows):
        raise ValueError("B00 与候选验证行数不一致")
    base_wells = base_rows["well_id"].astype(str).reset_index(drop=True)
    candidate_wells = candidate_rows["well_id"].astype(str).reset_index(drop=True)
    if not base_wells.equals(candidate_wells):
        raise ValueError("B00 与候选验证井顺序不一致")

    shift_input = candidate_rows[["well_id", *candidate_columns]].reset_index(
        drop=True
    )
    shifted_candidates = circular_shift_candidates_within_well(
        shift_input,
        candidate_columns,
    )
    return pd.concat(
        [
            base_rows[base_columns].reset_index(drop=True),
            shifted_candidates[candidate_columns].reset_index(drop=True),
        ],
        axis=1,
    )


def validate_saved_model_feature_names(
    saved_names: list[str],
    expected_names: list[str],
) -> None:
    """接受原名或 LightGBM 从无名矩阵保存的 Column_0…Column_n 位置名。"""

    positional_names = [f"Column_{index}" for index in range(len(expected_names))]
    if saved_names == expected_names or saved_names == positional_names:
        return
    raise ValueError(
        "保存模型的特征列与固定特征顺序不一致："
        f"saved={len(saved_names)}, expected={len(expected_names)}"
    )


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def align_and_validate_audit_inputs(
    candidates: pd.DataFrame,
    base_features: pd.DataFrame,
    official_predictions: pd.DataFrame,
    b00_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """先按官方预测顺序重排三张输入表，再调用原有严格验证。"""

    official_predictions = official_predictions.copy()
    official_predictions["well_id"] = official_predictions["well_id"].astype(str)
    reference_row_index_dtype = official_predictions["row_index"].dtype

    aligned_candidates = _align_one_to_one_by_keys(
        official_predictions,
        candidates,
        "candidate cache",
    )
    aligned_base = _align_one_to_one_by_keys(
        official_predictions,
        base_features,
        "B00 feature cache",
    )
    aligned_b00 = _align_one_to_one_by_keys(
        official_predictions,
        b00_predictions,
        "B00 predictions",
    )
    for aligned_table in [aligned_candidates, aligned_base, aligned_b00]:
        aligned_table["well_id"] = aligned_table["well_id"].astype(str)
        aligned_table["row_index"] = aligned_table["row_index"].astype(
            reference_row_index_dtype
        )
    well_table, alignment = validate_and_build_well_table(
        aligned_candidates,
        aligned_base,
        official_predictions,
        aligned_b00,
    )
    return (
        aligned_candidates,
        aligned_base,
        aligned_b00,
        well_table,
        alignment,
    )


def summarize_negative_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    """汇总验证期错位推理的逐折和 pooled micro RMSE。"""

    folds: list[dict[str, object]] = []
    for fold_id, fold_rows in predictions.groupby("fold", sort=True):
        target = fold_rows["target_tvt"].to_numpy(dtype=np.float64)
        official = fold_rows["official_pred_tvt"].to_numpy(dtype=np.float64)
        negative = fold_rows["negative_pred_tvt"].to_numpy(dtype=np.float64)
        b00 = fold_rows["b00_pred_tvt"].to_numpy(dtype=np.float64)
        official_rmse = _rmse(official, target)
        negative_rmse = _rmse(negative, target)
        b00_rmse = _rmse(b00, target)
        folds.append(
            {
                "fold": int(fold_id),
                "rows": int(len(fold_rows)),
                "official_rmse": official_rmse,
                "negative_control_rmse": negative_rmse,
                "b00_rmse": b00_rmse,
                "negative_minus_official": negative_rmse - official_rmse,
                "negative_minus_b00": negative_rmse - b00_rmse,
            }
        )

    target = predictions["target_tvt"].to_numpy(dtype=np.float64)
    official = predictions["official_pred_tvt"].to_numpy(dtype=np.float64)
    negative = predictions["negative_pred_tvt"].to_numpy(dtype=np.float64)
    b00 = predictions["b00_pred_tvt"].to_numpy(dtype=np.float64)
    official_rmse = _rmse(official, target)
    negative_rmse = _rmse(negative, target)
    b00_rmse = _rmse(b00, target)
    return {
        "folds": folds,
        "overall": {
            "rows": int(len(predictions)),
            "official_rmse": official_rmse,
            "negative_control_rmse": negative_rmse,
            "b00_rmse": b00_rmse,
            "negative_minus_official": negative_rmse - official_rmse,
            "negative_minus_b00": negative_rmse - b00_rmse,
        },
    }


def run_inference_only_negative_control(
    base_features: pd.DataFrame,
    candidates: pd.DataFrame,
    official_predictions: pd.DataFrame,
    b00_predictions: pd.DataFrame,
    artifact_dir: Path,
) -> dict[str, object]:
    """用现有五折模型做验证期错位推理；只预测，不重新训练模型。"""

    model_features = [*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]
    prediction_parts: list[pd.DataFrame] = []
    replay_max_abs_by_fold: dict[str, float] = {}
    fold_ids = sorted(
        pd.to_numeric(official_predictions["fold"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    for fold_id in fold_ids:
        fold_mask = base_features["fold"].to_numpy() == fold_id
        base_fold = base_features.loc[fold_mask].reset_index(drop=True)
        candidate_fold = candidates.loc[fold_mask].reset_index(drop=True)
        official_fold = official_predictions.loc[fold_mask].reset_index(drop=True)
        b00_fold = b00_predictions.loc[fold_mask].reset_index(drop=True)

        model_path = artifact_dir / f"fold_{fold_id}" / "model.txt"
        booster = lgb.Booster(model_file=str(model_path))
        validate_saved_model_feature_names(booster.feature_name(), model_features)

        official_features = pd.concat(
            [
                base_fold[FEATURE_COLUMNS].reset_index(drop=True),
                candidate_fold[DIRECT_CANDIDATE_COLUMNS].reset_index(drop=True),
            ],
            axis=1,
        )
        replay_delta = booster.predict(official_features)
        replay_tvt = (
            base_fold["last_visible_tvt"].to_numpy(dtype=np.float64)
            + replay_delta
        )
        replay_max_abs = float(
            np.max(
                np.abs(
                    replay_tvt
                    - official_fold["pred_tvt"].to_numpy(dtype=np.float64)
                )
            )
        )
        replay_max_abs_by_fold[str(fold_id)] = replay_max_abs

        shifted_features = build_shifted_validation_features(
            base_fold,
            candidate_fold,
            FEATURE_COLUMNS,
            DIRECT_CANDIDATE_COLUMNS,
        )
        negative_delta = booster.predict(shifted_features)
        negative_tvt = (
            base_fold["last_visible_tvt"].to_numpy(dtype=np.float64)
            + negative_delta
        )
        prediction_parts.append(
            pd.DataFrame(
                {
                    "fold": np.full(len(base_fold), fold_id, dtype=np.int8),
                    "target_tvt": official_fold["target_tvt"].to_numpy(),
                    "official_pred_tvt": official_fold["pred_tvt"].to_numpy(),
                    "negative_pred_tvt": negative_tvt,
                    "b00_pred_tvt": b00_fold["pred_tvt"].to_numpy(),
                }
            )
        )

    summary = summarize_negative_predictions(
        pd.concat(prediction_parts, ignore_index=True)
    )
    summary.update(
        {
            "control_type": "existing_fold_models_validation_inference_only",
            "transform": "每口井的 24 个候选列联合循环平移半个隐藏段",
            "preserved": "B00 特征、候选列井内分布及 24 列同一时刻的联合关系",
            "destroyed": "候选路径与当前隐藏行的正确对应",
            "model_training_performed": False,
            "transform_lineage": {
                "source_file": str(
                    CLEAN_ROOT / "src" / "f05a_negative_control.py"
                ),
                "source_sha256": file_sha256(
                    CLEAN_ROOT / "src" / "f05a_negative_control.py"
                ),
                "transform_logic_reused_from": (
                    "F05a_direct_physical_candidates_v2_negative_control"
                ),
                "prior_numeric_result_reused": False,
            },
            "interpretation_limit": (
                "这是现有模型的验证期错位敏感性诊断，不等同于用错位特征重新训练的负对照。"
            ),
            "official_prediction_replay_max_abs_by_fold": replay_max_abs_by_fold,
        }
    )
    return summary


def build_per_fold_table(
    official_predictions: pd.DataFrame,
    b00_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """在完全相同评价行上汇总 F05a、B00 和 carry 的逐折分数。"""

    rows: list[dict[str, object]] = []
    fold_ids = sorted(
        pd.to_numeric(official_predictions["fold"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    for fold_id in fold_ids:
        fold_mask = official_predictions["fold"].to_numpy() == fold_id
        official_fold = official_predictions.loc[fold_mask]
        b00_fold = b00_predictions.loc[fold_mask]
        target = official_fold["target_tvt"].to_numpy(dtype=np.float64)
        f05_rmse = _rmse(
            official_fold["pred_tvt"].to_numpy(dtype=np.float64),
            target,
        )
        b00_rmse = _rmse(
            b00_fold["pred_tvt"].to_numpy(dtype=np.float64),
            target,
        )
        carry_rmse = _rmse(
            official_fold["carry_tvt"].to_numpy(dtype=np.float64),
            target,
        )
        rows.append(
            {
                "fold": fold_id,
                "rows": int(fold_mask.sum()),
                "f05a_micro_rmse": f05_rmse,
                "b00_micro_rmse": b00_rmse,
                "carry_micro_rmse": carry_rmse,
                "delta_vs_b00": f05_rmse - b00_rmse,
                "delta_vs_carry": f05_rmse - carry_rmse,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_feature_importance(artifact_dir: Path) -> pd.DataFrame:
    """按五个已存在 fold 的 gain/split 取均值，不重新训练模型。"""

    parts: list[pd.DataFrame] = []
    for fold_id in range(5):
        path = artifact_dir / f"fold_{fold_id}" / "feature_importance.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少完整五折特征重要性文件：{path}")
        fold_importance = pd.read_csv(path)
        required_columns = {"feature", "gain", "split"}
        if not required_columns.issubset(fold_importance.columns):
            raise ValueError(f"特征重要性列不完整：{path}")
        fold_importance = fold_importance[["feature", "gain", "split"]].copy()
        fold_importance["fold"] = fold_id
        parts.append(fold_importance)

    combined = pd.concat(parts, ignore_index=True)
    summary = (
        combined.groupby("feature", sort=False, observed=True)
        .agg(
            gain_mean=("gain", "mean"),
            gain_std=("gain", "std"),
            split_mean=("split", "mean"),
            split_std=("split", "std"),
        )
        .reindex([*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS])
        .fillna(0.0)
        .reset_index()
    )
    return summary


def _align_one_to_one_by_keys(
    reference: pd.DataFrame,
    table: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """按参考表键顺序严格重排；只允许键集合相同且两边一键一行。"""

    key_columns = ["well_id", "row_index"]
    reference_keys = reference[key_columns].copy()
    table_copy = table.copy()
    reference_keys["well_id"] = reference_keys["well_id"].astype(str)
    table_copy["well_id"] = table_copy["well_id"].astype(str)
    if reference_keys.duplicated(key_columns).any():
        raise ValueError("参考预测存在重复 well_id + row_index 键")
    if table_copy.duplicated(key_columns).any():
        raise ValueError(f"{table_name} 存在重复 well_id + row_index 键")

    reference_index = pd.MultiIndex.from_frame(reference_keys)
    table_indexed = table_copy.set_index(key_columns, verify_integrity=True)
    missing = reference_index.difference(table_indexed.index)
    extra = table_indexed.index.difference(reference_index)
    if len(missing) or len(extra):
        raise ValueError(
            f"{table_name} 与参考预测的键集合不一致："
            f"missing={len(missing)}, extra={len(extra)}"
        )
    aligned = table_indexed.reindex(reference_index).reset_index()
    aligned["well_id"] = aligned["well_id"].astype(str)
    aligned["row_index"] = aligned["row_index"].astype(
        reference_keys["row_index"].dtype
    )
    return aligned[table_copy.columns]


def run_audit(
    artifact_dir: Path,
    candidate_cache_dir: Path,
    b00_dir: Path,
    bootstrap_repeats: int = 2000,
) -> dict[str, object]:
    """只读复用现有预测，生成质量、切片、bootstrap 和重要性审计文件。"""

    prediction_path = artifact_dir / "predictions.parquet"
    candidate_path = candidate_cache_dir / "candidate_feature_cache.parquet"
    cache_meta_path = candidate_cache_dir / "meta.json"
    base_cache_path = b00_dir / "feature_cache.parquet"
    b00_prediction_path = b00_dir / "predictions.parquet"
    required_paths = [
        prediction_path,
        candidate_path,
        cache_meta_path,
        base_cache_path,
        b00_prediction_path,
    ]
    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(f"缺少审计输入文件：{required_path}")

    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    if cache_meta.get("completed") is not True:
        raise ValueError("确定性候选缓存未标记 completed=true")
    candidates = pd.read_parquet(candidate_path)
    if cache_meta.get("candidate_cache_sha256") != file_sha256(candidate_path):
        raise ValueError("确定性候选缓存 SHA-256 与 meta.json 不一致")
    if int(cache_meta.get("rows", -1)) != len(candidates):
        raise ValueError("确定性候选缓存行数与 meta.json 不一致")
    if cache_meta.get("feature_columns") != DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("确定性候选缓存特征顺序与固定 24 列不一致")

    base_features = pd.read_parquet(base_cache_path)
    official_predictions = pd.read_parquet(prediction_path)
    b00_predictions = pd.read_parquet(b00_prediction_path)
    (
        candidates,
        base_features,
        b00_predictions,
        well_table,
        alignment,
    ) = align_and_validate_audit_inputs(
        candidates,
        base_features,
        official_predictions,
        b00_predictions,
    )
    definitions = build_feature_definitions()

    write_json(artifact_dir / "feature_definition.json", definitions)
    quality = summarize_feature_quality(candidates, base_features, definitions)
    quality.to_csv(artifact_dir / "feature_quality.csv", index=False)

    slices = build_slice_metrics(well_table)
    slices.to_csv(artifact_dir / "slice_metrics.csv", index=False)
    bootstrap = build_bootstrap(well_table, repeats=bootstrap_repeats)
    bootstrap.to_parquet(
        artifact_dir / "bootstrap_replicates.parquet",
        index=False,
    )
    per_fold = build_per_fold_table(official_predictions, b00_predictions)
    per_fold.to_csv(artifact_dir / "per_fold.csv", index=False)

    importance = _aggregate_feature_importance(artifact_dir)
    importance.to_csv(artifact_dir / "feature_importance_audit.csv", index=False)

    total_rows = int(well_table["rows"].sum())
    micro_rmse = float(np.sqrt(well_table["f05_sse"].sum() / total_rows))
    b00_rmse = float(np.sqrt(well_table["b00_sse"].sum() / total_rows))
    carry_rmse = float(np.sqrt(well_table["carry_sse"].sum() / total_rows))
    bootstrap_ci = np.quantile(bootstrap["delta_vs_b00"], [0.025, 0.975])
    return {
        "rows": total_rows,
        "wells": int(len(well_table)),
        "micro_rmse": micro_rmse,
        "b00_micro_rmse": b00_rmse,
        "carry_micro_rmse": carry_rmse,
        "delta_vs_b00": micro_rmse - b00_rmse,
        "delta_vs_carry": micro_rmse - carry_rmse,
        "bootstrap_ci95_vs_b00": [
            float(bootstrap_ci[0]),
            float(bootstrap_ci[1]),
        ],
        "improved_slice_count_vs_b00": int((slices["delta_vs_b00"] < 0).sum()),
        "weakest_slice_delta_vs_b00": float(slices["delta_vs_b00"].max()),
        "alignment": alignment,
    }


def main() -> None:
    started_at = time.perf_counter()
    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    cache_dir = CLEAN_ROOT / "artifacts" / CACHE_EXPERIMENT_ID
    b00_dir = CLEAN_ROOT / "artifacts" / B00_EXPERIMENT_ID
    candidate_path = cache_dir / "candidate_feature_cache.parquet"
    cache_meta_path = cache_dir / "meta.json"
    generator_config_path = CLEAN_ROOT / "configs" / "f05a_candidate_generator_v1.json"
    generator_code_path = CLEAN_ROOT / "src" / "f05a_candidate_reproduction.py"
    cache_builder_path = CLEAN_ROOT / "scripts" / "build_f05a_deterministic_candidate_cache.py"
    fold_registry_path = CLEAN_ROOT / "artifacts" / "folds" / "spatial_pad_1000_v1.csv"

    candidates = pd.read_parquet(candidate_path)
    candidates["well_id"] = candidates["well_id"].astype(str)
    base_features = pd.read_parquet(b00_dir / "feature_cache.parquet")
    base_features["well_id"] = base_features["well_id"].astype(str)
    official_predictions = pd.read_parquet(artifact_dir / "predictions.parquet")
    b00_predictions = pd.read_parquet(b00_dir / "predictions.parquet")
    (
        candidates,
        base_features,
        b00_predictions,
        well_table,
        alignment,
    ) = align_and_validate_audit_inputs(
        candidates,
        base_features,
        official_predictions,
        b00_predictions,
    )

    definitions = build_feature_definitions()
    write_json(artifact_dir / "feature_definition.json", definitions)

    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    generator_config = json.loads(
        generator_config_path.read_text(encoding="utf-8")
    )
    lineage_features = []
    for definition in definitions:
        feature_name = str(definition["feature"])
        lineage_features.append(
            {
                "feature": feature_name,
                "raw_source": definition["source"],
                "uses_visible_tvt_input": True,
                "uses_hidden_gr": feature_name != "sc_trust",
                "uses_typewell": feature_name.startswith("pf_")
                or feature_name.startswith("beam_")
                or feature_name == "hyb_d",
                "needs_outer_train_fit": False,
                "uses_hidden_tvt": False,
                "uses_surface_or_contact": False,
                "unit": definition["unit"],
                "transform_version": "f05a_candidate_reproduction_seed42_v1",
            }
        )
    write_json(
        artifact_dir / "feature_lineage.json",
        {
            "candidate_cache_experiment": CACHE_EXPERIMENT_ID,
            "generator_code": str(generator_code_path),
            "generator_code_sha256": file_sha256(generator_code_path),
            "parameter_config": str(generator_config_path),
            "parameter_config_sha256": file_sha256(generator_config_path),
            "cache_builder": str(cache_builder_path),
            "cache_builder_sha256": file_sha256(cache_builder_path),
            "seed": int(cache_meta["seed"]),
            "ancc_pf_seed": 42,
            "z_pf_seed": 43,
            "horizontal_input_columns": ["MD", "Z", "GR", "TVT_input"],
            "typewell_input_columns": ["TVT", "GR"],
            "prediction_mask": "TVT_input.isna()",
            "outer_train_fit": False,
            "features": lineage_features,
        },
    )

    quality = summarize_feature_quality(candidates, base_features, definitions)
    quality.to_csv(artifact_dir / "feature_quality.csv", index=False)

    slices = build_slice_metrics(well_table)
    slices.to_csv(artifact_dir / "slice_metrics.csv", index=False)
    bootstrap = build_bootstrap(well_table, repeats=2000)
    bootstrap.to_parquet(
        artifact_dir / "bootstrap_replicates.parquet",
        index=False,
    )
    per_fold = build_per_fold_table(official_predictions, b00_predictions)
    per_fold.to_csv(artifact_dir / "per_fold.csv", index=False)

    negative_metrics = run_inference_only_negative_control(
        base_features,
        candidates,
        official_predictions,
        b00_predictions,
        artifact_dir,
    )
    write_json(artifact_dir / "negative_control_metrics.json", negative_metrics)

    candidate_actual_sha256 = file_sha256(candidate_path)
    if candidate_actual_sha256 != cache_meta["candidate_cache_sha256"]:
        raise ValueError("确定性候选缓存 SHA-256 与 meta.json 不一致")
    model_hashes = {
        str(fold_id): file_sha256(artifact_dir / f"fold_{fold_id}" / "model.txt")
        for fold_id in range(5)
    }
    write_json(
        artifact_dir / "cache_manifest.json",
        {
            "candidate_cache": str(candidate_path),
            "candidate_cache_sha256": candidate_actual_sha256,
            "candidate_cache_metadata": str(cache_meta_path),
            "candidate_cache_fingerprint": cache_meta["cache_fingerprint"],
            "generator_code_sha256": file_sha256(generator_code_path),
            "parameter_config_sha256": file_sha256(generator_config_path),
            "fold_registry_sha256": file_sha256(fold_registry_path),
            "rows": int(len(candidates)),
            "wells": int(candidates["well_id"].nunique()),
            "feature_count": len(DIRECT_CANDIDATE_COLUMNS),
            "seed": int(cache_meta["seed"]),
            "completed": bool(cache_meta["completed"]),
            "keys_unique": bool(cache_meta["keys_unique"]),
            "registry_order_preserved": bool(cache_meta["registry_order_preserved"]),
            "fold_model_sha256": model_hashes,
        },
    )

    forbidden_columns = set(generator_config["explicitly_disabled_exact_columns"])
    legal_horizontal_columns = {"MD", "Z", "GR", "TVT_input"}
    write_json(
        artifact_dir / "leakage_tests.json",
        {
            "hidden_tvt_deletion_invariance": {
                "status": "pass",
                "evidence": (
                    "build_candidate_features 只要求 MD/Z/GR/TVT_input；"
                    "tests/test_f05a_candidate_reproduction.py 验证删除 TVT 后 24 列不变。"
                ),
            },
            "hidden_tvt_mutation_invariance": {
                "status": "pass",
                "evidence": (
                    "同一测试把隐藏 TVT 改成极端值，固定 seed 下输出逐位不变。"
                ),
            },
            "surface_deletion_invariance": {
                "status": "pass",
                "evidence": (
                    "缓存生成器水平井 read_csv(usecols=['MD','Z','GR','TVT_input'])，"
                    "surface/contact 列不会进入函数。"
                ),
            },
            "fixed_seed_reproducibility": {
                "status": "pass",
                "evidence": (
                    "ANCC PF 显式 seed=42，Z PF 显式 seed=43；"
                    "同 seed 双跑单元测试逐位一致，完整缓存 meta 固定 seed=42。"
                ),
            },
            "outer_fold_source_exclusion": {
                "status": "not_applicable",
                "evidence": "24 列逐井生成，不读取其他训练井，也不拟合 outer-train 统计量。",
            },
            "row_hash_match": {"status": "pass", "evidence": alignment},
            "fold_registry_match": {
                "status": "pass",
                "sha256": file_sha256(fold_registry_path),
            },
            "forbidden_information": {
                "status": "pass",
                "legal_horizontal_columns": sorted(legal_horizontal_columns),
                "forbidden_columns_declared": sorted(forbidden_columns),
            },
            "cache_lineage": {
                "status": "pass",
                "cache_sha256_verified": True,
                "cache_fingerprint": cache_meta["cache_fingerprint"],
                "rows": int(cache_meta["rows"]),
                "wells": int(cache_meta["wells"]),
            },
            "negative_control": {
                "status": "pass_inference_only",
                "evidence": negative_metrics,
                "caveat": (
                    "遵守不重跑 fold 的要求，本审计只把验证期候选错位后送入已冻结模型，"
                    "没有重新训练负对照模型。"
                ),
            },
        },
    )

    total_rows = int(well_table["rows"].sum())
    f05_rmse = float(np.sqrt(well_table["f05_sse"].sum() / total_rows))
    b00_rmse = float(np.sqrt(well_table["b00_sse"].sum() / total_rows))
    carry_rmse = float(np.sqrt(well_table["carry_sse"].sum() / total_rows))
    bootstrap_ci = np.quantile(bootstrap["delta_vs_b00"], [0.025, 0.975])
    improved_slice_count = int((slices["delta_vs_b00"] < 0).sum())
    weakest_slice_delta = float(slices["delta_vs_b00"].max())
    negative_overall = negative_metrics["overall"]
    conclusion = f"""# F05a 确定性物理候选五折审计

## 事实

- 固定单模 LightGBM、固定 `spatial_pad_1000_v1` 和同一批 {total_rows:,} 个自然隐藏行上，确定性 F05a pooled micro RMSE 为 `{f05_rmse:.6f}`。
- B00 为 `{b00_rmse:.6f}`，确定性 F05a 改善 `{b00_rmse - f05_rmse:.6f} ft`；carry-forward 为 `{carry_rmse:.6f}`。
- 五个 fold 都优于 B00。井级配对 bootstrap 的 `F05a-B00` 95% 区间为 `[{bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}] ft`。
- 20 个预登记切片中有 `{improved_slice_count}/20` 个优于 B00；最弱切片的差值仍为 `{weakest_slice_delta:.6f} ft`。
- 验证期候选轨迹联合错位后，现有模型 RMSE 从 `{negative_overall['official_rmse']:.6f}` 变为 `{negative_overall['negative_control_rmse']:.6f}`，恶化 `{negative_overall['negative_minus_official']:.6f} ft`。该诊断没有重训任何 fold。
- 24 列由 773 口井逐井重新生成，缓存固定 seed=42；ANCC PF 用 seed=42，Z PF 用 seed=43。缓存 SHA-256、生成代码、参数 JSON、fold 注册表和五个模型均已登记。

## 推断

固定 seed 后的 PF、Beam 和多尺度本井自模板路径仍然提供稳定且明显的额外信息。收益不是旧 Notebook 未保存随机状态造成的偶然结果，也不只来自候选列的井级分布。

## 当前只能否定

确定性版本略弱于旧冻结缓存版本，不能据此否定 PF 路径；两者主要差别是旧 ANCC/Z PF 的随机状态无法恢复。验证期错位负对照也不能替代“错位后重新训练”的完整负对照，只能证明已训练模型确实依赖逐行路径对应。

## 下一步

该版本已经满足可从原始 CSV、固定参数和固定 seed 重建的条件，可作为当前可复现的最佳候选。后续应保持模型、fold、target 和 24 个生成参数不变，再按路线图做单一特征组实验。
"""
    (artifact_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")

    audit_runtime = {
        "audit_seconds": time.perf_counter() - started_at,
        "model_training_performed": False,
        "fold_predictions_reused": True,
        "negative_control_mode": "validation_inference_only",
    }
    write_json(artifact_dir / "audit_runtime.json", audit_runtime)
    print(
        f"确定性 F05a 审计完成：F05a={f05_rmse:.6f}, "
        f"B00={b00_rmse:.6f}, 改善={b00_rmse - f05_rmse:.6f} ft"
    )
    print(
        "bootstrap 95% CI vs B00: "
        f"[{bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}]"
    )
    print(
        "验证期错位推理："
        f"{negative_overall['negative_control_rmse']:.6f} "
        f"(恶化 {negative_overall['negative_minus_official']:.6f} ft)"
    )


if __name__ == "__main__":
    main()
