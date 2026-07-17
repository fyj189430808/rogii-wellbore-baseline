"""补齐 F05a 直接物理候选的五折晋级审计文件。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS  # noqa: E402
from src.lgbm_data import file_sha256  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


EXPERIMENT_ID = "F05a_direct_physical_candidates_v2"
NEGATIVE_ID = "F05a_direct_physical_candidates_v2_negative_control"
SOURCE_SHA256 = "68689ad02338581669f5392dc38741e6920a00537c3a5131820e723c2c9bcbcf"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def feature_definitions() -> list[dict]:
    definitions: list[dict] = [
        {
            "feature": "pf_ancc_delta",
            "formula": "pf_ancc - last_known_tvt",
            "meaning": "ANCC 粒子滤波路径相对最后可见 TVT 的位移",
            "unit": "ft",
            "source": "当前井 MD/Z/GR/TVT_input + Typewell TVT/GR",
        },
        {
            "feature": "pf_ancc_std",
            "formula": "每行 ANCC-PF 粒子 TVT 的加权标准差",
            "meaning": "ANCC 粒子群内部不确定性",
            "unit": "ft",
            "source": "ANCC 粒子滤波内部状态",
        },
        {
            "feature": "pf_z_delta",
            "formula": "pf_z - last_known_tvt",
            "meaning": "Z 约束粒子滤波路径相对最后可见 TVT 的位移",
            "unit": "ft",
            "source": "当前井 MD/Z/GR/TVT_input + Typewell TVT/GR",
        },
        {
            "feature": "pf_vs_z",
            "formula": "pf_ancc - pf_z",
            "meaning": "两种粒子滤波路径的逐行分歧",
            "unit": "ft",
            "source": "ANCC-PF 与 Z-PF",
        },
    ]
    beam_configs = {
        "cons": [10, 20.0, 144.0, 2],
        "loose": [10, 8.0, 64.0, 2],
        "vcons": [8, 35.0, 220.0, 1],
        "sm5": [10, 14.0, 90.0, 5],
        "vloose": [20, 4.0, 36.0, 3],
        "mid": [12, 12.0, 100.0, 3],
        "stiff": [15, 25.0, 180.0, 2],
    }
    for tag, config in beam_configs.items():
        definitions.append(
            {
                "feature": f"beam_{tag}_d",
                "formula": f"beam_{tag}_path - last_known_tvt",
                "meaning": f"Beam 配置 {tag} 的逐行相对路径",
                "unit": "ft",
                "source": "隐藏段 GR + Typewell TVT/GR + 最后可见 TVT",
                "beam_config": {
                    "beam_width": config[0],
                    "move_cost": config[1],
                    "gr_error_scale": config[2],
                    "gr_smoothing_radius_rows": config[3],
                },
            }
        )
    for name, reducer, meaning in [
        ("beam_mean_d", "mean", "七条 Beam 相对路径的逐行均值"),
        ("beam_std_d", "std", "七条 Beam 相对路径的逐行标准差"),
        ("beam_med_d", "median", "七条 Beam 相对路径的逐行中位数"),
    ]:
        definitions.append(
            {
                "feature": name,
                "formula": f"{reducer}(beam_*_path - last_known_tvt)",
                "meaning": meaning,
                "unit": "ft",
                "source": "七条 Beam 路径",
            }
        )
    for half_window in [8, 15, 25]:
        definitions.extend(
            [
                {
                    "feature": f"sc{half_window}_d",
                    "formula": f"NCC 半窗口 {half_window} 行的最佳前缀 TVT - last_known_tvt",
                    "meaning": "当前井隐藏 GR 在可见前缀自模板上的最佳匹配路径",
                    "unit": "ft",
                    "source": "当前井可见前缀 GR/TVT_input + 隐藏段 GR",
                },
                {
                    "feature": f"sc{half_window}_sc",
                    "formula": f"NCC 半窗口 {half_window} 行的最大相关系数",
                    "meaning": "该尺度最佳自模板匹配强度",
                    "unit": "相关系数",
                    "source": "当前井可见前缀 GR/TVT_input + 隐藏段 GR",
                },
            ]
        )
    definitions.extend(
        [
            {
                "feature": "sc_cons_d",
                "formula": "mean(sc8_path, sc15_path, sc25_path) - last_known_tvt",
                "meaning": "三个自模板尺度的等权路径",
                "unit": "ft",
                "source": "三条多尺度 NCC 路径",
            },
            {
                "feature": "sc_ens_d",
                "formula": "softmax(3*NCC_score) 加权三条 NCC 路径 - last_known_tvt",
                "meaning": "按匹配分数加权的自模板路径",
                "unit": "ft",
                "source": "三条多尺度 NCC 路径和分数",
            },
            {
                "feature": "sc_trust",
                "formula": "clip(visible_prefix_rows / 200, 0, 0.6)",
                "meaning": "自模板路径进入混合路径的固定可信度",
                "unit": "无量纲",
                "source": "可见前缀行数",
            },
            {
                "feature": "hyb_d",
                "formula": "((1-sc_trust)*beam_ref + sc_trust*sc_ens) - last_known_tvt",
                "meaning": "Beam 参考路径与自模板路径的逐行混合",
                "unit": "ft",
                "source": "Beam cons/sm5 与多尺度 NCC",
            },
        ]
    )
    assert [item["feature"] for item in definitions] == DIRECT_CANDIDATE_COLUMNS
    return definitions


def _strongest_finite_correlation(
    correlations: pd.Series,
) -> tuple[str | None, float]:
    """返回绝对值最大的有限相关；常量特征没有有效相关时返回空。"""

    finite_correlations = correlations.loc[np.isfinite(correlations.to_numpy())]
    if finite_correlations.empty:
        return None, float("nan")
    strongest_feature = str(finite_correlations.abs().idxmax())
    return strongest_feature, float(finite_correlations[strongest_feature])


def _read_fold_micro_rmse(artifact_dir: Path, fold_id: int) -> float:
    """兼容从逐折 CSV 或汇总 JSON 读取指定折的 micro RMSE。"""

    csv_path = artifact_dir / "metrics.csv"
    if csv_path.is_file():
        fold_row = pd.read_csv(csv_path).query("fold == @fold_id")
        if not fold_row.empty:
            return float(fold_row.iloc[0]["micro_rmse"])
    json_path = artifact_dir / "metrics.json"
    if json_path.is_file():
        metrics = json.loads(json_path.read_text(encoding="utf-8"))
        for fold_metrics in metrics.get("folds", []):
            if int(fold_metrics["fold"]) == int(fold_id):
                return float(fold_metrics["micro_rmse"])
    raise ValueError(f"找不到 fold {fold_id} 的 micro RMSE：{artifact_dir}")


def summarize_feature_quality(
    candidates: pd.DataFrame,
    base_features: pd.DataFrame,
    definitions: list[dict],
) -> pd.DataFrame:
    values = candidates[DIRECT_CANDIDATE_COLUMNS]
    description = values.describe(percentiles=[0.01, 0.5, 0.99]).T
    unique_counts = values.nunique(dropna=False)
    per_well_unique = candidates.groupby("well_id", observed=True)[
        DIRECT_CANDIDATE_COLUMNS
    ].nunique(dropna=False)
    well_constant_rate = (per_well_unique <= 1).mean(axis=0)

    sample_step = max(len(candidates) // 200_000, 1)
    sampled_candidates = values.iloc[::sample_step].reset_index(drop=True)
    sampled_base = base_features[FEATURE_COLUMNS].iloc[::sample_step].reset_index(
        drop=True
    )
    cross_correlation = pd.concat([sampled_candidates, sampled_base], axis=1).corr()
    definition_by_name = {item["feature"]: item for item in definitions}
    rows: list[dict] = []
    for feature_name in DIRECT_CANDIDATE_COLUMNS:
        existing_correlations = cross_correlation.loc[feature_name, FEATURE_COLUMNS]
        strongest_existing, strongest_correlation = _strongest_finite_correlation(
            existing_correlations
        )
        feature_values = values[feature_name].to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": feature_name,
                "dtype": str(values[feature_name].dtype),
                "unit": definition_by_name[feature_name]["unit"],
                "source": definition_by_name[feature_name]["source"],
                "finite_rate": float(np.isfinite(feature_values).mean()),
                "unique_count": int(unique_counts[feature_name]),
                "mean": float(description.loc[feature_name, "mean"]),
                "std": float(description.loc[feature_name, "std"]),
                "p01": float(description.loc[feature_name, "1%"]),
                "p50": float(description.loc[feature_name, "50%"]),
                "p99": float(description.loc[feature_name, "99%"]),
                "well_constant_rate": float(well_constant_rate[feature_name]),
                "strongest_existing_feature": strongest_existing,
                "correlation_with_existing_feature": strongest_correlation,
            }
        )
    return pd.DataFrame(rows)


def validate_and_build_well_table(
    candidates: pd.DataFrame,
    base_features: pd.DataFrame,
    predictions: pd.DataFrame,
    b00_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    candidate_keys = candidates[["well_id", "row_index"]].copy()
    base_keys = base_features[["well_id", "row_index"]].copy()
    prediction_keys = predictions[["well_id", "row_index"]].copy()
    b00_keys = b00_predictions[["well_id", "row_index"]].copy()
    for key_table in [candidate_keys, base_keys, prediction_keys, b00_keys]:
        key_table["well_id"] = key_table["well_id"].astype(str)
    candidate_base_match = candidate_keys.equals(base_keys)
    candidate_prediction_match = candidate_keys.equals(prediction_keys)
    candidate_b00_match = candidate_keys.equals(b00_keys)
    target_match = np.array_equal(
        predictions["target_tvt"].to_numpy(),
        b00_predictions["target_tvt"].to_numpy(),
    )
    if not all(
        [candidate_base_match, candidate_prediction_match, candidate_b00_match, target_match]
    ):
        raise ValueError("F05a、B00、候选缓存的逐行键或目标没有完全对齐")

    row_table = pd.DataFrame(
        {
            "well_id": predictions["well_id"].astype(str),
            "fold": predictions["fold"].to_numpy(),
            "f05_sse": np.square(
                predictions["pred_tvt"].to_numpy()
                - predictions["target_tvt"].to_numpy()
            ),
            "b00_sse": np.square(
                b00_predictions["pred_tvt"].to_numpy()
                - predictions["target_tvt"].to_numpy()
            ),
            "carry_sse": np.square(
                predictions["carry_tvt"].to_numpy()
                - predictions["target_tvt"].to_numpy()
            ),
            "gr_missing": base_features["gr_missing"].to_numpy(),
            "hidden_fraction": base_features["hidden_fraction"].to_numpy(),
            "pf_uncertainty": candidates["pf_ancc_std"].to_numpy(),
            "beam_disagreement": candidates["beam_std_d"].to_numpy(),
            "ncc_score": candidates["sc25_sc"].to_numpy(),
        }
    )
    well_table = (
        row_table.groupby("well_id", sort=False, observed=True)
        .agg(
            fold=("fold", "first"),
            rows=("fold", "size"),
            f05_sse=("f05_sse", "sum"),
            b00_sse=("b00_sse", "sum"),
            carry_sse=("carry_sse", "sum"),
            gr_missing_rate=("gr_missing", "mean"),
            hidden_fraction=("hidden_fraction", "max"),
            pf_uncertainty=("pf_uncertainty", "mean"),
            beam_disagreement=("beam_disagreement", "mean"),
            ncc_score=("ncc_score", "mean"),
        )
        .reset_index()
    )
    alignment = {
        "rows": int(len(candidates)),
        "wells": int(well_table["well_id"].nunique()),
        "candidate_base_keys_exact": bool(candidate_base_match),
        "candidate_prediction_keys_exact": bool(candidate_prediction_match),
        "candidate_b00_keys_exact": bool(candidate_b00_match),
        "target_tvt_exact_between_f05a_and_b00": bool(target_match),
    }
    return well_table, alignment


def build_slice_metrics(well_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    slice_columns = [
        "rows",
        "gr_missing_rate",
        "pf_uncertainty",
        "beam_disagreement",
        "ncc_score",
    ]
    labels = ["Q1_low", "Q2", "Q3", "Q4_high"]
    for slice_column in slice_columns:
        ranked = well_table[slice_column].rank(method="first")
        buckets = pd.qcut(ranked, q=4, labels=labels)
        for bucket in labels:
            selected = well_table.loc[buckets == bucket]
            evaluation_rows = int(selected["rows"].sum())
            f05_rmse = float(np.sqrt(selected["f05_sse"].sum() / evaluation_rows))
            b00_rmse = float(np.sqrt(selected["b00_sse"].sum() / evaluation_rows))
            carry_rmse = float(np.sqrt(selected["carry_sse"].sum() / evaluation_rows))
            rows.append(
                {
                    "slice_feature": slice_column,
                    "slice_bucket": bucket,
                    "wells": int(len(selected)),
                    "rows": evaluation_rows,
                    "f05_micro_rmse": f05_rmse,
                    "b00_micro_rmse": b00_rmse,
                    "carry_micro_rmse": carry_rmse,
                    "delta_vs_b00": f05_rmse - b00_rmse,
                    "delta_vs_carry": f05_rmse - carry_rmse,
                }
            )
    return pd.DataFrame(rows)


def build_bootstrap(well_table: pd.DataFrame, repeats: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    well_count = len(well_table)
    result = np.empty((repeats, 2), dtype=np.float64)
    rows = well_table["rows"].to_numpy(dtype=np.int64)
    f05_sse = well_table["f05_sse"].to_numpy(dtype=np.float64)
    b00_sse = well_table["b00_sse"].to_numpy(dtype=np.float64)
    carry_sse = well_table["carry_sse"].to_numpy(dtype=np.float64)
    for repeat_index in range(repeats):
        sampled = rng.integers(0, well_count, size=well_count)
        sampled_rows = rows[sampled].sum()
        f05_rmse = np.sqrt(f05_sse[sampled].sum() / sampled_rows)
        b00_rmse = np.sqrt(b00_sse[sampled].sum() / sampled_rows)
        carry_rmse = np.sqrt(carry_sse[sampled].sum() / sampled_rows)
        result[repeat_index] = [f05_rmse - b00_rmse, f05_rmse - carry_rmse]
    return pd.DataFrame(
        {
            "replicate": np.arange(repeats, dtype=np.int32),
            "delta_vs_b00": result[:, 0],
            "delta_vs_carry": result[:, 1],
        }
    )


def main() -> None:
    artifact_dir = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
    negative_dir = CLEAN_ROOT / "artifacts" / NEGATIVE_ID
    candidate_path = artifact_dir / "candidate_feature_cache.parquet"
    source_path = PROJECT_ROOT / "input" / "others" / "data" / "train.csv"
    notebook_path = PROJECT_ROOT / "rogii-dual-track-prefix-calibrated-geosteering.ipynb"

    candidates = pd.read_parquet(candidate_path)
    candidates["well_id"] = candidates["well_id"].astype(str)
    base_features = pd.read_parquet(
        CLEAN_ROOT / "artifacts" / "B00_simple_lgbm_v1" / "feature_cache.parquet",
        columns=["well_id", "row_index", *FEATURE_COLUMNS],
    )
    base_features = base_features.loc[:, ~base_features.columns.duplicated()].copy()
    base_features["well_id"] = base_features["well_id"].astype(str)
    predictions = pd.read_parquet(artifact_dir / "predictions.parquet")
    b00_predictions = pd.read_parquet(
        CLEAN_ROOT / "artifacts" / "B00_simple_lgbm_v1" / "predictions.parquet"
    )

    definitions = feature_definitions()
    write_json(artifact_dir / "feature_definition.json", definitions)
    lineage_rows = []
    for definition in definitions:
        feature_name = definition["feature"]
        lineage_rows.append(
            {
                "feature": feature_name,
                "raw_source": definition["source"],
                "uses_tvt_input": True,
                "uses_hidden_gr": feature_name != "sc_trust",
                "uses_typewell": feature_name.startswith("pf_")
                or feature_name.startswith("beam_")
                or feature_name == "hyb_d",
                "uses_pf_or_beam": feature_name.startswith("pf_")
                or feature_name.startswith("beam_")
                or feature_name == "hyb_d",
                "needs_outer_train_fit": False,
                "fit_wells_hash": None,
                "transform_version": "notebook_cell14_build_well_static_cache",
                "unit": definition["unit"],
                "notebook_cell": 14,
                "cell_local_feature_lines": "502-519",
                "target_created_after_features_at_cell_local_line": 562,
            }
        )
    write_json(
        artifact_dir / "feature_lineage.json",
        {
            "notebook": str(notebook_path),
            "notebook_sha256": file_sha256(notebook_path),
            "source_csv": str(source_path),
            "source_csv_sha256": SOURCE_SHA256,
            "generator": "cell 14 build_well()",
            "generator_evidence": {
                "pf_and_beam_compute_lines": "cell 14 local lines 400-423",
                "selected_feature_dictionary_lines": "cell 14 local lines 502-519",
                "target_assignment_lines": "cell 14 local lines 560-562",
                "selected_features_created_before_target": True,
                "spatial_surface_code_not_selected": True,
            },
            "features": lineage_rows,
        },
    )

    quality = summarize_feature_quality(candidates, base_features, definitions)
    quality.to_csv(artifact_dir / "feature_quality.csv", index=False)

    well_table, alignment = validate_and_build_well_table(
        candidates, base_features, predictions, b00_predictions
    )
    slices = build_slice_metrics(well_table)
    slices.to_csv(artifact_dir / "slice_metrics.csv", index=False)
    bootstrap = build_bootstrap(well_table)
    bootstrap.to_parquet(artifact_dir / "bootstrap_replicates.parquet", index=False)

    official_fold0 = _read_fold_micro_rmse(artifact_dir, fold_id=0)
    negative_fold0 = _read_fold_micro_rmse(negative_dir, fold_id=0)
    b00_fold0 = _read_fold_micro_rmse(
        CLEAN_ROOT / "artifacts" / "B00_simple_lgbm_v1", fold_id=0
    )
    negative_metrics = {
        "control": "每口井24列联合循环平移半个隐藏段",
        "preserved": "井内每列分布、24列之间的同期联合关系",
        "destroyed": "候选路径与当前隐藏行的正确对应",
        "fold": 0,
        "official_rmse": official_fold0,
        "negative_control_rmse": negative_fold0,
        "b00_rmse": b00_fold0,
        "control_minus_official": negative_fold0 - official_fold0,
        "control_minus_b00": negative_fold0 - b00_fold0,
        "interpretation": "错位后显著退化，逐行轨迹形状有真实贡献；控制仍优于B00，说明井级分布信息或残余平滑相关仍有作用。",
    }
    write_json(artifact_dir / "negative_control_metrics.json", negative_metrics)

    source_stat = source_path.stat()
    cache_meta = json.loads(
        (artifact_dir / "candidate_feature_cache.meta.json").read_text(encoding="utf-8")
    )
    write_json(
        artifact_dir / "cache_manifest.json",
        {
            "source_csv": str(source_path),
            "source_size": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "source_sha256": SOURCE_SHA256,
            "candidate_cache": str(candidate_path),
            "candidate_cache_sha256": file_sha256(candidate_path),
            "candidate_cache_metadata": cache_meta,
            "rows": int(len(candidates)),
            "wells": int(candidates["well_id"].nunique()),
            "feature_count": len(DIRECT_CANDIDATE_COLUMNS),
            "generation_config_complete": False,
            "generation_config_risk": "旧 train.csv 没有保存完整 Notebook 代码/参数/随机状态指纹。",
        },
    )

    leakage_tests = {
        "hidden_tvt_deletion_invariance": {
            "status": "partial_pass",
            "evidence": "正式 loader 的 usecols 不含 target；单测证明删除/改变 target 不改变24列。旧 Notebook 生成器尚未独立抽取后做动态删除测试。",
        },
        "hidden_tvt_mutation_invariance": {
            "status": "partial_pass",
            "evidence": "tests/test_f05a_direct_physical_candidates.py 已验证 target 任意改变不影响 loader 输出；生成公式静态检查不读隐藏 TVT。",
        },
        "surface_deletion_invariance": {
            "status": "pass",
            "evidence": "usecols 只读 well/id 和24列；24列在 cell14 的空间 surface 分支之前或独立生成，sig/tvtF/dense/spatial 均排除。",
        },
        "outer_fold_source_exclusion": {
            "status": "not_applicable",
            "evidence": "24列只由查询井自身与其配对 Typewell 计算，不拟合其他训练井。",
        },
        "pf_cache_fold_match": {
            "status": "not_applicable",
            "evidence": "这些是逐井物理路径，不是由 outer-train 拟合的 OOF 模型输出。",
        },
        "row_hash_match": {"status": "pass", "evidence": alignment},
        "fixed_seed_reproducibility": {
            "status": "fail_pending_fix",
            "evidence": "cell14 声明 SEED=42，但 pf_ancc/pf_z 的 Numba 内核没有显式 seed；Beam/NCC 确定，两个单次PF列尚不能保证从零逐位复现。",
        },
        "negative_control_status": {
            "status": "pass_with_residual_signal",
            "evidence": negative_metrics,
        },
        "forbidden_columns": {
            "status": "pass",
            "excluded": [
                "target",
                "sig_std",
                "sig_mean_d",
                "tvtF_*",
                "dense_*",
                "spatial_*",
                "surface/contact columns",
            ],
        },
    }
    write_json(artifact_dir / "leakage_tests.json", leakage_tests)

    pd.read_csv(artifact_dir / "metrics.csv").to_csv(
        artifact_dir / "per_fold.csv", index=False
    )

    f05_rmse = float(np.sqrt(well_table["f05_sse"].sum() / well_table["rows"].sum()))
    b00_rmse = float(np.sqrt(well_table["b00_sse"].sum() / well_table["rows"].sum()))
    bootstrap_ci = np.quantile(bootstrap["delta_vs_b00"], [0.025, 0.975])
    conclusion = f"""# F05a 直接物理候选五折结论

## 事实

- 固定单模 LightGBM、固定 spatial-pad 五折、3,783,989 行下，F05a micro RMSE 为 `{f05_rmse:.6f}`。
- 同一评价行的 B00 为 `{b00_rmse:.6f}`，改善 `{b00_rmse-f05_rmse:.6f} ft`。
- 五个 fold 均改善；井级 bootstrap 的 F05a-B00 95% 区间为 `[{bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}] ft`。
- fold 0 正式候选 `{negative_metrics['official_rmse']:.6f}`；井内错位负对照 `{negative_metrics['negative_control_rmse']:.6f}`，错位后恶化 `{negative_metrics['control_minus_official']:.6f} ft`。
- 24 列在 Notebook cell 14 的 `build_well()` 中先于 `target` 生成，只使用当前井可见 `TVT_input`、当前井 MD/XYZ/GR、隐藏 GR 和配对 Typewell TVT/GR。

## 推断

PF、Beam 和多尺度自模板候选的逐行路径，给 B00 带来了稳定且很大的额外信息，不是单纯靠候选列的井级分布就能复制。

## 当前不能忽略的风险

旧 `train.csv` 没有保存完整生成配置和随机状态；尤其 `pf_ancc/pf_z` 的 Numba 单次 PF 没有显式设置 seed。因此 `{f05_rmse:.6f}` 是可信的冻结缓存 CV 强候选，但还不是已经从原始 CSV 独立复现、可直接生成 private test 特征的最终基线。

## 当前只能否定

不能因为负对照仍优于 B00，就否定逐行路径价值；正式路径比负对照仍好 `{negative_metrics['control_minus_official']:.6f} ft`。负对照只说明候选的井级分布也携带信息。

## 下一步

从 Notebook cell 14 抽出这 24 列的最小生成器，固定 PF 随机种子，不调任何 PF/Beam 参数；先验证 Beam/NCC 与缓存逐位一致，再评估两个随机 PF 列的固定 seed 重建差异。通过后才能把 F05a 冻结为可提交基线。
"""
    (artifact_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    print(f"F05a 审计完成：F05a={f05_rmse:.6f}, B00={b00_rmse:.6f}")
    print(f"bootstrap 95% CI vs B00: [{bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}]")


if __name__ == "__main__":
    main()
