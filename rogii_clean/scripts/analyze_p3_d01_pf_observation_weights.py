"""P3-D01：在合法 PF 回放完成后，独立读取开发井真值并生成诊断结论。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import diagnose_p3_d01_pf_observation_weights as legal_runner  # noqa: E402
from src.p3_d01_diagnostic_analysis import (  # noqa: E402
    TEMPERATURES,
    add_diagnostic_columns,
    aggregate_path_metrics,
    build_binned_metrics,
    correlation_tables,
    decide_pf_routes,
    score_well_paths,
)


EXPERIMENT_ID = "P3_D01_pf_observation_weight_audit_v1"
DEFAULT_CONFIG_PATH = CLEAN_ROOT / "configs" / "p3_d01_pf_observation_weight_audit_v1.json"
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "fold01", "all")
TARGET_COLUMNS = ["well_id", "fold", "row_index", "target_tvt", "pred_tvt"]
PATH_CACHE_COLUMNS = [
    "well_id",
    "row_index",
    "last_visible_tvt",
    "pf128_mean_tvt",
    "pf128_scale_3_delta",
    "pf128_scale_5_delta",
    "pf128_scale_8_delta",
    "pf128_scale_12_delta",
]
PERMUTATIONS = 1000
RANDOM_SEED = 29


def _resolve_from_clean(path_text: str) -> Path:
    """把配置路径统一解释为相对 rogii_clean 根目录。"""

    return (CLEAN_ROOT / path_text).resolve()


def _json_ready(value: Any) -> Any:
    """把 NumPy 标量递归转换为标准 JSON 标量。"""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    """原子写 JSON，避免中断留下半个文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """原子写 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    """原子写 UTF-8 文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def output_paths(artifact_dir: Path, mode: str) -> dict[str, Path]:
    """返回互不覆盖的模式输出；all 额外给出最终无后缀合同名。"""

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"P3-D01 不支持模式 {mode}")
    artifact_dir = Path(artifact_dir)
    paths = {
        "oracle_join": artifact_dir / "oracle" / f"per_well_error_join_{mode}.csv",
        "per_fold_correlations": artifact_dir / f"per_fold_correlations_{mode}.csv",
        "overall_correlations": artifact_dir / f"overall_correlations_{mode}.csv",
        "binned_metrics": artifact_dir / f"binned_metrics_{mode}.csv",
        "scale_path_metrics": artifact_dir / f"scale_path_metrics_{mode}.csv",
        "summary": artifact_dir / f"summary_{mode}.json",
        # legal runner 自己使用 runtime_<mode>.json 作为评分闸门；
        # 分析状态必须另存，不能覆盖这份合法生成证明。
        "runtime": artifact_dir / f"analysis_runtime_{mode}.json",
        "conclusion": artifact_dir / f"conclusion_{mode}.md",
    }
    if mode == "all":
        paths.update(
            {
                "final_per_well": artifact_dir / "per_well.csv",
                "final_per_fold_correlations": artifact_dir / "per_fold_correlations.csv",
                "final_overall_correlations": artifact_dir / "overall_correlations.csv",
                "final_binned_metrics": artifact_dir / "binned_metrics.csv",
                "final_scale_path_metrics": artifact_dir / "scale_path_metrics.csv",
                "final_summary": artifact_dir / "summary.json",
                "final_runtime": artifact_dir / "analysis_runtime.json",
                "final_conclusion": artifact_dir / "conclusion.md",
            }
        )
    return paths


def validate_legal_table(legal_table: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """确认合法表无真值派生列，并与当前模式的井、折、行数完全一致。"""

    required_columns = {"well_id", "fold", "hidden_rows"}
    if not required_columns.issubset(legal_table.columns):
        raise ValueError("P3-D01 legal 表缺少井号、fold 或隐藏行数")
    forbidden_fragments = ("target", "truth", "error", "rmse", "oracle", "surface")
    forbidden_columns = [
        column
        for column in legal_table.columns
        if any(fragment in column.lower() for fragment in forbidden_fragments)
    ]
    if forbidden_columns:
        raise ValueError(f"P3-D01 legal 表出现禁止列：{forbidden_columns}")
    legal = legal_table.copy()
    legal["well_id"] = legal["well_id"].astype(str)
    expected = selected.loc[:, ["well_id", "fold", "hidden_rows"]].copy()
    expected["well_id"] = expected["well_id"].astype(str)
    if legal["well_id"].isna().any() or legal["well_id"].duplicated().any():
        raise ValueError("P3-D01 legal 表必须每井恰好一行")
    audit = expected.merge(
        legal.loc[:, ["well_id", "fold", "hidden_rows"]],
        on="well_id",
        how="outer",
        suffixes=("_expected", "_legal"),
        indicator=True,
        validate="one_to_one",
    )
    if not audit["_merge"].eq("both").all():
        raise ValueError("P3-D01 legal 表井集合与当前模式不一致")
    if not np.array_equal(
        audit["fold_expected"].to_numpy(dtype=np.int64),
        audit["fold_legal"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("P3-D01 legal 表 fold 与冻结登记不一致")
    if not np.array_equal(
        audit["hidden_rows_expected"].to_numpy(dtype=np.int64),
        audit["hidden_rows_legal"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("P3-D01 legal 表隐藏行数与冻结登记不一致")
    return legal.sort_values("well_id", kind="mergesort").reset_index(drop=True)


def require_completed_legal_outputs(
    artifact_dir: Path,
    mode: str,
    selected: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """在接触目标前，确认 legal runtime 和逐井表已完整落盘。"""

    runtime_path = Path(artifact_dir) / f"runtime_{mode}.json"
    legal_path = Path(artifact_dir) / "legal" / f"per_well_{mode}.csv"
    if not runtime_path.is_file() or not legal_path.is_file():
        raise RuntimeError(f"P3-D01 legal {mode} 尚未完成")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("completed") is not True:
        raise RuntimeError(f"P3-D01 legal {mode} 含错误或尚未完成")
    if int(runtime.get("wells", -1)) != len(selected):
        raise RuntimeError(f"P3-D01 legal {mode} 井数不完整")
    expected_rows = int(selected["hidden_rows"].sum())
    if int(runtime.get("hidden_rows", -1)) != expected_rows:
        raise RuntimeError(f"P3-D01 legal {mode} 隐藏行数不完整")
    legal = validate_legal_table(pd.read_csv(legal_path), selected)
    return runtime, legal


def load_development_targets(
    predictions_path: Path,
    selected: pd.DataFrame,
    shadow_ids: set[str],
) -> pd.DataFrame:
    """用 Arrow 条件扫描只读取所选开发井，同时再次排除全部影子井。"""

    selected_ids = selected["well_id"].astype(str).tolist()
    selected_id_set = set(selected_ids)
    shadow_id_set = {str(well_id) for well_id in shadow_ids}
    if not selected_id_set or not selected_id_set.isdisjoint(shadow_id_set):
        raise ValueError("P3-D01 selected 为空或与影子井重叠")
    dataset = ds.dataset(Path(predictions_path), format="parquet")
    arrow_filter = ds.field("well_id").isin(sorted(selected_id_set))
    if shadow_id_set:
        arrow_filter = arrow_filter & ~ds.field("well_id").isin(sorted(shadow_id_set))
    table = dataset.to_table(columns=TARGET_COLUMNS, filter=arrow_filter)
    targets = table.to_pandas()
    targets["well_id"] = targets["well_id"].astype(str)
    if set(targets["well_id"]) != selected_id_set:
        raise ValueError("P3-D01 目标表未覆盖全部 selected 开发井")
    if not set(targets["well_id"]).isdisjoint(shadow_id_set):
        raise RuntimeError("P3-D01 目标扫描出现影子井")
    if targets.duplicated(["well_id", "row_index"]).any():
        raise ValueError("P3-D01 目标表含重复井内 row_index")
    numeric_columns = ["fold", "row_index", "target_tvt", "pred_tvt"]
    numeric_values = targets[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(numeric_values.astype(np.float64)).all():
        raise ValueError("P3-D01 开发目标含非有限值")
    observed = targets.groupby("well_id", sort=False).agg(
        fold=("fold", "first"),
        fold_count=("fold", "nunique"),
        hidden_rows=("row_index", "size"),
    )
    expected = selected.set_index(selected["well_id"].astype(str))[["fold", "hidden_rows"]]
    observed = observed.reindex(expected.index)
    if observed.isna().any().any() or not observed["fold_count"].eq(1).all():
        raise ValueError("P3-D01 开发目标的井或 fold 不完整")
    if not np.array_equal(observed["fold"].to_numpy(dtype=np.int64), expected["fold"].to_numpy(dtype=np.int64)):
        raise ValueError("P3-D01 开发目标 fold 与冻结登记不一致")
    if not np.array_equal(
        observed["hidden_rows"].to_numpy(dtype=np.int64),
        expected["hidden_rows"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("P3-D01 开发目标行数与冻结登记不一致")
    return targets.sort_values(["well_id", "row_index"], kind="mergesort").reset_index(drop=True)


def score_development_paths(
    targets: pd.DataFrame,
    selected: pd.DataFrame,
    source_path_cache_dir: Path,
) -> pd.DataFrame:
    """逐井读取冻结 P2-P01 合法路径并按 row_index 评分。"""

    score_rows: list[dict[str, Any]] = []
    for registry_row in selected.sort_values("well_id", kind="mergesort").itertuples(index=False):
        well_id = str(registry_row.well_id)
        well_targets = targets.loc[targets["well_id"].astype(str) == well_id, TARGET_COLUMNS]
        cache_path = Path(source_path_cache_dir) / f"{well_id}.parquet"
        if not cache_path.is_file():
            raise FileNotFoundError(f"P3-D01 找不到井 {well_id} 的旧合法路径缓存")
        legal_paths = pd.read_parquet(cache_path, columns=PATH_CACHE_COLUMNS)
        if not legal_paths["well_id"].astype(str).eq(well_id).all():
            raise ValueError(f"P3-D01 井 {well_id} 的旧路径缓存井号不一致")
        score = score_well_paths(well_targets, legal_paths.drop(columns="well_id"))
        if int(score["hidden_rows"]) != int(registry_row.hidden_rows):
            raise ValueError(f"P3-D01 井 {well_id} 的评分行数不一致")
        score_rows.append(score)
    scored = pd.DataFrame(score_rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if len(scored) != len(selected) or not scored["well_id"].is_unique:
        raise ValueError("P3-D01 逐井评分未保持一井一行")
    if not scored["best_scale_is_oracle"].eq(True).all():  # noqa: E712 - 明确审计布尔值
        raise ValueError("P3-D01 oracle temperature 标记缺失")
    return scored


def _correlation_pairs() -> list[tuple[str, str]]:
    """返回实验卡冻结的 18 组井级相关关系。"""

    pairs = [
        ("missing_gr_fraction", "pf128_mean_rmse"),
        ("longest_gr_gap_md_ft", "pf128_mean_rmse"),
    ]
    for temperature in TEMPERATURES:
        ess_column = f"scale_{temperature}_effective_sample_size"
        for outcome in ("hidden_rows", "observed_gr_fraction", "pf128_mean_rmse", "p2p02_rmse"):
            pairs.append((ess_column, outcome))
    return pairs


def _conclusion_text(mode: str, summary: dict[str, Any]) -> str:
    """按项目要求生成不夸大 whole-well 诊断的中文结论。"""

    decisions = summary["route_decisions"]
    pf01_supported = bool(decisions["pf01_supported"])
    pf02_supported = bool(decisions["pf02_supported"])
    if pf01_supported:
        next_step = "按原路线进入 P3-PF01：缺失感知、稳健的 GR 标定。"
    elif pf02_supported:
        next_step = "先做 P3-PF02 的最小实现：固定有效路径数的井级权重。"
    else:
        next_step = "先做最便宜的 PF01 小样本实现；本次负证据不否定沿 MD 的分段似然。"
    return f"""# P3-D01 {mode} 结论

## 数据直接证明的事实

- 本模式只评分 {summary['well_count']} 口开发井、{summary['hidden_rows']} 个自然隐藏行，影子井重叠为 0。
- PF01 井级证据支持：{pf01_supported}。
- PF02 井级证据支持：{pf02_supported}。
- 本实验没有训练模型，也没有输出正式特征或逐行预测。

## 基于事实的合理推断

- 井级缺失、GR 标定差异和粒子权重塌缩可以帮助决定下一条最低成本路线。
- 每井事后最优温度只是上限诊断，不是可提交方法。

## 仍然没有验证的猜测

- 当前只有整井累计似然，不能证明沿 MD 不同位置应该使用不同权重。
- PF03（沿 MD 分段或动态似然）仍需单独保存分段证据后验证。

## 当前实验只能否定的具体实现

- 若某项门槛未通过，只能说明当前整井统计没有给出足够支持，不能否定该信息源。

## 下一步最便宜的验证

- {next_step}
"""


def analyze_mode(
    *,
    mode: str,
    selected: pd.DataFrame,
    shadow_ids: set[str],
    predictions_path: Path,
    source_path_cache_dir: Path,
    artifact_dir: Path,
    permutations: int = PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """先验证 legal 完成，再读取开发目标、评分并保存全部诊断产物。"""

    started = time.perf_counter()
    artifact_dir = Path(artifact_dir)
    paths = output_paths(artifact_dir, mode)
    legal_runtime, legal = require_completed_legal_outputs(artifact_dir, mode, selected)

    # 目标读取必须位于 legal 完整性门控之后，不能调换顺序。
    targets = load_development_targets(predictions_path, selected, shadow_ids)
    scored = score_development_paths(targets, selected, source_path_cache_dir)
    score_without_keys = scored.drop(columns=["fold", "hidden_rows"])
    joined = legal.merge(score_without_keys, on="well_id", how="inner", validate="one_to_one")
    joined = add_diagnostic_columns(joined)
    if len(joined) != len(selected) or not joined["well_id"].is_unique:
        raise ValueError("P3-D01 legal 与评分结果连接后井数变化")

    path_metrics = aggregate_path_metrics(joined)
    overall_correlations, per_fold_correlations = correlation_tables(
        joined,
        _correlation_pairs(),
        permutations=permutations,
        seed=seed,
    )
    binned_metrics = build_binned_metrics(joined)
    route_decisions = decide_pf_routes(joined, overall_correlations, path_metrics)
    path_records = path_metrics.to_dict(orient="records")
    pooled_rmse = {str(row["path_name"]): float(row["pooled_rmse"]) for row in path_records}
    fixed_names = [f"scale_{temperature}" for temperature in TEMPERATURES]
    best_fixed_name = min(fixed_names, key=pooled_rmse.__getitem__)
    oracle_gap = pooled_rmse[best_fixed_name] - pooled_rmse["oracle_scale"]
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "frozen_full_baseline": {
            "baseline_id": "P3B00_group5_p2p02_v1",
            "well_count": 773,
            "micro_rmse": 10.305704992073148,
        },
        "well_count": int(len(joined)),
        "hidden_rows": int(joined["hidden_rows"].sum()),
        "shadow_overlap": 0,
        "pooled_rmse": pooled_rmse,
        "best_global_fixed_temperature": best_fixed_name,
        "best_fixed_minus_oracle_ft": float(oracle_gap),
        "route_decisions": route_decisions,
        "contains_target_derived_metrics": True,
        "model_training": False,
        "formal_feature_output": False,
        "shadow_target_access": False,
        "permutations": int(permutations),
        "seed": int(seed),
        "whole_well_only": True,
        "positional_weight_claim_allowed": False,
    }
    runtime = dict(legal_runtime)
    runtime.update(
        {
            "completed": True,
            "analysis_completed": True,
            "mode": mode,
            "wells": int(len(joined)),
            "hidden_rows": int(joined["hidden_rows"].sum()),
            "shadow_overlap": 0,
            "target_scan_after_legal_completion": True,
            "model_training": False,
            "formal_feature_output": False,
            "shadow_target_access": False,
            "analysis_elapsed_seconds": float(time.perf_counter() - started),
        }
    )
    conclusion = _conclusion_text(mode, summary)

    _write_csv(paths["oracle_join"], joined)
    _write_csv(paths["per_fold_correlations"], per_fold_correlations)
    _write_csv(paths["overall_correlations"], overall_correlations)
    _write_csv(paths["binned_metrics"], binned_metrics)
    _write_csv(paths["scale_path_metrics"], path_metrics)
    _write_json(paths["summary"], summary)
    _write_json(paths["runtime"], runtime)
    _write_text(paths["conclusion"], conclusion)

    if mode == "all":
        _write_csv(paths["final_per_well"], joined)
        _write_csv(paths["final_per_fold_correlations"], per_fold_correlations)
        _write_csv(paths["final_overall_correlations"], overall_correlations)
        _write_csv(paths["final_binned_metrics"], binned_metrics)
        _write_csv(paths["final_scale_path_metrics"], path_metrics)
        _write_json(paths["final_summary"], summary)
        _write_json(paths["final_runtime"], runtime)
        _write_text(paths["final_conclusion"], conclusion)
    return summary


def _load_shadow_ids(shadow_path: Path) -> set[str]:
    """只读取影子登记中的井号，不读取任何标签。"""

    header = pd.read_csv(shadow_path, nrows=0).columns.tolist()
    use_columns = ["well_id"]
    if "is_shadow" in header:
        use_columns.append("is_shadow")
    shadow = pd.read_csv(shadow_path, usecols=use_columns, dtype={"well_id": str})
    if "is_shadow" in shadow.columns:
        values = shadow["is_shadow"]
        if pd.api.types.is_bool_dtype(values):
            shadow = shadow.loc[values]
        else:
            normalized = values.astype(str).str.strip().str.lower()
            shadow = shadow.loc[normalized.isin({"1", "true", "yes", "y"})]
    return set(shadow["well_id"].astype(str))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析分析模式和只读来源。"""

    parser = argparse.ArgumentParser(description="分析 P3-D01 PF 观测权重审计")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """加载冻结开发井登记，并执行独立的目标评分阶段。"""

    arguments = parse_args(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    legal_runner.validate_config(config)
    fold_path = _resolve_from_clean(str(config["fold_registry"]))
    shadow_path = _resolve_from_clean(str(config["shadow_registry"]))
    development = legal_runner.load_development_registry(fold_path, shadow_path, config)
    selected = legal_runner.select_mode_registry(development, arguments.mode)
    shadow_ids = _load_shadow_ids(shadow_path)
    predictions_path = _resolve_from_clean(str(config["source_model_predictions"]))
    source_path_cache_dir = _resolve_from_clean(str(config["source_pf_legal_cache_dir"]))
    summary = analyze_mode(
        mode=arguments.mode,
        selected=selected,
        shadow_ids=shadow_ids,
        predictions_path=predictions_path,
        source_path_cache_dir=source_path_cache_dir,
        artifact_dir=arguments.artifact_dir.resolve(),
        permutations=PERMUTATIONS,
        seed=RANDOM_SEED,
    )
    print(
        f"P3-D01 {arguments.mode} 分析完成：{summary['well_count']} 口井，"
        f"{summary['hidden_rows']} 行，影子重叠 0",
        flush=True,
    )
    print(f"PF01 支持：{summary['route_decisions']['pf01_supported']}", flush=True)
    print(f"PF02 支持：{summary['route_decisions']['pf02_supported']}", flush=True)


if __name__ == "__main__":
    main()
