"""RF01a 稳定性审计 runner 的固定合同测试。"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import sys

import numpy as np
import pandas as pd
import pytest


# 测试文件位于 rogii_clean/tests；把干净项目根目录加入导入路径。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.run_rf01_stability_audit import (
    FEATURE_QUALITY_COLUMNS,
    FINAL_OUTPUT_RELATIVE_PATHS,
    LEAKAGE_TEST_NAMES,
    NEW_FOLDS,
    REUSED_FOLDS,
    aggregate_feature_importance,
    build_b00_comparison,
    build_conclusion_text,
    build_diagnostic_reporting_artifacts,
    build_feature_contract_documents,
    build_feature_quality_table,
    build_fold_cache_manifest,
    build_fixed_fold_directories,
    build_leakage_tests_document,
    build_paired_bootstrap_artifacts,
    build_run_manifest,
    classify_fold_pattern,
    compute_row_key_hash,
    merge_fixed_fold_predictions,
    preflight_audit,
    preserve_or_write_prediction_table,
    read_source_runtime_contract,
    run_new_folds,
    select_training_fingerprint,
    validate_config_contract,
    validate_evaluation_alignment,
    validate_reused_fold_runtime_contract,
    validate_existing_new_fold_reuse,
    validate_well_fold_assignments,
)
from src.metrics import paired_well_bootstrap


def test_fold_sets_are_frozen_and_disjoint() -> None:
    """审计只能复用 folds 0–1，并且只能新运行 folds 2–4。"""

    assert REUSED_FOLDS == (0, 1)
    assert NEW_FOLDS == (2, 3, 4)
    assert set(REUSED_FOLDS).isdisjoint(NEW_FOLDS)


@pytest.mark.parametrize(
    ("fold_deltas", "expected_pattern"),
    [
        ([-2.6, 0.1, -0.2, -0.3, -0.1], "fold1是唯一反向折"),
        ([-2.6, 0.1, 0.2, 0.3, 0.1], "fold0是唯一正向折"),
        ([-2.6, 0.1, -0.2, 0.3, -0.1], "其余折表现混合"),
    ],
)
def test_classify_fold_pattern_uses_pre_registered_three_way_rule(
    fold_deltas: list[float],
    expected_pattern: str,
) -> None:
    """折模式只能落入预注册的三个固定中文结论之一。"""

    assert classify_fold_pattern(fold_deltas) == expected_pattern


@pytest.mark.parametrize(
    "invalid_deltas",
    [
        [-2.6, 0.1, -0.2, -0.3],
        [-2.6, float("nan"), -0.2, -0.3, -0.1],
        [-2.6, float("inf"), -0.2, -0.3, -0.1],
    ],
)
def test_classify_fold_pattern_rejects_incomplete_or_non_finite_values(
    invalid_deltas: list[float],
) -> None:
    """缺折、NaN 或 Inf 都没有合法方向，不能进入结论。"""

    with pytest.raises(ValueError):
        classify_fold_pattern(invalid_deltas)


def test_classify_fold_pattern_treats_zero_as_mixed() -> None:
    """恰好为零既非改善也非反向，必须归入混合模式。"""

    assert (
        classify_fold_pattern([-2.6, 0.1, -0.2, 0.0, -0.1])
        == "其余折表现混合"
    )


def make_row_key_frame() -> pd.DataFrame:
    """构造跨井、跨折且输入顺序故意未排序的最小行键表。"""

    return pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_a"],
            "fold": [1, 0, 0],
            "row_index": [8, 2, 1],
        }
    )


def test_row_key_hash_is_independent_of_input_order() -> None:
    """同一组行键任意重排后必须得到完全相同的 SHA-256。"""

    row_keys = make_row_key_frame()
    shuffled_row_keys = row_keys.sample(frac=1.0, random_state=29).reset_index(drop=True)

    original_hash = compute_row_key_hash(row_keys)
    shuffled_hash = compute_row_key_hash(shuffled_row_keys)

    assert len(original_hash) == 64
    assert original_hash == shuffled_hash


def test_row_key_hash_changes_when_any_key_changes() -> None:
    """即使只改变一个 row_index，也必须改变评价行集合指纹。"""

    row_keys = make_row_key_frame()
    changed_row_keys = row_keys.copy()
    changed_row_keys.loc[0, "row_index"] = 9

    assert compute_row_key_hash(row_keys) != compute_row_key_hash(changed_row_keys)


def test_run_new_folds_calls_training_only_for_folds_2_3_4() -> None:
    """假训练函数绝不能收到 folds 0–1，也不能在审计目录创建旧折。"""

    received_folds: list[int] = []
    received_fold_dirs: list[Path] = []

    def fake_train_fold(
        feature_table: pd.DataFrame,
        registry: pd.DataFrame,
        fold_id: int,
        model_params: dict,
        artifact_dir: Path,
        fingerprint: str,
        feature_columns: list[str],
    ) -> dict:
        """记录通用 train_fold 形状的调用，不做任何模型训练。"""

        received_folds.append(fold_id)
        received_fold_dirs.append(artifact_dir / f"fold_{fold_id}")
        return {"fold": fold_id, "fingerprint": fingerprint}

    audit_artifact_dir = Path("audit-output")
    summaries = run_new_folds(
        feature_table=pd.DataFrame({"feature": [1.0]}),
        registry=pd.DataFrame({"fold": [0]}),
        model_params={"n_estimators": 1734},
        artifact_dir=audit_artifact_dir,
        fingerprint="fixed-audit-fingerprint",
        feature_columns=["feature"],
        new_folds=[2, 3, 4],
        train_callable=fake_train_fold,
    )

    assert received_folds == [2, 3, 4]
    assert [summary["fold"] for summary in summaries] == [2, 3, 4]
    assert received_fold_dirs == [
        audit_artifact_dir / "fold_2",
        audit_artifact_dir / "fold_3",
        audit_artifact_dir / "fold_4",
    ]
    assert audit_artifact_dir / "fold_0" not in received_fold_dirs
    assert audit_artifact_dir / "fold_1" not in received_fold_dirs


def test_run_new_folds_rejects_any_non_registered_fold_set() -> None:
    """即使调用者传入其他折，函数也必须在调用训练前拒绝。"""

    received_folds: list[int] = []

    def fake_train_fold(*args, **kwargs) -> dict:
        received_folds.append(int(args[2]))
        return {"fold": int(args[2])}

    with pytest.raises(ValueError, match="new_folds"):
        run_new_folds(
            feature_table=pd.DataFrame({"feature": [1.0]}),
            registry=pd.DataFrame({"fold": [0]}),
            model_params={"n_estimators": 1734},
            artifact_dir=Path("audit-output"),
            fingerprint="fixed-audit-fingerprint",
            feature_columns=["feature"],
            new_folds=[0, 1, 2, 3, 4],
            train_callable=fake_train_fold,
        )

    assert received_folds == []


def load_real_contract_values() -> tuple[dict, dict, dict, str, str]:
    """读取已登记的三份 JSON 及 fold/model 内容哈希供合同单测使用。"""

    audit_config_path = CLEAN_ROOT / "configs" / "rf01_stability_audit_v1.json"
    audit_config = json.loads(audit_config_path.read_text(encoding="utf-8"))
    source_config_path = CLEAN_ROOT / audit_config["source_config"]
    model_config_path = CLEAN_ROOT / audit_config["model_config"]
    fold_registry_path = CLEAN_ROOT / audit_config["fold_registry"]

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    fold_hash = hashlib.sha256(fold_registry_path.read_bytes()).hexdigest()
    model_hash = hashlib.sha256(model_config_path.read_bytes()).hexdigest()
    return audit_config, source_config, model_config, fold_hash, model_hash


def test_real_audit_configuration_matches_all_frozen_contracts() -> None:
    """真实登记必须禁止重选和调参，并锁定 RF01a、17 列、模型与 fold。"""

    audit_config, source_config, model_config, fold_hash, model_hash = (
        load_real_contract_values()
    )

    feature_columns = validate_config_contract(
        audit_config,
        source_config,
        model_config,
        actual_fold_hash=fold_hash,
        actual_model_config_hash=model_hash,
    )

    assert audit_config["allow_reselection"] is False
    assert audit_config["allow_parameter_change"] is False
    assert source_config["feature_version"] == "rf01_huber_slopes_17_v1"
    assert len(feature_columns) == 17
    assert feature_columns == source_config["feature_columns"]


@pytest.mark.parametrize(
    ("config_name", "field_name", "invalid_value", "expected_message"),
    [
        ("audit", "allow_reselection", True, "重新选择"),
        ("audit", "allow_parameter_change", True, "调参"),
        ("source", "feature_version", "another_rf01_version", "feature_version"),
        ("source", "feature_columns", ["wrong_feature"], "17 列"),
        ("source", "model_config", "configs/other_model.json", "模型配置"),
        ("audit", "fold_registry_sha256", "wrong-fold-hash", "fold hash"),
        ("audit", "pre_registered_selection_reasons", ["changed"], "选择理由"),
    ],
)
def test_config_contract_rejects_reselection_parameter_or_source_drift(
    config_name: str,
    field_name: str,
    invalid_value,
    expected_message: str,
) -> None:
    """任一预登记字段漂移都必须作为合同异常拒绝。"""

    audit_config, source_config, model_config, fold_hash, model_hash = (
        load_real_contract_values()
    )
    mutable_configs = {
        "audit": copy.deepcopy(audit_config),
        "source": copy.deepcopy(source_config),
    }
    mutable_configs[config_name][field_name] = invalid_value

    with pytest.raises(ValueError, match=expected_message):
        validate_config_contract(
            mutable_configs["audit"],
            mutable_configs["source"],
            model_config,
            actual_fold_hash=fold_hash,
            actual_model_config_hash=model_hash,
        )


def make_aligned_evaluation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造覆盖五折的最小 RF01a 特征键与 B00 预测表。"""

    feature_table = pd.DataFrame(
        {
            "well_id": [f"well_{fold}" for fold in range(5)],
            "fold": list(range(5)),
            "row_index": [10 + fold for fold in range(5)],
            "target_tvt": [100.0 + fold for fold in range(5)],
        }
    )
    b00_predictions = feature_table.copy()
    b00_predictions["pred_tvt"] = [100.5 + fold for fold in range(5)]
    return feature_table, b00_predictions


def test_evaluation_alignment_returns_verified_row_hash() -> None:
    """完全对齐时返回的 eval_row_hash 必须来自固定行键计算。"""

    feature_table, b00_predictions = make_aligned_evaluation_tables()

    eval_row_hash = validate_evaluation_alignment(
        feature_table,
        b00_predictions,
        expected_rows=5,
        expected_wells=5,
    )

    assert eval_row_hash == compute_row_key_hash(feature_table)


def test_evaluation_alignment_rejects_b00_row_key_mismatch() -> None:
    """B00 任一行键变化都必须在训练前触发合同异常。"""

    feature_table, b00_predictions = make_aligned_evaluation_tables()
    b00_predictions.loc[0, "row_index"] = 999

    with pytest.raises(ValueError, match="行键"):
        validate_evaluation_alignment(
            feature_table,
            b00_predictions,
            expected_rows=5,
            expected_wells=5,
        )


def test_evaluation_alignment_rejects_b00_target_mismatch() -> None:
    """B00 任一 target_tvt 变化都必须在训练前触发合同异常。"""

    feature_table, b00_predictions = make_aligned_evaluation_tables()
    b00_predictions.loc[0, "target_tvt"] += 0.25

    with pytest.raises(ValueError, match="target_tvt"):
        validate_evaluation_alignment(
            feature_table,
            b00_predictions,
            expected_rows=5,
            expected_wells=5,
        )


def test_well_fold_assignments_must_match_registry_well_by_well() -> None:
    """逐折总行数相同也不够，每口井必须保持注册表中的 fold。"""

    feature_table, _ = make_aligned_evaluation_tables()
    registry = pd.DataFrame(
        {
            "well_id": feature_table["well_id"],
            "fold": feature_table["fold"],
        }
    )
    validate_well_fold_assignments(feature_table, registry)

    wrong_registry = registry.copy()
    wrong_registry.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="逐井 fold"):
        validate_well_fold_assignments(feature_table, wrong_registry)


def test_reused_fold_runtime_fingerprints_must_equal_cache_metadata() -> None:
    """源 folds 0–1 的 runtime 必须属于同一份 RF01a feature cache。"""

    source_fingerprint = "a" * 64
    cache_metadata = {"fingerprint": source_fingerprint, "rows": 5, "wells": 5}
    runtimes = {
        0: {
            "fold": 0,
            "fingerprint": source_fingerprint,
            "features": 17,
            "trees": 1734,
        },
        1: {
            "fold": 1,
            "fingerprint": source_fingerprint,
            "features": 17,
            "trees": 1734,
        },
    }

    assert (
        validate_reused_fold_runtime_contract(cache_metadata, runtimes)
        == source_fingerprint
    )

    runtimes[1]["fingerprint"] = "different-cache"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_reused_fold_runtime_contract(cache_metadata, runtimes)


def make_five_fold_prediction_inputs() -> tuple[
    pd.DataFrame,
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
]:
    """构造一折一行的固定源折、新折和完整评价键。"""

    feature_table, _ = make_aligned_evaluation_tables()
    all_fold_predictions: dict[int, pd.DataFrame] = {}
    for fold_id in range(5):
        fold_prediction = feature_table.loc[
            feature_table["fold"] == fold_id,
            ["well_id", "fold", "row_index", "target_tvt"],
        ].copy()
        fold_prediction["carry_tvt"] = fold_prediction["target_tvt"]
        fold_prediction["pred_tvt"] = fold_prediction["target_tvt"] + 0.5
        all_fold_predictions[fold_id] = fold_prediction

    reused = {fold_id: all_fold_predictions[fold_id] for fold_id in REUSED_FOLDS}
    new = {fold_id: all_fold_predictions[fold_id] for fold_id in NEW_FOLDS}
    return feature_table, reused, new


def test_merge_fixed_fold_predictions_uses_source_0_1_and_new_2_3_4() -> None:
    """完整预测只能由固定的两份源折和三份审计新折组成。"""

    feature_table, reused, new = make_five_fold_prediction_inputs()

    merged = merge_fixed_fold_predictions(
        reused,
        new,
        feature_table,
        expected_rows=5,
        expected_wells=5,
    )

    assert merged["fold"].tolist() == [0, 1, 2, 3, 4]
    assert compute_row_key_hash(merged) == compute_row_key_hash(feature_table)


def test_b00_comparison_explicitly_uses_b00_prediction_not_carry() -> None:
    """B00 比较的 baseline 必须是 B00 pred_tvt，不能落回默认 carry_tvt。"""

    candidate_predictions = pd.DataFrame(
        {
            "well_id": ["well_a", "well_b"],
            "fold": [0, 1],
            "row_index": [1, 1],
            "target_tvt": [0.0, 0.0],
            "carry_tvt": [0.0, 0.0],
            "pred_tvt": [1.0, 1.0],
        }
    )
    b00_predictions = candidate_predictions.copy()
    b00_predictions["pred_tvt"] = [2.0, 2.0]

    comparison_predictions, comparison_per_well, comparison_metrics = (
        build_b00_comparison(candidate_predictions, b00_predictions)
    )

    assert comparison_predictions["b00_pred_tvt"].tolist() == [2.0, 2.0]
    assert comparison_metrics["overall"]["micro_rmse"] == 1.0
    assert comparison_metrics["overall"]["baseline_micro_rmse"] == 2.0
    assert comparison_metrics["overall"]["micro_rmse_delta_vs_baseline"] == -1.0
    assert comparison_per_well["baseline_rmse"].tolist() == [2.0, 2.0]


def test_feature_importance_aggregates_all_five_fixed_folds() -> None:
    """平均重要性必须同时包含源 folds 0–1 和审计 folds 2–4。"""

    fold_importance = {
        fold_id: pd.DataFrame(
            {
                "feature": ["feature_a", "feature_b"],
                "gain": [float(fold_id), float(fold_id + 10)],
                "split": [float(fold_id + 1), float(fold_id + 11)],
            }
        )
        for fold_id in range(5)
    }

    mean_importance = aggregate_feature_importance(
        fold_importance,
        feature_columns=["feature_a", "feature_b"],
    )

    by_feature = mean_importance.set_index("feature")
    assert by_feature.loc["feature_a", "gain"] == 2.0
    assert by_feature.loc["feature_b", "gain"] == 12.0


def test_conclusion_keeps_original_decision_and_only_one_fixed_pattern() -> None:
    """结论首行不可改判，第二行只能是预注册三模式之一。"""

    conclusion = build_conclusion_text([-2.6, 0.1, -0.2, -0.3, -0.1])

    assert conclusion.splitlines() == [
        "原 RF01 仍不晋级，本审计不改变原决定。",
        "fold1是唯一反向折",
    ]


def test_fixed_fold_directories_keep_reused_and_new_outputs_separate() -> None:
    """汇总路径必须从 RF01a 取 0–1，并从审计目录取 2–4。"""

    source_dir = Path("artifacts/RF01a_huber_slopes_v1")
    audit_dir = Path("artifacts/RF01_stability_audit_v1")

    fold_directories = build_fixed_fold_directories(source_dir, audit_dir)

    assert fold_directories == {
        0: source_dir / "fold_0",
        1: source_dir / "fold_1",
        2: audit_dir / "fold_2",
        3: audit_dir / "fold_3",
        4: audit_dir / "fold_4",
    }
    assert fold_directories[0].parent != audit_dir
    assert fold_directories[1].parent != audit_dir


def test_preflight_missing_contract_fails_without_creating_audit_artifact() -> None:
    """预检输入缺失时立即报错，并且不能留下任何审计 artifact。"""

    missing_config_path = CLEAN_ROOT / "configs" / "__missing_rf01_audit__.json"
    audit_artifact_dir = CLEAN_ROOT / "artifacts" / "RF01_stability_audit_v1"
    artifact_existed_before = audit_artifact_dir.exists()

    with pytest.raises(FileNotFoundError, match="审计配置"):
        preflight_audit(missing_config_path, clean_root=CLEAN_ROOT)

    assert audit_artifact_dir.exists() is artifact_existed_before


def test_run_manifest_records_computed_eval_row_hash_and_lineage() -> None:
    """训练前运行清单必须保存计算值，而不是猜测旧 canonical hash。"""

    audit_config = {
        "experiment_id": "RF01_stability_audit_v1",
        "reused_folds": [0, 1],
        "new_folds": [2, 3, 4],
    }
    eval_row_hash = "a" * 64
    audit_fingerprint = "b" * 64
    source_fingerprint = "c" * 64

    manifest = build_run_manifest(
        audit_config,
        eval_row_hash=eval_row_hash,
        audit_fingerprint=audit_fingerprint,
        source_fingerprint=source_fingerprint,
        source_feature_cache=Path("artifacts/source/feature_cache.parquet"),
        source_artifact_dir=Path("artifacts/source"),
        artifact_dir=Path("artifacts/audit"),
    )

    assert manifest["eval_row_hash"] == eval_row_hash
    assert manifest["audit_fingerprint"] == audit_fingerprint
    assert manifest["source_fingerprint"] == source_fingerprint
    assert manifest["reused_folds"] == [0, 1]
    assert manifest["new_folds"] == [2, 3, 4]
    assert manifest["source_feature_cache"].endswith("feature_cache.parquet")


def test_final_output_contract_lists_every_required_summary_artifact() -> None:
    """最终汇总文件名必须完整，且不把 fold_0/fold_1 放进审计目录。"""

    assert set(FINAL_OUTPUT_RELATIVE_PATHS) == {
        "predictions.parquet",
        "metrics.json",
        "per_well.csv",
        "feature_list.json",
        "parameter_list.json",
        "feature_definition.json",
        "feature_lineage.json",
        "feature_quality.csv",
        "cache_manifest.json",
        "leakage_tests.json",
        "negative_control_metrics.json",
        "runtime.json",
        "config.json",
        "feature_importance.csv",
        "per_fold.csv",
        "slice_metrics.csv",
        "bootstrap_replicates.parquet",
        "comparison_vs_B00/predictions.parquet",
        "comparison_vs_B00/per_well.csv",
        "comparison_vs_B00/metrics.json",
        "conclusion.md",
    }
    assert all(not path.startswith("fold_0/") for path in FINAL_OUTPUT_RELATIVE_PATHS)
    assert all(not path.startswith("fold_1/") for path in FINAL_OUTPUT_RELATIVE_PATHS)


def test_feature_documents_and_quality_use_fixed_contract_schema() -> None:
    """17 列定义/lineage 完整，quality 第一列和字段顺序严格固定。"""

    source_config = json.loads(
        (CLEAN_ROOT / "configs" / "rf01a_huber_slopes_v1.json").read_text(
            encoding="utf-8"
        )
    )
    feature_columns = source_config["feature_columns"]
    feature_definition, feature_lineage = build_feature_contract_documents(
        feature_columns
    )

    required_lineage_fields = {
        "raw_source",
        "uses_TVT_input",
        "uses_hidden_GR",
        "uses_Typewell",
        "uses_PF_Beam",
        "requires_outer_train_fit",
        "fit_wells_hash",
        "transform_version",
        "unit",
    }
    assert set(feature_definition["features"]) == set(feature_columns)
    assert set(feature_lineage) == set(feature_columns)
    assert all(
        set(feature_lineage[feature]) == required_lineage_fields
        for feature in feature_columns
    )
    assert feature_lineage["gr_raw"]["uses_hidden_GR"] is True
    assert feature_lineage["u_huber_slope_500"]["uses_TVT_input"] is True
    assert all(
        feature_lineage[feature]["requires_outer_train_fit"] is False
        for feature in feature_columns
    )

    small_feature_table = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [2.0, 4.0, np.nan, 8.0],
        }
    )
    small_lineage = {
        "feature_a": {"raw_source": ["A"], "unit": "ft"},
        "feature_b": {"raw_source": ["B"], "unit": "ratio"},
    }
    quality = build_feature_quality_table(
        small_feature_table,
        ["feature_a", "feature_b"],
        small_lineage,
    )

    assert tuple(quality.columns) == FEATURE_QUALITY_COLUMNS
    assert quality.columns[0] == "feature"
    assert quality["feature"].tolist() == ["feature_a", "feature_b"]
    by_feature = quality.set_index("feature")
    assert by_feature.loc["feature_b", "finite_rate"] == 0.75
    assert by_feature.loc["feature_b", "unique_count"] == 3
    assert by_feature.loc["feature_a", "correlation_with_existing_feature"] == 1.0


def test_leakage_test_names_match_fixed_contract() -> None:
    """leakage_tests.json 必须逐项包含 AGENTS 固定的八个检查名。"""

    assert LEAKAGE_TEST_NAMES == (
        "hidden TVT deletion invariance",
        "hidden TVT mutation invariance",
        "surface deletion invariance",
        "outer-fold source exclusion",
        "PF cache fold match",
        "row hash match",
        "fixed-seed reproducibility",
        "negative-control status",
    )


def test_bootstrap_replicates_are_real_and_recompute_the_saved_summary() -> None:
    """2000 行 B00 配对明细必须与统一 seed=42 摘要逐项一致。"""

    per_well = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "rows": [2, 3],
            "prediction_sse": [1.0, 9.0],
            "baseline_sse": [4.0, 12.0],
        }
    )
    replicates, summary = build_paired_bootstrap_artifacts(
        per_well,
        candidate_id="RF01a_huber_slopes_v1",
        baseline_id="B00_simple_lgbm_v1",
        n_resamples=2000,
        seed=42,
    )

    assert list(replicates.columns) == [
        "replicate",
        "seed",
        "candidate_id",
        "baseline_id",
        "sampled_wells",
        "unique_sampled_wells",
        "sampled_rows",
        "candidate_sse",
        "baseline_sse",
        "candidate_rmse",
        "baseline_rmse",
        "delta_rmse",
    ]
    assert len(replicates) == 2000
    assert set(replicates["baseline_id"]) == {"B00_simple_lgbm_v1"}
    assert set(replicates["seed"]) == {42}
    recomputed_summary = {
        "n_resamples": 2000,
        "seed": 42,
        "mean_delta": float(replicates["delta_rmse"].mean()),
        "ci95_low": float(replicates["delta_rmse"].quantile(0.025)),
        "ci95_high": float(replicates["delta_rmse"].quantile(0.975)),
        "probability_better": float((replicates["delta_rmse"] < 0.0).mean()),
    }
    assert summary == recomputed_summary
    assert summary == paired_well_bootstrap(per_well, n_resamples=2000, seed=42)


def test_completed_folds_keep_legacy_training_fingerprint_after_report_change() -> None:
    """仅报告代码变化时，完整合法的新折必须纯复用旧 training fingerprint。"""

    legacy_training_fingerprint = "a" * 64
    reporting_fingerprint = "b" * 64
    expected_contract = {
        "experiment_id": "RF01_stability_audit_v1",
        "eval_row_hash": "c" * 64,
        "source_fingerprint": "d" * 64,
        "fold_registry_sha256": "e" * 64,
        "expected_rows": 3_783_989,
        "expected_wells": 773,
        "model_config": "configs/lgbm_feature_baseline_v1.json",
        "reused_folds": [0, 1],
        "new_folds": [2, 3, 4],
    }
    existing_manifest = {
        **expected_contract,
        "audit_fingerprint": legacy_training_fingerprint,
    }
    new_fold_runtimes = {
        fold_id: {"fold": fold_id, "fingerprint": legacy_training_fingerprint}
        for fold_id in NEW_FOLDS
    }

    training_fingerprint, policy = select_training_fingerprint(
        reporting_fingerprint=reporting_fingerprint,
        existing_manifest=existing_manifest,
        new_fold_runtimes=new_fold_runtimes,
        expected_contract=expected_contract,
    )

    assert training_fingerprint == legacy_training_fingerprint
    assert policy == "legacy_completed_folds_reuse"

    changed_manifest = dict(existing_manifest)
    changed_manifest["eval_row_hash"] = "f" * 64
    with pytest.raises(ValueError, match="eval_row_hash"):
        select_training_fingerprint(
            reporting_fingerprint=reporting_fingerprint,
            existing_manifest=changed_manifest,
            new_fold_runtimes=new_fold_runtimes,
            expected_contract=expected_contract,
        )


def test_existing_prediction_file_is_validated_and_preserved_byte_for_byte() -> None:
    """已有预测逐位相同时不重写，任一值变化时拒绝覆盖。"""

    prediction = pd.DataFrame(
        {
            "well_id": ["a", "b"],
            "fold": [0, 1],
            "row_index": [1, 1],
            "target_tvt": [10.0, 20.0],
            "carry_tvt": [9.0, 19.0],
            "pred_tvt": [10.5, 20.5],
        }
    )
    with tempfile.TemporaryDirectory(
        prefix="rf01_prediction_preserve_",
        dir=CLEAN_ROOT / "tests",
    ) as temporary_directory:
        prediction_path = Path(temporary_directory) / "predictions.parquet"
        prediction.to_parquet(prediction_path, index=False)
        hash_before = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

        record = preserve_or_write_prediction_table(prediction, prediction_path)

        assert record["action"] == "preserved_existing_identical"
        assert record["sha256_before"] == hash_before
        assert record["sha256_after"] == hash_before

        changed_prediction = prediction.copy()
        changed_prediction.loc[0, "pred_tvt"] += 1.0
        with pytest.raises(ValueError, match="已有预测"):
            preserve_or_write_prediction_table(changed_prediction, prediction_path)
        assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == hash_before


def test_diagnostic_reporting_tables_are_honest_and_use_b00_fold_metrics() -> None:
    """无预注册负对照/切片时写明不适用，同时保存全体与逐折真实指标。"""

    comparison_metrics = {
        "baseline_id": "B00_simple_lgbm_v1",
        "overall": {
            "rows": 5,
            "wells": 2,
            "micro_rmse": 1.0,
            "baseline_micro_rmse": 2.0,
            "micro_rmse_delta_vs_baseline": -1.0,
            "macro_well_rmse": 1.2,
            "median_well_rmse": 1.1,
            "p90_well_rmse": 1.5,
            "worst_well_rmse": 2.0,
            "well_win_rate": 0.5,
        },
        "folds": [
            {
                "fold": 0,
                "rows": 5,
                "wells": 2,
                "micro_rmse": 1.0,
                "baseline_micro_rmse": 2.0,
                "micro_rmse_delta_vs_baseline": -1.0,
                "macro_well_rmse": 1.2,
                "median_well_rmse": 1.1,
                "p90_well_rmse": 1.5,
                "worst_well_rmse": 2.0,
                "well_win_rate": 0.5,
            }
        ],
    }

    per_fold, slice_metrics, negative_control = build_diagnostic_reporting_artifacts(
        comparison_metrics
    )

    assert per_fold.loc[0, "b00_micro_rmse"] == 2.0
    assert per_fold.loc[0, "rf01a_minus_b00_micro_rmse"] == -1.0
    assert slice_metrics["slice"].tolist() == ["all_natural_hidden_rows"]
    assert slice_metrics.loc[0, "status"] == "full_population_only"
    assert slice_metrics.loc[0, "b00_micro_rmse"] == 2.0
    assert negative_control["status"] == "not_applicable"
    assert negative_control["experiment_type"] == "diagnostic_only"
    assert negative_control["extra_randomized_control_run"] is False


def test_leakage_document_uses_fixed_names_and_honest_evidence_statuses() -> None:
    """未重跑的不变性不得伪称通过，实际 hash/fold 检查必须保存证据。"""

    feature_lineage = {
        "feature_a": {
            "raw_source": ["MD"],
            "uses_PF_Beam": False,
        }
    }
    cache_manifest = {
        "fold_artifacts": [
            {
                "outer_fold": 0,
                "train_validation_intersection_count": 0,
            }
        ],
        "full_prediction_row_hash": "a" * 64,
    }
    prediction_records = {
        "predictions.parquet": {
            "sha256_before": "b" * 64,
            "sha256_after": "b" * 64,
        }
    }
    negative_control = {"status": "not_applicable"}

    leakage_document = build_leakage_tests_document(
        eval_row_hash="a" * 64,
        feature_lineage=feature_lineage,
        cache_manifest=cache_manifest,
        prediction_file_records=prediction_records,
        negative_control=negative_control,
    )

    assert tuple(leakage_document["checks"]) == LEAKAGE_TEST_NAMES
    assert (
        leakage_document["checks"]["hidden TVT deletion invariance"]["status"]
        == "not_rerun_diagnostic_reuse"
    )
    assert leakage_document["checks"]["surface deletion invariance"]["status"] == "passed_static_lineage"
    assert leakage_document["checks"]["outer-fold source exclusion"]["status"] == "passed"
    assert leakage_document["checks"]["row hash match"]["status"] == "passed"
    assert leakage_document["checks"]["fixed-seed reproducibility"]["status"] == "passed_prediction_sha_unchanged"
    assert leakage_document["checks"]["negative-control status"]["status"] == "not_applicable"


def test_fold_cache_manifest_records_actual_well_row_and_file_hashes() -> None:
    """每折 manifest 必须保存真实 fit/validation、预测行和四类文件 SHA。"""

    registry = pd.DataFrame(
        {
            "well_id": ["well_0", "well_1"],
            "fold": [0, 1],
        }
    )
    prediction = pd.DataFrame(
        {
            "well_id": ["well_0"],
            "fold": [0],
            "row_index": [1],
            "target_tvt": [10.0],
            "carry_tvt": [9.0],
            "pred_tvt": [10.5],
        }
    )
    runtime = {"fold": 0, "fingerprint": "a" * 64}
    with tempfile.TemporaryDirectory(
        prefix="rf01_cache_manifest_",
        dir=CLEAN_ROOT / "tests",
    ) as temporary_directory:
        fold_dir = Path(temporary_directory) / "fold_0"
        fold_dir.mkdir()
        prediction.to_parquet(fold_dir / "predictions.parquet", index=False)
        (fold_dir / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
        (fold_dir / "feature_importance.csv").write_text(
            "feature,gain,split\nfeature_a,1,1\n",
            encoding="utf-8",
        )
        (fold_dir / "model.txt").write_text("model", encoding="utf-8")

        manifest = build_fold_cache_manifest(
            fold_id=0,
            artifact_source="reused_RF01a",
            registry=registry,
            prediction=prediction,
            runtime=runtime,
            fold_dir=fold_dir,
        )

    assert manifest["outer_fold"] == 0
    assert manifest["fit_wells_count"] == 1
    assert manifest["validation_wells_count"] == 1
    assert manifest["train_validation_intersection_count"] == 0
    assert len(manifest["fit_wells_hash"]) == 64
    assert len(manifest["validation_wells_hash"]) == 64
    assert manifest["prediction_row_hash"] == compute_row_key_hash(prediction)
    assert set(manifest["file_sha256"]) == {
        "predictions.parquet",
        "runtime.json",
        "feature_importance.csv",
        "model.txt",
    }
    assert all(len(value) == 64 for value in manifest["file_sha256"].values())


def test_row_key_hash_rejects_empty_or_duplicate_key_sets() -> None:
    """空评价集或重复行键不能生成貌似合法的 SHA-256。"""

    empty_row_keys = pd.DataFrame(columns=["well_id", "fold", "row_index"])
    duplicate_row_keys = pd.DataFrame(
        {
            "well_id": ["well_a", "well_a"],
            "fold": [0, 0],
            "row_index": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="为空"):
        compute_row_key_hash(empty_row_keys)
    with pytest.raises(ValueError, match="重复"):
        compute_row_key_hash(duplicate_row_keys)


def test_source_fingerprint_comes_from_cache_metadata_and_reused_runtimes() -> None:
    """后来扩大的 generic runner 不能倒推并误杀历史 RF01a 指纹。"""

    source_artifact_dir = CLEAN_ROOT / "artifacts" / "RF01a_huber_slopes_v1"
    cache_metadata_path = source_artifact_dir / "feature_cache.meta.json"

    cache_metadata, fold_runtimes, source_fingerprint = read_source_runtime_contract(
        cache_metadata_path,
        source_artifact_dir,
    )

    expected_source_fingerprint = (
        "311049158a7eb64f775fa86766c35834ff0f4d3d19f3d0e6969709206771e5a5"
    )
    assert source_fingerprint == expected_source_fingerprint
    assert cache_metadata["fingerprint"] == expected_source_fingerprint
    assert {runtime["fingerprint"] for runtime in fold_runtimes.values()} == {
        expected_source_fingerprint
    }


def test_matching_new_fold_cache_is_fully_validated_before_training() -> None:
    """会被 generic runner 复用的新折必须先有完整且合法的四类产物。"""

    feature_table = pd.DataFrame(
        {
            "well_id": ["well_2", "well_3", "well_4"],
            "fold": [2, 3, 4],
            "row_index": [1, 1, 1],
            "target_tvt": [102.0, 103.0, 104.0],
            "carry_tvt": [101.0, 102.0, 103.0],
        }
    )
    audit_fingerprint = "d" * 64

    # 系统临时目录在本机受 ACL 限制，因此把短生命周期目录放在 tests 下。
    with tempfile.TemporaryDirectory(
        prefix="rf01_reuse_",
        dir=CLEAN_ROOT / "tests",
    ) as temporary_directory:
        audit_artifact_dir = Path(temporary_directory)
        fold_dir = audit_artifact_dir / "fold_2"
        fold_dir.mkdir()

        fold_prediction = pd.DataFrame(
            {
                "well_id": ["well_2"],
                "fold": [2],
                "row_index": [1],
                "target_tvt": [102.0],
                "carry_tvt": [101.0],
                "pred_tvt": [102.5],
            }
        )
        fold_prediction.to_parquet(fold_dir / "predictions.parquet", index=False)
        (fold_dir / "model.txt").write_text("model", encoding="utf-8")
        (fold_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "fold": 2,
                    "fingerprint": audit_fingerprint,
                    "features": 1,
                    "trees": 1734,
                    "validation_rows": 1,
                    "validation_wells": 1,
                    "seconds": 1.0,
                }
            ),
            encoding="utf-8",
        )

        # generic runner 此时会复用前三个文件；importance 缺失必须提前阻断。
        with pytest.raises(FileNotFoundError, match="feature importance"):
            validate_existing_new_fold_reuse(
                feature_table,
                audit_artifact_dir,
                audit_fingerprint=audit_fingerprint,
                feature_columns=["feature_a"],
            )

        pd.DataFrame(
            {"feature": ["feature_a"], "gain": [1.0], "split": [2.0]}
        ).to_csv(fold_dir / "feature_importance.csv", index=False)

        # carry_tvt 会进入根 metrics，必须与 feature cache 逐位一致。
        corrupted_prediction = fold_prediction.copy()
        corrupted_prediction["carry_tvt"] = 999.0
        corrupted_prediction.to_parquet(
            fold_dir / "predictions.parquet",
            index=False,
        )
        with pytest.raises(ValueError, match="carry_tvt"):
            validate_existing_new_fold_reuse(
                feature_table,
                audit_artifact_dir,
                audit_fingerprint=audit_fingerprint,
                feature_columns=["feature_a"],
            )
        fold_prediction.to_parquet(fold_dir / "predictions.parquet", index=False)

        # finalize 必读 seconds；缺失必须在任何其他折训练前阻断。
        runtime_without_seconds = json.loads(
            (fold_dir / "runtime.json").read_text(encoding="utf-8")
        )
        runtime_without_seconds.pop("seconds")
        (fold_dir / "runtime.json").write_text(
            json.dumps(runtime_without_seconds),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="seconds"):
            validate_existing_new_fold_reuse(
                feature_table,
                audit_artifact_dir,
                audit_fingerprint=audit_fingerprint,
                feature_columns=["feature_a"],
            )
        runtime_without_seconds["seconds"] = 1.0
        (fold_dir / "runtime.json").write_text(
            json.dumps(runtime_without_seconds),
            encoding="utf-8",
        )

        reusable_folds = validate_existing_new_fold_reuse(
            feature_table,
            audit_artifact_dir,
            audit_fingerprint=audit_fingerprint,
            feature_columns=["feature_a"],
        )

        assert reusable_folds == [2]
