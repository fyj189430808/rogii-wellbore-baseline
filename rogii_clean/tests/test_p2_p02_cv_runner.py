"""P2-P02 四尺度 PF 路径固定 41 特征 runner 测试。"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_p2_p02_multiscale_pf_paths_cv.py"
CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_p02_multiscale_pf_paths_v1.json"
MODEL_CONFIG_PATH = CLEAN_ROOT / "configs" / "lgbm_feature_baseline_v1.json"
P01_FEATURE_LIST_PATH = (
    CLEAN_ROOT
    / "artifacts"
    / "P2_P01_multiseed_pf_mean_v1"
    / "model_cv"
    / "feature_list.json"
)


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    """在工作区旁创建 Windows 可写临时目录。"""

    with tempfile.TemporaryDirectory(
        prefix=".pytest_p2_p02_cv_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _load_runner():
    """按脚本路径加载 P02 runner，不执行命令行 main。"""

    assert RUNNER_PATH.is_file(), f"P02 runner 尚未实现：{RUNNER_PATH.name}"
    specification = importlib.util.spec_from_file_location(
        "run_p2_p02_multiscale_pf_paths_cv",
        RUNNER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    """读取冻结 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_only_accepts_exact_folds01_or_all() -> None:
    """P02 只能先跑 0、1 或按凭证续跑 all。"""

    runner = _load_runner()
    assert runner.parse_fold_spec("0,1") == [0, 1]
    assert runner.parse_fold_spec("all") == [0, 1, 2, 3, 4]

    for invalid in ["", "0", "1", "4,2", "2,3,4", "0,1,2,3,4", "0, 1"]:
        with pytest.raises(ValueError, match="fold|折"):
            runner.parse_fold_spec(invalid)


def test_contract_is_exactly_p01_37_plus_all_four_scales() -> None:
    """正式模型必须保留 P01 37 列并把预注册四尺度整体加入。"""

    runner = _load_runner()
    config = _read_json(CONFIG_PATH)
    model_config = _read_json(MODEL_CONFIG_PATH)
    p01_feature_manifest = _read_json(P01_FEATURE_LIST_PATH)

    features = runner.validate_frozen_contract(
        config,
        model_config,
        p01_feature_manifest,
    )

    assert len(features) == 41
    assert features[:37] == p01_feature_manifest["features"]
    assert features[37:] == [
        "pf128_scale_3_delta",
        "pf128_scale_5_delta",
        "pf128_scale_8_delta",
        "pf128_scale_12_delta",
    ]

    changed_scales = copy.deepcopy(config)
    changed_scales["new_feature_names"] = ["pf128_scale_8_delta"]
    with pytest.raises(ValueError, match="四|scale|41|特征"):
        runner.validate_frozen_contract(
            changed_scales,
            model_config,
            p01_feature_manifest,
        )

    changed_count = copy.deepcopy(config)
    changed_count["formal_feature_count"] = 40
    with pytest.raises(ValueError, match="41|feature_count|特征"):
        runner.validate_frozen_contract(
            changed_count,
            model_config,
            p01_feature_manifest,
        )


@pytest.mark.parametrize(
    ("parameter_name", "changed_value"),
    [
        ("n_estimators", 1733),
        ("learning_rate", 0.02),
        ("num_leaves", 31),
        ("reg_lambda", 1.0),
    ],
)
def test_complete_lgbm_parameters_are_hard_frozen(
    parameter_name: str,
    changed_value: object,
) -> None:
    """P02 不得借四尺度实验调整任何模型参数。"""

    runner = _load_runner()
    config = _read_json(CONFIG_PATH)
    model_config = _read_json(MODEL_CONFIG_PATH)
    p01_feature_manifest = _read_json(P01_FEATURE_LIST_PATH)
    changed = copy.deepcopy(model_config)
    changed["params"][parameter_name] = changed_value

    with pytest.raises(ValueError, match=parameter_name + "|模型|完整参数"):
        runner.validate_frozen_contract(
            config,
            changed,
            p01_feature_manifest,
        )


def test_p01_baseline_and_all_external_hashes_are_code_frozen() -> None:
    """P01 基线和输入来源哈希不能与文件一起事后改写。"""

    runner = _load_runner()
    config = _read_json(CONFIG_PATH)
    runner.validate_declared_external_hashes(config)

    assert config["model_config_sha256"] == runner.FROZEN_MODEL_CONFIG_SHA256
    assert (
        config["baseline_predictions_sha256"]
        == runner.FROZEN_EXTERNAL_SHA256["baseline_predictions"]
    )

    changed = copy.deepcopy(config)
    changed["baseline_predictions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="baseline_predictions|P01|SHA"):
        runner.validate_declared_external_hashes(changed)


@pytest.mark.parametrize(
    "forbidden_feature",
    [
        "pf128_seed0_delta",
        "pf128_seed_std",
        "target_delta",
        "ANCC",
        "oracle_best_scale",
    ],
)
def test_model_policy_rejects_diagnostics_targets_surfaces_and_oracle(
    forbidden_feature: str,
) -> None:
    """P01 诊断列和评分期信息不能进入 P02 的正式 41 列。"""

    runner = _load_runner()
    p01_features = _read_json(P01_FEATURE_LIST_PATH)["features"]
    illegal = [*p01_features, *runner.FROZEN_SCALE_FEATURES[:-1], forbidden_feature]

    with pytest.raises(ValueError, match="禁止|41|正式特征"):
        runner.validate_model_feature_names(illegal, p01_features)


def _p01_legal_cache(
    well_id: str,
    row_indices: list[int],
    fingerprint: str,
    anchor: float,
) -> pd.DataFrame:
    """构造 P01 生成器的完整合法缓存 schema。"""

    rows = len(row_indices)
    mean = np.arange(1, rows + 1, dtype=np.float32)
    anchor32 = np.float32(anchor)
    return pd.DataFrame(
        {
            "well_id": [well_id] * rows,
            "row_index": np.asarray(row_indices, dtype=np.int64),
            "last_visible_tvt": np.full(rows, anchor32, dtype=np.float32),
            "pf128_mean_tvt": (anchor32 + mean).astype(np.float32),
            "pf128_mean_delta": mean,
            "pf128_seed0_delta": mean + np.float32(0.1),
            "pf128_scale_3_delta": mean + np.float32(3.0),
            "pf128_scale_5_delta": mean + np.float32(5.0),
            "pf128_scale_8_delta": mean + np.float32(8.0),
            "pf128_scale_12_delta": mean + np.float32(12.0),
            "pf128_seed_std": np.full(rows, 0.5, dtype=np.float32),
            "_cache_fingerprint": [fingerprint] * rows,
        }
    )


def test_p01_cache_merge_adds_mean_and_four_scales_but_not_diagnostics(
    local_tmp_path: Path,
) -> None:
    """P02 只从完整 P01 缓存选择五条正式路径列。"""

    runner = _load_runner()
    cache_dir = local_tmp_path / "legal_cache"
    cache_dir.mkdir()
    fingerprint = "p01-current"
    _p01_legal_cache("a", [10, 12], fingerprint, 100.0).to_parquet(
        cache_dir / "a.parquet",
        index=False,
    )
    _p01_legal_cache("b", [3], fingerprint, 200.0).to_parquet(
        cache_dir / "b.parquet",
        index=False,
    )
    feature_table = pd.DataFrame(
        {
            "well_id": ["b", "a", "a"],
            "row_index": [3, 12, 10],
            "last_visible_tvt": [200.0, 100.0, 100.0],
            "target_delta": [0.0, 0.0, 0.0],
        }
    )
    registry = pd.DataFrame(
        {"well_id": ["a", "b"], "fold": [0, 1], "hidden_rows": [2, 1]}
    )

    merged = runner.merge_p01_path_cache(
        feature_table,
        registry,
        cache_dir,
        fingerprint,
    )

    assert merged[["well_id", "row_index"]].values.tolist() == [
        ["b", 3],
        ["a", 12],
        ["a", 10],
    ]
    for feature_name in ["pf128_mean_delta", *runner.FROZEN_SCALE_FEATURES]:
        assert feature_name in merged.columns
    assert "pf128_seed0_delta" not in merged.columns
    assert "pf128_seed_std" not in merged.columns


def test_p01_cache_merge_rejects_fingerprint_anchor_and_leak(
    local_tmp_path: Path,
) -> None:
    """旧 P01 指纹、错误路径起点或标签列都必须立即停止。"""

    runner = _load_runner()
    cache_dir = local_tmp_path / "legal_cache"
    cache_dir.mkdir()
    cache = _p01_legal_cache("a", [10], "old", 100.0)
    cache.to_parquet(cache_dir / "a.parquet", index=False)
    feature_table = pd.DataFrame(
        {"well_id": ["a"], "row_index": [10], "last_visible_tvt": [100.0]}
    )
    registry = pd.DataFrame(
        {"well_id": ["a"], "fold": [0], "hidden_rows": [1]}
    )

    with pytest.raises(ValueError, match="指纹"):
        runner.merge_p01_path_cache(
            feature_table,
            registry,
            cache_dir,
            "current",
        )

    cache["_cache_fingerprint"] = "current"
    cache["last_visible_tvt"] = np.float32(101.0)
    cache.to_parquet(cache_dir / "a.parquet", index=False)
    with pytest.raises(ValueError, match="last_visible_tvt|起点"):
        runner.merge_p01_path_cache(
            feature_table,
            registry,
            cache_dir,
            "current",
        )

    cache["last_visible_tvt"] = np.float32(100.0)
    cache["target_tvt"] = np.float32(999.0)
    cache.to_parquet(cache_dir / "a.parquet", index=False)
    with pytest.raises(ValueError, match="禁止|未登记"):
        runner.merge_p01_path_cache(
            feature_table,
            registry,
            cache_dir,
            "current",
        )


def test_program_controls_and_generator_fingerprint_must_match() -> None:
    """P02 必须继承 P01 已通过的隐藏真值隔离控制。"""

    runner = _load_runner()
    config = _read_json(CONFIG_PATH)
    controls = {
        "experiment_fingerprint": config["p01_generator_fingerprint"],
        "maximum_repeat_difference_ft": 0.0,
        "maximum_concurrent_difference_ft": 0.0,
        "maximum_hidden_tvt_mutation_difference_ft": 0.0,
        "success_checks": {"feature_generation_supported": True},
    }
    runner.validate_p01_program_controls(
        config,
        controls,
        computed_generator_fingerprint=config["p01_generator_fingerprint"],
    )

    wrong = copy.deepcopy(controls)
    wrong["experiment_fingerprint"] = "old"
    with pytest.raises(ValueError, match="指纹"):
        runner.validate_p01_program_controls(
            config,
            wrong,
            computed_generator_fingerprint=config["p01_generator_fingerprint"],
        )


def _tiny_predictions(candidate_better: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造两折候选和 P01 基线预测。"""

    target = [10.0, 10.0, 20.0, 20.0]
    p01_prediction = [12.0, 12.0, 22.0, 22.0]
    candidate_prediction = (
        [11.0, 11.0, 21.0, 21.0]
        if candidate_better
        else [13.0, 13.0, 23.0, 23.0]
    )
    common = {
        "well_id": ["a", "a", "b", "b"],
        "fold": [0, 0, 1, 1],
        "row_index": [0, 1, 0, 1],
        "target_tvt": target,
        "carry_tvt": [9.0, 9.0, 19.0, 19.0],
    }
    return (
        pd.DataFrame({**common, "pred_tvt": candidate_prediction}),
        pd.DataFrame({**common, "pred_tvt": p01_prediction}),
    )


def test_comparison_and_gate_use_p01_not_p2b00_or_carry() -> None:
    """P02 的全部晋级指标必须相对当前最佳 P01。"""

    runner = _load_runner()
    candidate, p01 = _tiny_predictions(candidate_better=True)
    _, comparison = runner.build_p01_comparison(candidate, p01)
    checks = runner.evaluate_model_success_checks(
        comparison,
        _read_json(CONFIG_PATH),
        stage="folds01",
    )

    assert comparison["comparison"] == "candidate_minus_P01"
    assert comparison["overall"]["micro_rmse"] == pytest.approx(1.0)
    assert comparison["overall"]["baseline_micro_rmse"] == pytest.approx(2.0)
    assert checks["folds01_combined_improvement_ft"] == pytest.approx(1.0)
    assert checks["folds01_pass"] is True


def test_folds2_to4_require_current_passed_folds01_file(
    local_tmp_path: Path,
) -> None:
    """后三折只能由当前 P02 指纹且通过的磁盘凭证解锁。"""

    runner = _load_runner()
    path = local_tmp_path / "comparison_vs_p01_folds01.json"
    valid = {
        "cv_fingerprint": "current",
        "success_checks": {"stage": "folds01", "folds01_pass": True},
    }
    path.write_text(json.dumps(valid), encoding="utf-8")
    runner.load_passed_folds01_comparison(local_tmp_path, "current")

    invalid = copy.deepcopy(valid)
    invalid["cv_fingerprint"] = "old"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="指纹"):
        runner.load_passed_folds01_comparison(local_tmp_path, "current")

    invalid = copy.deepcopy(valid)
    invalid["success_checks"]["folds01_pass"] = False
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="未通过|门槛"):
        runner.load_passed_folds01_comparison(local_tmp_path, "current")


def test_reverse_control_reverses_each_scale_within_well_only() -> None:
    """负对照保持逐井分布和行身份，只破坏四尺度的当前位置对应。"""

    runner = _load_runner()
    table = pd.DataFrame(
        {
            "well_id": ["b", "a", "a", "b"],
            "row_index": [2, 2, 1, 1],
            "pf128_mean_delta": [90.0, 80.0, 70.0, 60.0],
            "pf128_scale_3_delta": [20.0, 2.0, 1.0, 10.0],
            "pf128_scale_5_delta": [40.0, 4.0, 3.0, 30.0],
            "pf128_scale_8_delta": [60.0, 6.0, 5.0, 50.0],
            "pf128_scale_12_delta": [80.0, 8.0, 7.0, 70.0],
        }
    )
    before = table.copy(deep=True)
    reversed_table = runner.reverse_scale_paths_within_well(table)

    pd.testing.assert_frame_equal(table, before)
    assert reversed_table["pf128_mean_delta"].tolist() == before[
        "pf128_mean_delta"
    ].tolist()
    # 原表 a 井按 row_index 是 [1,2]，scale3 为 [1,2]；反转后应为 [2,1]。
    a_rows = reversed_table.loc[reversed_table["well_id"].eq("a")].sort_values(
        "row_index"
    )
    assert a_rows["pf128_scale_3_delta"].tolist() == [2.0, 1.0]
    for feature_name in runner.FROZEN_SCALE_FEATURES:
        for well_id in ["a", "b"]:
            original_values = sorted(
                before.loc[before["well_id"].eq(well_id), feature_name].tolist()
            )
            reversed_values = sorted(
                reversed_table.loc[
                    reversed_table["well_id"].eq(well_id), feature_name
                ].tolist()
            )
            assert reversed_values == original_values


def test_oracle_scale_audit_is_per_well_and_diagnostic_only() -> None:
    """oracle 只能事后逐井选裸路径，不产生可并入模型的行级列。"""

    runner = _load_runner()
    table = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "target_tvt": [10.0, 10.0, 20.0, 20.0],
            "last_visible_tvt": [0.0, 0.0, 0.0, 0.0],
            "pf128_scale_3_delta": [10.0, 10.0, 25.0, 25.0],
            "pf128_scale_5_delta": [15.0, 15.0, 20.0, 20.0],
            "pf128_scale_8_delta": [14.0, 14.0, 24.0, 24.0],
            "pf128_scale_12_delta": [13.0, 13.0, 23.0, 23.0],
        }
    )

    audit = runner.build_oracle_scale_audit(table)

    assert audit["diagnostic_only"] is True
    assert audit["best_scale_counts"] == {"scale_3": 1, "scale_5": 1}
    assert audit["oracle_best_per_well_micro_rmse"] == pytest.approx(0.0)
    assert len(audit["per_well"]) == 2

