"""P2-P01 固定 37 特征 LightGBM CV runner 测试。"""

from __future__ import annotations

import copy
import hashlib
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

RUNNER_PATH = CLEAN_ROOT / "scripts" / "run_p2_p01_multiseed_pf_mean_cv.py"
P01_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_p01_multiseed_pf_mean_v1.json"
P2B00_CONFIG_PATH = CLEAN_ROOT / "configs" / "p2_cv00_group5_c01_v1.json"
MODEL_CONFIG_PATH = CLEAN_ROOT / "configs" / "lgbm_feature_baseline_v1.json"
BASELINE_FEATURE_LIST_PATH = (
    CLEAN_ROOT / "artifacts" / "P2_CV00_group5_c01_v1" / "feature_list.json"
)


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    """在工作区旁建立可写临时目录，避免 Windows 系统临时目录权限差异。"""

    with tempfile.TemporaryDirectory(
        prefix=".pytest_p2_p01_cv_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def _load_runner():
    """按文件路径加载新 runner，测试不触发命令行主流程。"""

    assert RUNNER_PATH.is_file(), f"CV runner 尚未实现：{RUNNER_PATH.name}"
    specification = importlib.util.spec_from_file_location(
        "run_p2_p01_multiseed_pf_mean_cv",
        RUNNER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    """读取测试所需的冻结 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def test_fold_spec_allows_only_exact_folds01_or_all() -> None:
    """正式入口只允许预注册的两阶段运行，不能从中间折直接启动。"""

    runner = _load_runner()

    assert runner.parse_fold_spec("0,1") == [0, 1]
    assert runner.parse_fold_spec("all") == [0, 1, 2, 3, 4]

    for invalid in [
        "",
        "0",
        "1",
        "4,2",
        "2,3,4",
        "0,0",
        "5",
        "0,a",
        "0,1,2,3,4",
    ]:
        with pytest.raises(ValueError, match="fold|折"):
            runner.parse_fold_spec(invalid)


def test_contract_is_exactly_frozen_c01_plus_one_formal_feature() -> None:
    """模型必须是原 C01 36 列加且只加 pf128_mean_delta。"""

    runner = _load_runner()
    p01_config = _read_json(P01_CONFIG_PATH)
    baseline_config = _read_json(P2B00_CONFIG_PATH)
    model_config = _read_json(MODEL_CONFIG_PATH)
    baseline_feature_list = _read_json(BASELINE_FEATURE_LIST_PATH)

    features = runner.validate_frozen_contract(
        p01_config,
        baseline_config,
        model_config,
        baseline_feature_list,
    )

    assert len(features) == 37
    assert features[:36] == baseline_feature_list["features"]
    assert features[-1] == "pf128_mean_delta"
    assert features.count("pf128_mean_delta") == 1

    changed_count = copy.deepcopy(p01_config)
    changed_count["formal_model_feature_count"] = 38
    with pytest.raises(ValueError, match="37|feature_count|特征"):
        runner.validate_frozen_contract(
            changed_count,
            baseline_config,
            model_config,
            baseline_feature_list,
        )

    changed_model = copy.deepcopy(model_config)
    changed_model["params"]["n_estimators"] = 1733
    with pytest.raises(ValueError, match="1734|模型|n_estimators"):
        runner.validate_frozen_contract(
            p01_config,
            baseline_config,
            changed_model,
            baseline_feature_list,
        )

    for parameter_name, changed_value in [
        ("learning_rate", 0.02),
        ("num_leaves", 31),
    ]:
        changed_parameter = copy.deepcopy(model_config)
        changed_parameter["params"][parameter_name] = changed_value
        with pytest.raises(ValueError, match=parameter_name + "|完整参数|模型"):
            runner.validate_frozen_contract(
                p01_config,
                baseline_config,
                changed_parameter,
                baseline_feature_list,
            )


def test_model_config_hash_is_independently_frozen_in_runner() -> None:
    """同时修改 P01 内声明的哈希和外部模型文件也不能绕开代码常量。"""

    runner = _load_runner()
    p01_config = _read_json(P01_CONFIG_PATH)

    assert p01_config["model_config_sha256"] == runner.FROZEN_MODEL_CONFIG_SHA256
    changed = copy.deepcopy(p01_config)
    changed["model_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model_config|模型|SHA"):
        runner.validate_declared_model_config_hash(changed)


@pytest.mark.parametrize(
    "forbidden_feature",
    [
        "pf128_seed0_delta",
        "pf128_scale_3_delta",
        "pf128_scale_5_delta",
        "pf128_scale_8_delta",
        "pf128_scale_12_delta",
        "pf128_seed_std",
        "target_delta",
        "ANCC",
        "oracle_rank",
    ],
)
def test_model_feature_policy_rejects_diagnostics_targets_and_surfaces(
    forbidden_feature: str,
) -> None:
    """seed0、scale、std、标签、surface 和 oracle 一律不能成为模型输入。"""

    runner = _load_runner()
    baseline_features = _read_json(BASELINE_FEATURE_LIST_PATH)["features"]
    illegal_features = [*baseline_features, forbidden_feature]

    with pytest.raises(ValueError, match="禁止|正式特征|37"):
        runner.validate_model_feature_names(illegal_features, baseline_features)


def _legal_cache_frame(
    well_id: str,
    row_indices: list[int],
    fingerprint: str,
    last_visible_tvt: float,
) -> pd.DataFrame:
    """构造与 P01 生成器完全同 schema 的小型单井缓存。"""

    row_count = len(row_indices)
    mean_delta = np.arange(1, row_count + 1, dtype=np.float32)
    anchor = np.float32(last_visible_tvt)
    return pd.DataFrame(
        {
            "well_id": [well_id] * row_count,
            "row_index": np.asarray(row_indices, dtype=np.int64),
            "last_visible_tvt": np.full(row_count, anchor, dtype=np.float32),
            "pf128_mean_tvt": (anchor + mean_delta).astype(np.float32),
            "pf128_mean_delta": mean_delta,
            "pf128_seed0_delta": mean_delta + np.float32(0.1),
            "pf128_scale_3_delta": mean_delta + np.float32(0.2),
            "pf128_scale_5_delta": mean_delta + np.float32(0.3),
            "pf128_scale_8_delta": mean_delta + np.float32(0.4),
            "pf128_scale_12_delta": mean_delta + np.float32(0.5),
            "pf128_seed_std": np.full(row_count, 0.5, dtype=np.float32),
            "_cache_fingerprint": [fingerprint] * row_count,
        }
    )


def test_per_well_p01_merge_preserves_base_order_and_adds_only_mean_delta(
    local_tmp_path: Path,
) -> None:
    """逐井缓存必须按 well_id-row_index 一对一并入，不能带入诊断列。"""

    runner = _load_runner()
    cache_dir = local_tmp_path / "legal_cache"
    cache_dir.mkdir()
    fingerprint = "frozen-p01-fingerprint"

    _legal_cache_frame("well_a", [10, 12], fingerprint, 100.0).to_parquet(
        cache_dir / "well_a.parquet",
        index=False,
    )
    _legal_cache_frame("well_b", [3], fingerprint, 200.0).to_parquet(
        cache_dir / "well_b.parquet",
        index=False,
    )
    feature_table = pd.DataFrame(
        {
            "well_id": ["well_b", "well_a", "well_a"],
            "row_index": [3, 12, 10],
            "last_visible_tvt": [200.0, 100.0, 100.0],
            "target_delta": [0.0, 0.0, 0.0],
        }
    )
    registry = pd.DataFrame(
        {
            "well_id": ["well_a", "well_b"],
            "fold": [0, 1],
            "hidden_rows": [2, 1],
        }
    )

    merged = runner.merge_p01_feature_cache(
        feature_table,
        registry,
        cache_dir,
        fingerprint,
    )

    assert merged[["well_id", "row_index"]].values.tolist() == [
        ["well_b", 3],
        ["well_a", 12],
        ["well_a", 10],
    ]
    assert merged["pf128_mean_delta"].tolist() == [1.0, 2.0, 1.0]
    assert "pf128_seed0_delta" not in merged.columns
    assert "pf128_scale_5_delta" not in merged.columns
    assert "pf128_seed_std" not in merged.columns


def test_per_well_p01_merge_rejects_wrong_fingerprint_and_anchor(
    local_tmp_path: Path,
) -> None:
    """旧代码缓存或不同路径起点即使键相同，也不能静默进入 CV。"""

    runner = _load_runner()
    cache_dir = local_tmp_path / "legal_cache"
    cache_dir.mkdir()
    cache = _legal_cache_frame("well_a", [10], "old-fingerprint", 100.0)
    cache.to_parquet(cache_dir / "well_a.parquet", index=False)
    registry = pd.DataFrame(
        {"well_id": ["well_a"], "fold": [0], "hidden_rows": [1]}
    )
    feature_table = pd.DataFrame(
        {
            "well_id": ["well_a"],
            "row_index": [10],
            "last_visible_tvt": [100.0],
        }
    )

    with pytest.raises(ValueError, match="指纹"):
        runner.merge_p01_feature_cache(
            feature_table,
            registry,
            cache_dir,
            "new-fingerprint",
        )

    cache["_cache_fingerprint"] = "new-fingerprint"
    cache["last_visible_tvt"] = np.float32(101.0)
    cache.to_parquet(cache_dir / "well_a.parquet", index=False)
    with pytest.raises(ValueError, match="last_visible_tvt|起点"):
        runner.merge_p01_feature_cache(
            feature_table,
            registry,
            cache_dir,
            "new-fingerprint",
        )


def _sha256(path: Path) -> str:
    """计算测试文件 SHA-256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_frozen_file_hashes_are_checked_before_loading_cache(
    local_tmp_path: Path,
) -> None:
    """baseline、模型、两份缓存、fold 和清单都必须先核对哈希。"""

    runner = _load_runner()
    names = [
        "baseline_config",
        "model_config",
        "base_feature_cache",
        "candidate_feature_cache",
        "fold_registry",
        "baseline_feature_list",
        "baseline_predictions",
    ]
    config: dict[str, str] = {}
    for name in names:
        if name == "model_config":
            config[name] = str(MODEL_CONFIG_PATH)
            config[f"{name}_sha256"] = runner.FROZEN_MODEL_CONFIG_SHA256
            continue
        path = local_tmp_path / f"{name}.bin"
        path.write_bytes(f"frozen-{name}".encode("utf-8"))
        config[name] = path.name
        config[f"{name}_sha256"] = _sha256(path)

    observed = runner.validate_frozen_file_hashes(config, local_tmp_path)
    assert set(observed) == set(names)

    changed_path = local_tmp_path / "candidate_feature_cache.bin"
    changed_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="candidate_feature_cache|SHA"):
        runner.validate_frozen_file_hashes(config, local_tmp_path)


def _tiny_predictions(candidate_better: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造两折两井的候选与 P2B00 预测。"""

    target = np.array([10.0, 10.0, 20.0, 20.0])
    baseline_prediction = np.array([12.0, 12.0, 22.0, 22.0])
    candidate_prediction = (
        np.array([11.0, 11.0, 21.0, 21.0])
        if candidate_better
        else np.array([13.0, 13.0, 23.0, 23.0])
    )
    common = {
        "well_id": ["well_a", "well_a", "well_b", "well_b"],
        "fold": [0, 0, 1, 1],
        "row_index": [0, 1, 0, 1],
        "target_tvt": target,
        "carry_tvt": [9.0, 9.0, 19.0, 19.0],
    }
    candidate = pd.DataFrame({**common, "pred_tvt": candidate_prediction})
    baseline = pd.DataFrame({**common, "pred_tvt": baseline_prediction})
    return candidate, baseline


def test_p2b00_comparison_and_folds01_gate_use_pooled_rows() -> None:
    """预筛门槛比较候选与 P2B00 本身，不得误比较 carry。"""

    runner = _load_runner()
    candidate, baseline = _tiny_predictions(candidate_better=True)
    _, comparison = runner.build_p2b00_comparison(candidate, baseline)
    config = _read_json(P01_CONFIG_PATH)
    checks = runner.evaluate_model_success_checks(comparison, config, stage="folds01")

    assert comparison["overall"]["micro_rmse"] == pytest.approx(1.0)
    assert comparison["overall"]["baseline_micro_rmse"] == pytest.approx(2.0)
    assert checks["folds01_combined_improvement_ft"] == pytest.approx(1.0)
    assert checks["maximum_single_fold_degradation_ft"] == pytest.approx(-1.0)
    assert checks["folds01_pass"] is True

    worse_candidate, baseline = _tiny_predictions(candidate_better=False)
    _, worse_comparison = runner.build_p2b00_comparison(worse_candidate, baseline)
    failed = runner.evaluate_model_success_checks(
        worse_comparison,
        config,
        stage="folds01",
    )
    assert failed["folds01_pass"] is False


def test_p2b00_comparison_rejects_key_or_target_disagreement() -> None:
    """候选和基线必须是完全相同的自然隐藏评价行与真值。"""

    runner = _load_runner()
    candidate, baseline = _tiny_predictions(candidate_better=True)
    baseline.loc[0, "target_tvt"] += 1.0

    with pytest.raises(ValueError, match="target|真值|行键"):
        runner.build_p2b00_comparison(candidate, baseline)


def test_folds2_to4_require_current_fingerprint_passed_folds01_file(
    local_tmp_path: Path,
) -> None:
    """继续后三折前必须从磁盘重读当前指纹且已通过的 folds01 结果。"""

    runner = _load_runner()
    comparison_path = local_tmp_path / "comparison_vs_p2b00_folds01.json"
    valid = {
        "cv_fingerprint": "current-fingerprint",
        "success_checks": {
            "stage": "folds01",
            "folds01_pass": True,
        },
    }
    comparison_path.write_text(
        json.dumps(valid, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = runner.load_passed_folds01_comparison(
        local_tmp_path,
        "current-fingerprint",
    )
    assert loaded["success_checks"]["folds01_pass"] is True

    wrong_fingerprint = copy.deepcopy(valid)
    wrong_fingerprint["cv_fingerprint"] = "old-fingerprint"
    comparison_path.write_text(
        json.dumps(wrong_fingerprint, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="指纹"):
        runner.load_passed_folds01_comparison(
            local_tmp_path,
            "current-fingerprint",
        )

    failed_gate = copy.deepcopy(valid)
    failed_gate["success_checks"]["folds01_pass"] = False
    comparison_path.write_text(
        json.dumps(failed_gate, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未通过|门槛"):
        runner.load_passed_folds01_comparison(
            local_tmp_path,
            "current-fingerprint",
        )
