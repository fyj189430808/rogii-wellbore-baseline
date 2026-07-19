from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p2_p03_pf_seed_dispersion_cv as p03  # noqa: E402
from scripts.run_simple_lgbm_cv import read_json  # noqa: E402


def _legal_cache(well_id: str = "well_a") -> pd.DataFrame:
    """构造两行完整 P01 合法缓存，供 P03 合并边界测试。"""

    return pd.DataFrame(
        {
            "well_id": [well_id, well_id],
            "row_index": [10, 11],
            "last_visible_tvt": np.array([100.0, 100.0], dtype=np.float32),
            "pf128_mean_tvt": np.array([101.0, 102.0], dtype=np.float32),
            "pf128_mean_delta": np.array([1.0, 2.0], dtype=np.float32),
            "pf128_seed0_delta": np.array([1.2, 2.2], dtype=np.float32),
            "pf128_scale_3_delta": np.array([0.8, 1.8], dtype=np.float32),
            "pf128_scale_5_delta": np.array([0.9, 1.9], dtype=np.float32),
            "pf128_scale_8_delta": np.array([1.1, 2.1], dtype=np.float32),
            "pf128_scale_12_delta": np.array([1.3, 2.3], dtype=np.float32),
            "pf128_seed_std": np.array([0.2, 0.4], dtype=np.float32),
            "_cache_fingerprint": ["fingerprint", "fingerprint"],
        }
    )


def _base_table(well_id: str = "well_a") -> pd.DataFrame:
    """构造只含 P03 合并所需锚点的自然隐藏行表。"""

    return pd.DataFrame(
        {
            "well_id": [well_id, well_id],
            "row_index": [10, 11],
            "last_visible_tvt": np.array([100.0, 100.0], dtype=np.float32),
        }
    )


def _registry(well_id: str = "well_a") -> pd.DataFrame:
    """构造一口井的最小按井折注册表。"""

    return pd.DataFrame({"well_id": [well_id], "fold": [0], "hidden_rows": [2]})


def _contract_inputs() -> tuple[dict, dict, dict]:
    """读取真实冻结配置、模型参数和 P02 41 列清单。"""

    config = read_json(CLEAN_ROOT / "configs" / "p2_p03_pf_seed_dispersion_v1.json")
    model_config = read_json(CLEAN_ROOT / str(config["model_config"]))
    baseline_features = read_json(CLEAN_ROOT / str(config["baseline_feature_list"]))
    return config, model_config, baseline_features


def test_p03_contract_is_p02_41_plus_seed_std() -> None:
    """正式特征必须严格为 P02 41 列后追加 seed 标准差。"""

    config, model_config, baseline_features = _contract_inputs()
    actual = p03.validate_frozen_contract(config, model_config, baseline_features)

    assert len(actual) == 42
    assert actual[:-1] == baseline_features["features"]
    assert actual[-1] == "pf128_seed_std"
    assert p03.FROZEN_MODEL_PARAMS == model_config["params"]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("baseline_feature_count", 40),
        ("formal_feature_count", 41),
        ("new_feature_names", ["pf128_seed0_delta"]),
        ("fold_version", "spatial_pad_1000_v1"),
    ],
)
def test_p03_contract_rejects_changed_top_level(field: str, bad_value: object) -> None:
    """配置不能同时篡改基线、列数、新列或折协议。"""

    config, model_config, baseline_features = _contract_inputs()
    changed = copy.deepcopy(config)
    changed[field] = bad_value

    with pytest.raises(ValueError):
        p03.validate_frozen_contract(changed, model_config, baseline_features)


def test_p03_allows_seed_std_but_rejects_diagnostics_and_labels() -> None:
    """seed 标准差是正式列，seed0、标签、surface 和 oracle 不是。"""

    assert not p03._contains_forbidden_feature("pf128_seed_std")
    for forbidden in (
        "pf128_seed0_delta",
        "target_tvt",
        "TVT",
        "EGFDU",
        "surface_height",
        "oracle_rank",
        "geology_zone",
    ):
        assert p03._contains_forbidden_feature(forbidden)


def test_merge_cache_adds_six_pf_columns_once(tmp_path: Path) -> None:
    """单井缓存应一次性加入均值、四尺度和 seed 标准差。"""

    cache = _legal_cache()
    cache.to_parquet(tmp_path / "well_a.parquet", index=False)

    merged = p03.merge_p01_path_and_dispersion_cache(
        _base_table(),
        _registry(),
        tmp_path,
        "fingerprint",
    )

    for name in p03.P01_CACHE_FORMAL_FEATURES:
        assert name in merged.columns
    assert "pf128_seed0_delta" not in merged.columns
    assert len(merged) == 2
    assert merged["pf128_seed_std"].tolist() == pytest.approx([0.2, 0.4])


@pytest.mark.parametrize("mutation", ["missing_std", "nan_std", "wrong_anchor", "duplicate_key"])
def test_merge_cache_rejects_invalid_cache(tmp_path: Path, mutation: str) -> None:
    """缺列、非有限值、错误锚点和重复行键都必须在训练前失败。"""

    cache = _legal_cache()
    base = _base_table()
    if mutation == "missing_std":
        cache = cache.drop(columns=["pf128_seed_std"])
    elif mutation == "nan_std":
        cache.loc[0, "pf128_seed_std"] = np.nan
    elif mutation == "wrong_anchor":
        base.loc[0, "last_visible_tvt"] = 999.0
    elif mutation == "duplicate_key":
        cache.loc[1, "row_index"] = cache.loc[0, "row_index"]
    cache.to_parquet(tmp_path / "well_a.parquet", index=False)

    with pytest.raises((ValueError, KeyError)):
        p03.merge_p01_path_and_dispersion_cache(
            base,
            _registry(),
            tmp_path,
            "fingerprint",
        )


def test_merge_cache_rejects_wrong_fingerprint(tmp_path: Path) -> None:
    """缓存必须来自 P01 已冻结生成器指纹。"""

    cache = _legal_cache()
    cache["_cache_fingerprint"] = "wrong"
    cache.to_parquet(tmp_path / "well_a.parquet", index=False)

    with pytest.raises(ValueError, match="指纹"):
        p03.merge_p01_path_and_dispersion_cache(
            _base_table(),
            _registry(),
            tmp_path,
            "fingerprint",
        )


def test_reverse_control_only_reverses_seed_std_within_well() -> None:
    """负对照只能破坏 std 的行位置，不能改其他 41 列或跨井混值。"""

    table = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "row_index": [0, 1, 0, 1],
            "pf128_seed_std": [1.0, 2.0, 10.0, 20.0],
            "pf128_mean_delta": [3.0, 4.0, 5.0, 6.0],
        }
    )

    reversed_table = p03.reverse_seed_std_within_well(table)

    assert reversed_table["pf128_seed_std"].tolist() == [2.0, 1.0, 20.0, 10.0]
    assert reversed_table["pf128_mean_delta"].tolist() == table["pf128_mean_delta"].tolist()
    assert table["pf128_seed_std"].tolist() == [1.0, 2.0, 10.0, 20.0]


def test_full5_requires_strictly_negative_bootstrap_upper_bound() -> None:
    """根 AGENTS 要求 CI 上界小于 0，恰好等于 0 不能晋级。"""

    config, _, _ = _contract_inputs()
    comparison = {
        "overall": {
            "baseline_micro_rmse": 10.5,
            "micro_rmse": 10.0,
            "p90_well_rmse": 12.0,
            "baseline_p90_well_rmse": 12.0,
            "well_win_rate": 0.8,
        },
        "folds": [
            {"fold": fold, "micro_rmse": 10.0, "baseline_micro_rmse": 10.5}
            for fold in range(5)
        ],
        "paired_well_bootstrap": {"ci95_high": 0.0},
    }

    checks = p03.evaluate_model_success_checks(comparison, config, "full5")

    assert checks["comparison_baseline"] == "P2_P02_multiscale_pf_paths_v1"
    assert checks["full5_numeric_pass"] is False


def test_oracle_audit_is_diagnostic_summary_not_row_features() -> None:
    """oracle 只能返回独立 JSON 摘要，不能产生可合并的行级正式特征。"""

    table = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "pf128_seed_std": [0.1, 0.2, 1.0, 2.0],
            "last_visible_tvt": [100.0, 100.0, 200.0, 200.0],
            "pf128_mean_delta": [1.0, 2.0, 1.0, 2.0],
            "target_tvt": [101.0, 103.0, 202.0, 203.0],
        }
    )

    audit = p03.build_seed_dispersion_oracle_audit(table)

    assert audit["diagnostic_only"] is True
    assert audit["used_for_feature_or_model_selection"] is False
    assert "formal_features" not in audit
    assert not isinstance(audit.get("quantile_bins"), pd.DataFrame)
