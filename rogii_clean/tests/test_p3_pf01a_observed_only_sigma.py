"""P3-PF01a 的最小硬测试：只改 sigma、影子隔离、缓存可续跑。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from scripts import run_p3_pf01a_observed_only_sigma as runner  # noqa: E402
from scripts.diagnose_p3_d01_pf_observation_weights import load_development_registry  # noqa: E402
from src.p2_p01_multiseed_pf import FEATURE_COLUMNS, prepare_particle_filter_inputs  # noqa: E402
from src.p3_pf01a_observed_only_sigma import observed_only_gr_sigma  # noqa: E402


PARAMETERS = {
    "typewell_grid_step_ft": 0.2,
    "gr_sigma_min_api": 10.0,
    "gr_sigma_max_api": 60.0,
    "initial_rate_visible_tail_rows": 30,
    "number_of_particles": 2,
    "number_of_seeds": 2,
    "seed_base": 0,
    "rate_momentum": 0.998,
    "rate_noise": 0.002,
    "position_noise_ft": 0.005,
    "resample_position_noise_ft": 0.1,
    "resample_rate_noise": 0.001,
    "resample_effective_fraction": 0.5,
    "initial_position_spread_ft": 4.5,
    "position_limit_beyond_typewell_ft": 100.0,
    "initial_rate_std": 0.01,
    "minimum_md_step_ft": 1.0,
    "squared_gr_residual_cap": 600.0,
    "likelihood_floor": 1e-300,
    "likelihood_scales": [3.0, 5.0, 8.0, 12.0],
}


def make_horizontal(gr: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD": [0.0, 1.0, 2.0, 3.0],
            "Z": [100.0, 100.0, 100.0, 100.0],
            "GR": gr,
            "TVT_input": [0.0, 1.0, 2.0, np.nan],
            "TVT": [0.0, 1.0, 2.0, 9999.0],
        }
    )


def make_typewell() -> pd.DataFrame:
    return pd.DataFrame({"TVT": [0.0, 1.0, 2.0, 3.0], "GR": [10.0, 20.0, 30.0, 40.0]})


def test_no_missing_visible_gr_matches_legacy_sigma() -> None:
    horizontal = make_horizontal([20.0, 40.0, 70.0, 40.0])
    legacy = prepare_particle_filter_inputs(horizontal, make_typewell(), PARAMETERS)["gr_sigma"]
    assert observed_only_gr_sigma(horizontal, make_typewell(), PARAMETERS) == legacy


def test_missing_visible_gr_uses_only_observed_residuals() -> None:
    horizontal = make_horizontal([10.0, np.nan, 70.0, 40.0])
    # observed residual 是 [0, 40]，总体标准差正好是 20。
    assert observed_only_gr_sigma(horizontal, make_typewell(), PARAMETERS) == 20.0


def test_hidden_tvt_mutation_does_not_change_sigma() -> None:
    horizontal = make_horizontal([10.0, np.nan, 70.0, 40.0])
    before = observed_only_gr_sigma(horizontal, make_typewell(), PARAMETERS)
    horizontal.loc[horizontal["TVT_input"].isna(), "TVT"] = -1e12
    after = observed_only_gr_sigma(horizontal, make_typewell(), PARAMETERS)
    assert after == before


def test_shadow_well_is_removed(tmp_path: Path) -> None:
    fold_path = tmp_path / "fold.csv"
    shadow_path = tmp_path / "shadow.csv"
    pd.DataFrame(
        {"well_id": ["a", "b"], "fold": [0, 1], "hidden_rows": [1, 1]}
    ).to_csv(fold_path, index=False)
    pd.DataFrame({"well_id": ["b"], "is_shadow": [True]}).to_csv(shadow_path, index=False)
    development = load_development_registry(fold_path, shadow_path)
    assert development["well_id"].tolist() == ["a"]


def test_smoke_cache_resume(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    horizontal = make_horizontal([10.0, np.nan, 70.0, 40.0])
    horizontal[["MD", "Z", "GR", "TVT_input"]].to_csv(
        raw_dir / "a__horizontal_well.csv", index=False
    )
    make_typewell().to_csv(raw_dir / "a__typewell.csv", index=False)

    fake_features = pd.DataFrame(
        {column: ([3] if column == "row_index" else [1.0]) for column in FEATURE_COLUMNS}
    )
    fake_quality = {
        "legacy_gr_sigma": 25.0,
        "observed_only_gr_sigma": 20.0,
        "visible_observed_gr_rows": 2,
        "pf_best_ll_per_row": -1.0,
        "pf_ll_spread": 0.5,
    }
    monkeypatch.setattr(
        runner,
        "build_observed_only_sigma_pf_features",
        lambda *_args, **_kwargs: (fake_features.copy(), dict(fake_quality)),
    )
    task = {
        "well_id": "a",
        "fold": 0,
        "hidden_rows": 1,
        "raw_train_dir": str(raw_dir),
        "artifact_dir": str(tmp_path / "artifact"),
        "fingerprint": "fixed-test-fingerprint",
        "particle_filter": PARAMETERS,
    }
    first = runner.generate_one_well(task)
    assert first["cache_hit"] is False
    monkeypatch.setattr(
        runner,
        "build_observed_only_sigma_pf_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应重算")),
    )
    second = runner.generate_one_well(task)
    assert second["cache_hit"] is True

