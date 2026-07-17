"""用原五折模型检查固定 seed 重建特征造成的预测漂移；只作敏感性诊断。"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.f05a_direct_physical_candidates import DIRECT_CANDIDATE_COLUMNS  # noqa: E402
from src.lgbm_features import FEATURE_COLUMNS  # noqa: E402


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def _keys_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """比较井号和行号的实际值，不受 int32/int64 存储宽度影响。"""

    return bool(
        np.array_equal(
            left["well_id"].astype(str).to_numpy(),
            right["well_id"].astype(str).to_numpy(),
        )
        and np.array_equal(
            left["row_index"].to_numpy(dtype=np.int64),
            right["row_index"].to_numpy(dtype=np.int64),
        )
    )


def main() -> None:
    smoke_dir = CLEAN_ROOT / "artifacts" / "F05a_deterministic_feature_reproduction_v1"
    model_dir = CLEAN_ROOT / "artifacts" / "F05a_direct_physical_candidates_v2"
    b00_dir = CLEAN_ROOT / "artifacts" / "B00_simple_lgbm_v1"
    regenerated = pd.read_parquet(smoke_dir / "regenerated_features.parquet")
    regenerated["well_id"] = regenerated["well_id"].astype(str)
    wells = regenerated["well_id"].drop_duplicates().tolist()

    base = pd.read_parquet(b00_dir / "feature_cache.parquet")
    base["well_id"] = base["well_id"].astype(str)
    base = base.loc[base["well_id"].isin(wells)].reset_index(drop=True)
    regenerated = regenerated.reset_index(drop=True)
    if not _keys_equal(base, regenerated):
        raise ValueError("B00 与三井重建特征的键顺序不一致")

    frozen_predictions = pd.read_parquet(model_dir / "predictions.parquet")
    frozen_predictions["well_id"] = frozen_predictions["well_id"].astype(str)
    frozen_predictions = frozen_predictions.loc[
        frozen_predictions["well_id"].isin(wells)
    ].reset_index(drop=True)
    b00_predictions = pd.read_parquet(b00_dir / "predictions.parquet")
    b00_predictions["well_id"] = b00_predictions["well_id"].astype(str)
    b00_predictions = b00_predictions.loc[
        b00_predictions["well_id"].isin(wells)
    ].reset_index(drop=True)
    if not _keys_equal(base, frozen_predictions):
        raise ValueError("冻结 F05a 预测与重建特征的键顺序不一致")

    feature_table = pd.concat(
        [
            base.reset_index(drop=True),
            regenerated[DIRECT_CANDIDATE_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )
    rows: list[dict] = []
    for well_id in wells:
        mask = feature_table["well_id"] == well_id
        well_features = feature_table.loc[mask]
        fold_id = int(well_features["fold"].iloc[0])
        model = lgb.Booster(model_file=str(model_dir / f"fold_{fold_id}" / "model.txt"))
        fixed_seed_delta = model.predict(
            well_features[[*FEATURE_COLUMNS, *DIRECT_CANDIDATE_COLUMNS]],
            num_iteration=model.current_iteration(),
        )
        fixed_seed_tvt = (
            well_features["last_visible_tvt"].to_numpy(dtype=np.float64)
            + fixed_seed_delta
        )
        frozen_well = frozen_predictions.loc[
            frozen_predictions["well_id"] == well_id
        ].reset_index(drop=True)
        b00_well = b00_predictions.loc[
            b00_predictions["well_id"] == well_id
        ].reset_index(drop=True)
        target = frozen_well["target_tvt"].to_numpy(dtype=np.float64)
        frozen_tvt = frozen_well["pred_tvt"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "well_id": well_id,
                "fold": fold_id,
                "rows": int(len(target)),
                "frozen_feature_model_rmse": rmse(target, frozen_tvt),
                "fixed_seed_feature_same_model_rmse": rmse(target, fixed_seed_tvt),
                "b00_rmse": rmse(
                    target,
                    b00_well["pred_tvt"].to_numpy(dtype=np.float64),
                ),
                "carry_rmse": rmse(
                    target,
                    frozen_well["carry_tvt"].to_numpy(dtype=np.float64),
                ),
                "fixed_minus_frozen_prediction_mae": float(
                    np.mean(np.abs(fixed_seed_tvt - frozen_tvt))
                ),
                "fixed_minus_frozen_prediction_max_abs": float(
                    np.max(np.abs(fixed_seed_tvt - frozen_tvt))
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(smoke_dir / "prediction_sensitivity.csv", index=False)
    print(result.to_string(index=False))
    print("注意：这里复用了旧模型，没有用固定seed特征重新训练，只是敏感性诊断。")


if __name__ == "__main__":
    main()
