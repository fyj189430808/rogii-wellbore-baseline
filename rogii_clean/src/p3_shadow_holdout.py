"""P3 标签无关 shadow holdout 的元数据与确定性抽样。"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LEGAL_METADATA_COLUMNS = (
    "well_id",
    "fold",
    "hidden_row_count",
    "md_span",
    "hidden_gr_observed_rate",
    "x_median",
    "y_median",
    "azimuth_deg",
    "typewell_fingerprint",
)
_NUMERIC_METADATA_COLUMNS = (
    "hidden_row_count",
    "md_span",
    "hidden_gr_observed_rate",
    "x_median",
    "y_median",
    "azimuth_deg",
)
_FORBIDDEN_TOKENS = (
    "tvt",
    "target",
    "surface",
    "residual",
    "error",
    "rmse",
    "oracle",
)


def _assert_no_forbidden_columns(columns: object) -> None:
    """阻断标签、地层面和任何误差诊断字段进入选择器。"""

    forbidden = [
        str(column)
        for column in columns
        if any(token in str(column).casefold() for token in _FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden metadata columns: {sorted(forbidden)}")


def _read_csv_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """只读取明确列出的测试期合法原始字段。"""

    try:
        return pd.read_csv(path, usecols=columns)
    except ValueError as error:
        raise ValueError(f"{path.name} missing required legal columns {columns}") from error


def _typewell_fingerprint(typewell_path: Path) -> str:
    """以有序的 Typewell TVT/GR 数值形成跨平台稳定 SHA-256 指纹。"""

    typewell = _read_csv_columns(typewell_path, ["TVT", "GR"])
    values = typewell.loc[:, ["TVT", "GR"]].apply(pd.to_numeric, errors="coerce")
    digest = hashlib.sha256()
    for tvt, gr in values.itertuples(index=False, name=None):
        digest.update(f"{float(tvt):.12g},{float(gr):.12g}\n".encode("ascii"))
    return digest.hexdigest()


def _well_azimuth(horizontal: pd.DataFrame) -> float:
    """使用轨迹首末有限 XY 点计算方位；垂直或退化轨迹记为 0 度。"""

    xy = horizontal.loc[:, ["X", "Y"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return 0.0
    delta_x, delta_y = xy[-1] - xy[0]
    if delta_x == 0.0 and delta_y == 0.0:
        return 0.0
    return float(np.degrees(np.arctan2(delta_y, delta_x)) % 360.0)


def collect_legal_well_metadata(raw_train_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    """从原始井和冻结 fold 表提取严格白名单的井级 shadow 元数据。"""

    if not {"well_id", "fold"}.issubset(registry.columns):
        raise ValueError("registry must contain well_id and fold")
    registry_subset = registry.loc[:, ["well_id", "fold"]].copy()
    registry_subset["well_id"] = registry_subset["well_id"].astype(str)
    if registry_subset["well_id"].duplicated().any():
        raise ValueError("registry contains duplicate well_id")
    if registry_subset["fold"].isna().any():
        raise ValueError("registry contains missing fold")

    records: list[dict[str, object]] = []
    for well_id, fold in registry_subset.sort_values("well_id").itertuples(index=False):
        horizontal_path = Path(raw_train_dir) / f"{well_id}__horizontal_well.csv"
        typewell_path = Path(raw_train_dir) / f"{well_id}__typewell.csv"
        if not horizontal_path.is_file() or not typewell_path.is_file():
            raise FileNotFoundError(f"missing raw files for well {well_id}")

        horizontal = _read_csv_columns(horizontal_path, ["MD", "X", "Y", "GR", "TVT_input"])
        numeric = horizontal.loc[:, ["MD", "X", "Y", "GR"]].apply(
            pd.to_numeric, errors="coerce"
        )
        md = numeric["MD"].to_numpy(dtype=np.float64)
        if not np.isfinite(md).any():
            raise ValueError(f"well {well_id} has no finite MD")
        hidden = horizontal["TVT_input"].isna().to_numpy()
        hidden_md = numeric.loc[hidden, "MD"].dropna().to_numpy(dtype=np.float64)
        if len(hidden_md) == 0:
            raise ValueError(f"well {well_id} has no finite hidden MD")
        hidden_gr = numeric.loc[hidden, "GR"]
        records.append(
            {
                "well_id": str(well_id),
                "fold": int(fold),
                "hidden_row_count": int(hidden.sum()),
                "md_span": float(np.max(hidden_md) - np.min(hidden_md)),
                "hidden_gr_observed_rate": float(hidden_gr.notna().mean()) if len(hidden_gr) else 0.0,
                "x_median": float(np.nanmedian(numeric["X"].to_numpy(dtype=np.float64))),
                "y_median": float(np.nanmedian(numeric["Y"].to_numpy(dtype=np.float64))),
                "azimuth_deg": _well_azimuth(horizontal),
                "typewell_fingerprint": _typewell_fingerprint(typewell_path),
            }
        )

    metadata = pd.DataFrame(records, columns=LEGAL_METADATA_COLUMNS)
    _validate_legal_metadata(metadata, require_all_folds=False)
    return metadata.sort_values("well_id").reset_index(drop=True)


def _validate_legal_metadata(metadata: pd.DataFrame, require_all_folds: bool) -> None:
    """验证标签无关选择器的最小输入合同。"""

    _assert_no_forbidden_columns(metadata.columns)
    missing = set(LEGAL_METADATA_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")
    if metadata["well_id"].astype(str).duplicated().any():
        raise ValueError("metadata contains duplicate well_id")
    folds = pd.to_numeric(metadata["fold"], errors="coerce")
    if folds.isna().any() or not np.equal(folds, np.floor(folds)).all():
        raise ValueError("fold must be an integer")
    if require_all_folds and set(folds.astype(int)) != {0, 1, 2, 3, 4}:
        raise ValueError("metadata must cover frozen folds 0 through 4")
    for column in _NUMERIC_METADATA_COLUMNS:
        values = pd.to_numeric(metadata[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"metadata column {column} must be finite")
    rates = pd.to_numeric(metadata["hidden_gr_observed_rate"], errors="coerce")
    if ((rates < 0.0) | (rates > 1.0)).any():
        raise ValueError("hidden_gr_observed_rate must be in [0, 1]")
    if metadata["typewell_fingerprint"].isna().any():
        raise ValueError("typewell_fingerprint must be populated")


def _stable_hash(salt: str, value: object) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def _quantile_labels(values: pd.Series, well_ids: pd.Series, bins: int = 4) -> pd.Series:
    """基于值和井号的稳定排序打标签，避免输入行顺序影响分箱。"""

    ordered = pd.DataFrame({"value": values.to_numpy(), "well_id": well_ids.astype(str).to_numpy()})
    ordered = ordered.sort_values(["value", "well_id"], kind="mergesort").reset_index()
    group_count = min(int(bins), len(ordered))
    labels = np.floor(np.arange(len(ordered)) * group_count / len(ordered)).astype(int)
    result = pd.Series(index=ordered["index"].to_numpy(), data=labels, dtype="int64")
    return result.reindex(values.index).astype(str)


def _selection_categories(metadata: pd.DataFrame) -> list[str]:
    """生成每口井的边际平衡类别；全部由允许元数据确定。"""

    categories: list[str] = []
    for column in (
        "hidden_row_count",
        "md_span",
        "hidden_gr_observed_rate",
        "x_median",
        "y_median",
        "azimuth_deg",
    ):
        labels = _quantile_labels(metadata[column], metadata["well_id"])
        categories.extend(f"{column}:{label}" for label in labels)
    # 以上 extend 是列主序，改为逐井组装以便贪心成本计算。
    per_row: list[str] = []
    label_columns = {
        column: _quantile_labels(metadata[column], metadata["well_id"])
        for column in (
            "hidden_row_count",
            "md_span",
            "hidden_gr_observed_rate",
            "x_median",
            "y_median",
            "azimuth_deg",
        )
    }
    for index, row in metadata.iterrows():
        row_categories = [f"fold:{int(row['fold'])}"]
        row_categories.extend(
            f"{column}:{label_columns[column].loc[index]}" for column in label_columns
        )
        row_categories.append(f"typewell:{row['typewell_fingerprint']}")
        per_row.append("\x1f".join(row_categories))
    return per_row


def _fold_quotas(metadata: pd.DataFrame, number_of_shadow_wells: int, salt: str) -> dict[int, int]:
    """均匀分配 shadow 数量；余数使用 SHA-256 固定打破平局。"""

    folds = [0, 1, 2, 3, 4]
    base, remainder = divmod(int(number_of_shadow_wells), len(folds))
    quota_order = sorted(folds, key=lambda fold: _stable_hash(salt, f"quota:{fold}"))
    quotas = {fold: base + int(fold in quota_order[:remainder]) for fold in folds}
    availability = metadata.groupby("fold")["well_id"].count().to_dict()
    for fold, quota in quotas.items():
        if int(availability.get(fold, 0)) < quota:
            raise ValueError(f"fold {fold} cannot satisfy shadow quota {quota}")
    return quotas


def select_balanced_shadow(
    metadata: pd.DataFrame,
    number_of_shadow_wells: int,
    salt: str,
) -> pd.DataFrame:
    """以 SHA-256 平局规则和边际平衡贪心法选择固定的 shadow 井。"""

    _validate_legal_metadata(metadata, require_all_folds=True)
    if not isinstance(number_of_shadow_wells, int) or isinstance(number_of_shadow_wells, bool):
        raise ValueError("number_of_shadow_wells must be an integer")
    if not 5 <= number_of_shadow_wells <= len(metadata):
        raise ValueError("number_of_shadow_wells must be between 5 and metadata size")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")

    candidates = metadata.loc[:, LEGAL_METADATA_COLUMNS].copy()
    candidates["fold"] = candidates["fold"].astype(int)
    candidates = candidates.sort_values("well_id").reset_index(drop=True)
    category_strings = _selection_categories(candidates)
    candidate_categories = [value.split("\x1f") for value in category_strings]
    total_category_counts = Counter(category for values in candidate_categories for category in values)
    expected_counts = {
        category: count * float(number_of_shadow_wells) / len(candidates)
        for category, count in total_category_counts.items()
    }
    quotas = _fold_quotas(candidates, number_of_shadow_wells, salt)
    selected_indices: list[int] = []
    selected_counts: Counter[str] = Counter()
    selected_per_fold: Counter[int] = Counter()

    while len(selected_indices) < number_of_shadow_wells:
        best_index: int | None = None
        best_key: tuple[float, str, str] | None = None
        for index, row in candidates.iterrows():
            if index in selected_indices or selected_per_fold[int(row["fold"])] >= quotas[int(row["fold"])]:
                continue
            updated = selected_counts.copy()
            updated.update(candidate_categories[index])
            score = sum(
                ((updated[category] - expected_count) ** 2) / max(expected_count, 1.0)
                for category, expected_count in expected_counts.items()
            )
            key = (float(score), _stable_hash(salt, row["well_id"]), str(row["well_id"]))
            if best_key is None or key < best_key:
                best_index, best_key = int(index), key
        if best_index is None:
            raise RuntimeError("unable to satisfy shadow selection quotas")
        selected_indices.append(best_index)
        selected_counts.update(candidate_categories[best_index])
        selected_per_fold[int(candidates.loc[best_index, "fold"])] += 1

    selected = candidates.loc[selected_indices].copy()
    selected["is_shadow"] = True
    selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    selected["selection_hash"] = selected["well_id"].map(lambda well_id: _stable_hash(salt, well_id))
    return selected.sort_values("well_id").reset_index(drop=True)
