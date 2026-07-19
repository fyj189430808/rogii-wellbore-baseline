"""冻结 P3B00 对 P2-P02 的只读血缘，不复制 OOF 预测。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "P3B00_group5_p2p02_v1"
SOURCE_EXPERIMENT_ID = "P2_P02_multiscale_pf_paths_v1"
FOLD_VERSION = "balanced_well_5fold_v1"
EXPECTED_MICRO_RMSE = 10.305704992073148
EXPECTED_FOLD_RMSE = [
    10.170796687354473,
    9.4624641719035,
    9.181973787681697,
    10.8754729397572,
    11.639196676252652,
]
SOURCE_FILE_NAMES = ("config.json", "metrics.json", "feature_list.json", "predictions.parquet")


def _sha256(path: Path) -> str:
    """分块计算大文件哈希，预测文件始终不加载进内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while block := input_file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """读取并限制冻结输入为 JSON 对象。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"冻结来源必须是 JSON 对象：{path}")
    return value


def _relative_path(clean_root: Path, path: Path) -> str:
    """禁止血缘指向 clean_root 以外的不可控文件。"""

    try:
        return path.relative_to(clean_root).as_posix()
    except ValueError as error:
        raise ValueError(f"冻结来源不在 clean_root 内：{path}") from error


def _validate_sources(source_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    """严格核对 P2-P02 的身份、41 特征和五折固定指标。"""

    source_paths = {name.removesuffix(".json").removesuffix(".parquet"): source_dir / name for name in SOURCE_FILE_NAMES}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P3B00 缺少冻结来源：{missing}")

    source_config = _read_json(source_paths["config"])
    if source_config.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("P3B00 来源不是 P2_P02_multiscale_pf_paths_v1")
    if source_config.get("fold_version") != FOLD_VERSION:
        raise ValueError("P3B00 来源 fold_version 不等于 balanced_well_5fold_v1")

    metrics = _read_json(source_paths["metrics"])
    folds = metrics.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("P3B00 必须具有恰好五个 P2-P02 折指标")
    observed_fold_rmse = [fold.get("micro_rmse") if isinstance(fold, dict) else None for fold in folds]
    observed_fold_ids = [fold.get("fold") if isinstance(fold, dict) else None for fold in folds]
    if observed_fold_ids != list(range(5)) or observed_fold_rmse != EXPECTED_FOLD_RMSE:
        raise ValueError("P3B00 五折 micro RMSE 与冻结 P2-P02 不一致")
    overall = metrics.get("overall")
    if not isinstance(overall, dict) or overall.get("micro_rmse") != EXPECTED_MICRO_RMSE:
        raise ValueError("P3B00 pooled micro RMSE 与冻结 P2-P02 不一致")

    feature_list = _read_json(source_paths["feature_list"])
    features = feature_list.get("features")
    if feature_list.get("feature_count") != 41 or not isinstance(features, list) or len(features) != 41:
        raise ValueError("P3B00 来源必须是 41 个正式特征")
    if len(set(map(str, features))) != 41:
        raise ValueError("P3B00 来源特征不可重复")
    return source_config, metrics, feature_list, source_paths


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_p3_baseline(clean_root: Path, output_dir: Path) -> dict[str, object]:
    """写入 P3B00 指针血缘；不读取、复制或重编码 predictions.parquet。"""

    root = Path(clean_root).resolve()
    source_dir = root / "artifacts" / SOURCE_EXPERIMENT_ID
    source_config, metrics, feature_list, source_paths = _validate_sources(source_dir)
    destination = Path(output_dir).resolve()
    if destination == source_dir.resolve():
        raise ValueError("P3B00 输出目录不能覆盖 P2-P02 来源目录")
    destination.mkdir(parents=True, exist_ok=True)

    fold_rmse = list(EXPECTED_FOLD_RMSE)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_experiment_id": SOURCE_EXPERIMENT_ID,
        "fold_version": FOLD_VERSION,
        "validation_unit": "complete_well",
        "expected_feature_count": 41,
        "expected_micro_rmse": EXPECTED_MICRO_RMSE,
        "expected_fold_rmse": fold_rmse,
        "prediction_policy": "pointer_and_sha256_only_no_prediction_copy",
    }
    lineage = {
        "experiment_id": EXPERIMENT_ID,
        "baseline_experiment_id": SOURCE_EXPERIMENT_ID,
        "prediction_policy": config["prediction_policy"],
        "sources": {
            name: {"path": _relative_path(root, path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "source_config_experiment_id": source_config["experiment_id"],
    }
    _write_json(destination / "config.json", config)
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "feature_list.json", feature_list)
    _write_json(destination / "lineage.json", lineage)
    (destination / "conclusion.md").write_text(
        "# P3B00 冻结结论\n\n"
        f"事实：P2-P02 的 41 个特征和 pooled micro RMSE {EXPECTED_MICRO_RMSE} 已冻结。\n\n"
        "推断：后续 P3 实验可把该产物作为唯一固定基线。\n\n"
        "仍未验证：P3 的新诊断或特征是否改善该基线。\n\n"
        "当前只能否定：未复制 predictions.parquet，不能把此冻结视为新的 OOF 计算。\n\n"
        "下一步：在独立 P3 产物中创建标签隔离的 shadow holdout。\n",
        encoding="utf-8",
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "micro_rmse": EXPECTED_MICRO_RMSE,
        "fold_rmse": fold_rmse,
        "feature_count": 41,
        "output_dir": str(destination),
    }
