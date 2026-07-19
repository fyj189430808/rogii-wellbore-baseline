"""P3B00 冻结必须只记录 P2-P02 的不可变血缘。"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def workspace_tmp_path() -> Iterator[Path]:
    """Windows 默认临时目录可能不可写，测试临时目录固定在工作区旁。"""

    with tempfile.TemporaryDirectory(prefix=".pytest_freeze_p3_", dir=CLEAN_ROOT.parent) as directory:
        yield Path(directory)


def test_freeze_records_p2_p02_lineage_without_copying_predictions(
    workspace_tmp_path: Path,
) -> None:
    """冻结产物保存四个来源的相对路径和哈希，预测文件仍仅在来源目录。"""

    from scripts.freeze_p3_b00 import freeze_p3_baseline

    source_dir = workspace_tmp_path / "artifacts" / "P2_P02_multiscale_pf_paths_v1"
    source_dir.mkdir(parents=True)
    config_path = source_dir / "config.json"
    metrics_path = source_dir / "metrics.json"
    feature_list_path = source_dir / "feature_list.json"
    predictions_path = source_dir / "predictions.parquet"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "P2_P02_multiscale_pf_paths_v1",
                "fold_version": "balanced_well_5fold_v1",
            }
        ),
        encoding="utf-8",
    )
    fold_rmse = [10.170796687354473, 9.4624641719035, 9.181973787681697,
                 10.8754729397572, 11.639196676252652]
    metrics_path.write_text(
        json.dumps(
            {
                "folds": [
                    {"fold": fold, "micro_rmse": rmse}
                    for fold, rmse in enumerate(fold_rmse)
                ],
                "overall": {"micro_rmse": 10.305704992073148},
            }
        ),
        encoding="utf-8",
    )
    feature_list_path.write_text(
        json.dumps({"feature_count": 41, "features": [f"f{i}" for i in range(41)]}),
        encoding="utf-8",
    )
    predictions_path.write_bytes(b"tiny-oof-predictions")

    output_dir = workspace_tmp_path / "artifacts" / "P3B00_group5_p2p02_v1"
    result = freeze_p3_baseline(workspace_tmp_path, output_dir)

    assert result["experiment_id"] == "P3B00_group5_p2p02_v1"
    assert result["micro_rmse"] == 10.305704992073148
    assert result["fold_rmse"] == fold_rmse
    assert result["feature_count"] == 41
    lineage = json.loads((output_dir / "lineage.json").read_text(encoding="utf-8"))
    for source_name, source_path in {
        "config": config_path,
        "metrics": metrics_path,
        "feature_list": feature_list_path,
        "predictions": predictions_path,
    }.items():
        assert lineage["sources"][source_name]["path"] == str(
            source_path.relative_to(workspace_tmp_path).as_posix()
        )
        assert lineage["sources"][source_name]["sha256"] == _sha256(source_path)
    assert not (output_dir / "predictions.parquet").exists()
    assert json.loads((output_dir / "config.json").read_text(encoding="utf-8"))["baseline_experiment_id"] == "P2_P02_multiscale_pf_paths_v1"
    assert json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))["overall"]["micro_rmse"] == 10.305704992073148
    assert len(json.loads((output_dir / "feature_list.json").read_text(encoding="utf-8"))["features"]) == 41
    assert "事实：" in (output_dir / "conclusion.md").read_text(encoding="utf-8")
