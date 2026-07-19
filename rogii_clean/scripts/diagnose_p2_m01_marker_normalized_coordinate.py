"""运行 P2-M01 外折 marker 转移与候选排名诊断。

脚本故意分成两个不能颠倒的阶段：

1. 合法阶段只读取所有井 Typewell 的 ``TVT/GR``，并且只读取 outer-train
   （fold 1～4）Typewell 的 ``Geology``。随后构造并落盘 fold 0 的 marker。
2. 只有合法 marker 已经通过校验并保存后，才读取 fold 0 自身 ``Geology``、
   冻结隐藏真值和 F01b 得分，用于明确标注的 oracle 排名诊断。

本实验不训练模型、不生成预测路径，也不会把 oracle 结果写回合法 marker 表。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# clean 项目根目录；所有配置中的相对路径都以这里为基准。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CLEAN_ROOT.parent
sys.path.insert(0, str(CLEAN_ROOT))

from scripts.audit_surface_lineage import extract_typewell_markers  # noqa: E402
from scripts.diagnose_p2_s01_outer_fold_surface import (  # noqa: E402
    file_sha256,
    stable_frame_hash,
    stable_json_hash,
    write_json_atomic,
)
from src.p2_f01_continuous_gr_path import build_emission_costs  # noqa: E402
from src.p2_m01_marker_normalized_coordinate import (  # noqa: E402
    aggregate_outer_train_markers,
    deterministic_contiguous_gate,
    marker_coordinate_to_tvt,
    normalized_gated_rank,
    tvt_to_marker_coordinate,
    typewell_template_fingerprint,
)


EXPERIMENT_ID = "P2_M01_marker_normalized_coordinate_v1"
DEFAULT_CONFIG_PATH = (
    CLEAN_ROOT / "configs" / "p2_m01_marker_normalized_coordinate_v1.json"
)
DEFAULT_ARTIFACT_DIR = CLEAN_ROOT / "artifacts" / EXPERIMENT_ID
SUPPORTED_MODES = ("smoke", "all")
FROZEN_MARKERS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
FROZEN_OFFSET_GRID = np.arange(-40.0, 40.0 + 0.1, 2.0, dtype=np.float64)
EXPECTED_ALL_WELLS = 773
EXPECTED_ALL_HIDDEN_ROWS = 3_783_989
F01B_MINIMUM_VALID_SCALES = 3


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """先写临时 CSV 再替换，避免中断留下半个结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """先写临时 parquet 再替换，保证排名表原子落盘。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    """拒绝结果出现后改变实验卡中冻结的 marker 或匹配规则。"""

    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("P2-M01 experiment_id 不匹配")
    if int(config.get("fold_id", -1)) != 0:
        raise ValueError("P2-M01 v1 只能诊断 fold 0")
    if list(config.get("marker_names", [])) != FROZEN_MARKERS:
        raise ValueError("P2-M01 marker_names 被修改")
    if int(config.get("template_tail_rows", -1)) != 500:
        raise ValueError("P2-M01 template_tail_rows 必须固定为 500")
    if config.get("template_matching") != (
        "exact_outer_train_same_fingerprint_different_well_id"
    ):
        raise ValueError("P2-M01 template_matching 被修改")
    if config.get("marker_aggregation") != "outer_train_donor_median":
        raise ValueError("P2-M01 marker_aggregation 被修改")
    if float(config.get("maximum_donor_marker_range_ft", -1.0)) != 1.0:
        raise ValueError("P2-M01 donor marker 极差必须固定为 1 ft")
    if int(config.get("random_gate_seed", -1)) != 29:
        raise ValueError("P2-M01 random_gate_seed 必须固定为 29")
    if int(config.get("cluster_bootstrap_repetitions", -1)) != 2000:
        raise ValueError("P2-M01 bootstrap 次数必须固定为 2000")

    configured_offsets = np.asarray(config.get("offset_grid_ft", []), dtype=np.float64)
    if not np.array_equal(configured_offsets, FROZEN_OFFSET_GRID):
        raise ValueError("P2-M01 offset_grid_ft 被修改")


def validate_external_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """核对 fold、F01b 和 P2B00 均是实验卡登记的冻结文件。"""

    checked_files = {
        "fold_registry": (
            CLEAN_ROOT / str(config["fold_registry"]),
            str(config["fold_registry_sha256"]),
        ),
        "f01b_summary_fold0": (
            CLEAN_ROOT / str(config["f01b_artifact_dir"]) / "summary_fold0.json",
            str(config["f01b_summary_fold0_sha256"]),
        ),
        "f01b_per_well_fold0": (
            CLEAN_ROOT / str(config["f01b_artifact_dir"]) / "per_well_fold0.csv",
            str(config["f01b_per_well_fold0_sha256"]),
        ),
        "baseline_predictions": (
            CLEAN_ROOT / str(config["baseline_predictions"]),
            str(config["baseline_predictions_sha256"]),
        ),
    }

    observed_hashes: dict[str, str] = {}
    for input_name, (input_path, expected_hash) in checked_files.items():
        if not input_path.is_file():
            raise FileNotFoundError(f"P2-M01 找不到冻结输入：{input_path}")
        observed_hash = file_sha256(input_path)
        if observed_hash != expected_hash:
            raise ValueError(f"P2-M01 {input_name} SHA-256 不一致")
        observed_hashes[input_name] = observed_hash

    summary_path = checked_files["f01b_summary_fold0"][0]
    f01b_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if f01b_summary.get("experiment_fingerprint") != config.get(
        "f01b_experiment_fingerprint"
    ):
        raise ValueError("P2-M01 F01b 实验指纹不一致")
    if int(f01b_summary.get("wells", -1)) != int(config["expected_fold_wells"]):
        raise ValueError("P2-M01 F01b fold 0 井数不一致")
    if int(f01b_summary.get("hidden_rows", -1)) != int(
        config["expected_hidden_rows"]
    ):
        raise ValueError("P2-M01 F01b fold 0 隐藏行数不一致")
    return observed_hashes


def load_registry(config: dict[str, Any]) -> pd.DataFrame:
    """读取固定按井五折，并核对全体井数和隐藏行数。"""

    registry_path = CLEAN_ROOT / str(config["fold_registry"])
    registry = pd.read_csv(registry_path, dtype={"well_id": str, "pad_id": str})
    required_columns = {"well_id", "fold", "hidden_rows"}
    missing_columns = required_columns.difference(registry.columns)
    if missing_columns:
        raise ValueError(f"P2-M01 fold 表缺列：{sorted(missing_columns)}")
    registry["well_id"] = registry["well_id"].astype(str)
    registry["fold"] = pd.to_numeric(registry["fold"], errors="raise").astype(int)
    registry["hidden_rows"] = pd.to_numeric(
        registry["hidden_rows"], errors="raise"
    ).astype(int)
    if bool(registry["well_id"].duplicated().any()):
        raise ValueError("P2-M01 fold 表含重复 well_id")
    if sorted(registry["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("P2-M01 fold 表必须包含 0～4")
    if len(registry) != EXPECTED_ALL_WELLS:
        raise ValueError("P2-M01 fold 表全体井数不等于 773")
    if int(registry["hidden_rows"].sum()) != EXPECTED_ALL_HIDDEN_ROWS:
        raise ValueError("P2-M01 fold 表全体隐藏行数不一致")
    return registry.reset_index(drop=True)


def select_validation_registry(
    registry: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """只保留预注册 fold 0，并保持注册表原有顺序。"""

    if not {"well_id", "fold"}.issubset(registry.columns):
        raise ValueError("registry 必须含 well_id 和 fold")
    selected = registry.loc[registry["fold"].eq(int(config["fold_id"]))].copy()
    selected["well_id"] = selected["well_id"].astype(str)
    return selected.reset_index(drop=True)


def validate_legal_marker_table(
    marker_table: pd.DataFrame,
    expected_fold: int,
) -> None:
    """确保合法 marker 表没有混入任何 oracle、真值或 Geology 字段。"""

    forbidden_fragments = (
        "true",
        "oracle",
        "target",
        "hidden",
        "geology",
        "withheld",
        "surface",
    )
    forbidden_columns = [
        column
        for column in marker_table.columns
        if any(fragment in column.lower() for fragment in forbidden_fragments)
    ]
    if forbidden_columns:
        raise ValueError(f"P2-M01 legal marker 出现禁止列：{forbidden_columns}")

    required_columns = {
        "well_id",
        "fold",
        "template_fingerprint",
        "marker_name",
        "marker_tvt",
        "marker_usable",
        "marker_ambiguous",
        "donor_well_count",
        "donor_value_count",
        "donor_marker_range_ft",
    }
    missing_columns = required_columns.difference(marker_table.columns)
    if missing_columns:
        raise ValueError(f"P2-M01 legal marker 缺列：{sorted(missing_columns)}")
    if not marker_table["fold"].astype(int).eq(int(expected_fold)).all():
        raise ValueError("P2-M01 legal marker fold 不等于预注册 fold")
    if bool(marker_table.duplicated(["well_id", "marker_name"]).any()):
        raise ValueError("P2-M01 legal marker 含重复井-marker 键")
    if marker_table["template_fingerprint"].astype(str).str.len().eq(0).any():
        raise ValueError("P2-M01 legal marker 模板指纹为空")


def experiment_fingerprint(config: dict[str, Any]) -> str:
    """把配置、runner、核心函数和三个冻结输入放入单一指纹。"""

    parts = {
        "config": config,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "core_sha256": file_sha256(
            CLEAN_ROOT / "src" / "p2_m01_marker_normalized_coordinate.py"
        ),
        "f01_emission_core_sha256": file_sha256(
            CLEAN_ROOT / "src" / "p2_f01_continuous_gr_path.py"
        ),
        "marker_extractor_sha256": file_sha256(
            CLEAN_ROOT / "scripts" / "audit_surface_lineage.py"
        ),
        "fold_registry_sha256": config["fold_registry_sha256"],
        "f01b_summary_fold0_sha256": config["f01b_summary_fold0_sha256"],
        "baseline_predictions_sha256": config["baseline_predictions_sha256"],
    }
    return stable_json_hash(parts)


def run_program_controls(config: dict[str, Any]) -> dict[str, Any]:
    """在真实 fold 前验证裁剪匹配、自身排除和 q 往返公式。"""

    number_of_rows = 1200
    tvt = 10_000.0 + 0.5 * np.arange(number_of_rows, dtype=np.float64)
    gr = 70.0 + 8.0 * np.sin(np.arange(number_of_rows) / 17.0)

    # 六个标签从不同的已观察边界开始；第一行保持空标签，防止左截断。
    geology = np.full(number_of_rows, "", dtype=object)
    marker_starts = [100, 300, 360, 520, 590, 850]
    for marker_position, marker_name in zip(marker_starts, FROZEN_MARKERS):
        geology[marker_position] = marker_name
    synthetic_typewell = pd.DataFrame({"TVT": tvt, "GR": gr, "Geology": geology})

    # 裁掉浅部 50 行不改变最后 500 行，因此两个模板必须同指纹。
    cropped_without_geology = synthetic_typewell.iloc[50:][["TVT", "GR"]].copy()
    full_fingerprint = typewell_template_fingerprint(
        synthetic_typewell[["TVT", "GR"]],
        tail_rows=int(config["template_tail_rows"]),
    )
    cropped_fingerprint = typewell_template_fingerprint(
        cropped_without_geology,
        tail_rows=int(config["template_tail_rows"]),
    )

    extracted = extract_typewell_markers(synthetic_typewell)
    donor_rows: list[dict[str, Any]] = []
    expected_marker_values: dict[str, float] = {}
    for marker_name in FROZEN_MARKERS:
        marker_info = extracted[marker_name]
        marker_tvt = float(marker_info["marker_first_sample_tvt"])
        expected_marker_values[marker_name] = marker_tvt
        for donor_well_id, donor_fold in (("donor_a", 1), ("donor_b", 2)):
            donor_rows.append(
                {
                    "well_id": donor_well_id,
                    "fold": donor_fold,
                    "template_fingerprint": full_fingerprint,
                    "marker_name": marker_name,
                    "marker_tvt": marker_tvt,
                    "marker_boundary_observed": True,
                }
            )
        # 查询井故意放入 +100 ft 的错误 marker；合法汇总必须排除它。
        donor_rows.append(
            {
                "well_id": "cropped_query",
                "fold": 0,
                "template_fingerprint": full_fingerprint,
                "marker_name": marker_name,
                "marker_tvt": marker_tvt + 100.0,
                "marker_boundary_observed": True,
            }
        )

    transferred = aggregate_outer_train_markers(
        query_well_id="cropped_query",
        query_fold=0,
        query_template_fingerprint=cropped_fingerprint,
        donor_markers=pd.DataFrame(donor_rows),
        marker_names=FROZEN_MARKERS,
        maximum_marker_range_ft=float(config["maximum_donor_marker_range_ft"]),
    )
    transferred_values = transferred.set_index("marker_name")["marker_tvt"]
    recovery_errors = [
        abs(float(transferred_values.loc[name]) - expected_marker_values[name])
        for name in FROZEN_MARKERS
    ]

    # 查询井若没有被排除，结果会整体偏离 donor marker，下面检查会立刻失败。
    query_excluded = all(error <= 1e-12 for error in recovery_errors)

    marker_positions = np.asarray(
        [expected_marker_values[name] for name in FROZEN_MARKERS],
        dtype=np.float64,
    )
    q_test_tvt = (marker_positions[:-1] + marker_positions[1:]) / 2.0
    q_values, _, q_valid = tvt_to_marker_coordinate(q_test_tvt, marker_positions)
    recovered_tvt, reverse_valid = marker_coordinate_to_tvt(q_values, marker_positions)
    if not bool(q_valid.all() and reverse_valid.all()):
        maximum_q_roundtrip_error = float("inf")
    else:
        maximum_q_roundtrip_error = float(np.max(np.abs(recovered_tvt - q_test_tvt)))

    return {
        "cropped_template_fingerprint_equal": full_fingerprint
        == cropped_fingerprint,
        "query_well_excluded_from_donors": bool(query_excluded),
        "maximum_crop_recovery_marker_error_ft": float(max(recovery_errors)),
        "maximum_q_roundtrip_error": maximum_q_roundtrip_error,
    }


def evaluate_success_checks(
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, bool]:
    """逐项应用实验卡中全部覆盖、oracle、合法门控和负对照门槛。"""

    conditions = config["success_conditions"]

    def finite_metric(name: str, default: float) -> float:
        value = float(metrics.get(name, default))
        return value if math.isfinite(value) else default

    checks = {
        "template_match_well_coverage_pass": finite_metric(
            "template_match_well_coverage", -math.inf
        )
        >= float(conditions["minimum_template_match_well_coverage"]),
        "auditable_block_coverage_pass": finite_metric(
            "auditable_block_coverage", -math.inf
        )
        >= float(conditions["minimum_auditable_block_coverage"]),
        "withheld_marker_error_pass": finite_metric(
            "withheld_marker_p90_error_ft", math.inf
        )
        <= float(conditions["maximum_withheld_marker_p90_error_ft"]),
        "ambiguous_well_fraction_pass": finite_metric(
            "ambiguous_well_fraction", math.inf
        )
        <= float(conditions["maximum_ambiguous_well_fraction"]),
        "cross_zone_alias_pass": finite_metric(
            "cross_zone_fraction_among_wrong_top1", -math.inf
        )
        >= float(conditions["minimum_cross_zone_fraction_among_wrong_top1"]),
        "oracle_zone_rank_pass": finite_metric(
            "oracle_zone_normalized_rank_improvement", -math.inf
        )
        >= float(conditions["minimum_oracle_zone_normalized_rank_improvement"]),
        "oracle_zone_top5_pass": finite_metric("oracle_zone_top5_gain", -math.inf)
        >= float(conditions["minimum_oracle_zone_top5_gain"]),
        "oracle_zone_bootstrap_pass": finite_metric(
            "oracle_zone_bootstrap_ci_upper", math.inf
        )
        < float(conditions["maximum_oracle_zone_bootstrap_ci_upper"]),
        "pf_true_zone_agreement_pass": finite_metric(
            "pf_true_zone_agreement", -math.inf
        )
        >= float(conditions["minimum_pf_true_zone_agreement"]),
        "legal_zone_rank_pass": finite_metric(
            "legal_zone_normalized_rank_improvement", -math.inf
        )
        >= float(conditions["minimum_legal_zone_normalized_rank_improvement"]),
        "legal_zone_bootstrap_pass": finite_metric(
            "legal_zone_bootstrap_ci_upper", math.inf
        )
        < float(conditions["maximum_legal_zone_bootstrap_ci_upper"]),
        "wrong_marker_control_pass": finite_metric(
            "legal_gain_vs_wrong_marker_gate", -math.inf
        )
        >= float(conditions["minimum_legal_gain_vs_wrong_marker_gate"]),
        "random_count_control_pass": finite_metric(
            "legal_gain_vs_random_count_gate", -math.inf
        )
        >= float(conditions["minimum_legal_gain_vs_random_count_gate"]),
        "q_roundtrip_pass": finite_metric("maximum_q_roundtrip_error", math.inf)
        <= float(conditions["maximum_q_roundtrip_error"]),
        "crop_recovery_pass": finite_metric(
            "maximum_crop_recovery_marker_error_ft", math.inf
        )
        <= float(conditions["maximum_crop_recovery_marker_error_ft"]),
    }
    checks["marker_zone_rank_supported"] = bool(all(checks.values()))
    return checks


def _inspect_one_typewell(task: dict[str, Any]) -> dict[str, Any]:
    """读取一口井的合法模板身份；仅 outer-train 额外读取 Geology。"""

    well_id = str(task["well_id"])
    fold_id = int(task["fold"])
    typewell_path = Path(str(task["typewell_path"]))
    marker_names = list(task["marker_names"])
    tail_rows = int(task["tail_rows"])
    validation_fold = int(task["validation_fold"])

    if fold_id == validation_fold:
        # 关键隔离：fold 0 在合法阶段物理上不读取 Geology 列。
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    else:
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR", "Geology"])

    template_fingerprint = typewell_template_fingerprint(
        typewell[["TVT", "GR"]],
        tail_rows=tail_rows,
    )
    fingerprint_row = {
        "well_id": well_id,
        "fold": fold_id,
        "template_fingerprint": template_fingerprint,
    }

    donor_rows: list[dict[str, Any]] = []
    if fold_id != validation_fold:
        extracted_markers = extract_typewell_markers(typewell)
        for marker_name in marker_names:
            marker_info = extracted_markers[marker_name]
            donor_rows.append(
                {
                    "well_id": well_id,
                    "fold": fold_id,
                    "template_fingerprint": template_fingerprint,
                    "marker_name": marker_name,
                    # 实验卡固定使用标签第一次出现的采样 TVT，而不是边界中点。
                    "marker_tvt": float(marker_info["marker_first_sample_tvt"]),
                    "marker_boundary_observed": bool(
                        marker_info["marker_boundary_observed"]
                    ),
                }
            )
    return {"fingerprint_row": fingerprint_row, "donor_rows": donor_rows}


def build_legal_template_inputs(
    registry: pd.DataFrame,
    raw_train_dir: Path,
    config: dict[str, Any],
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 773 口井指纹和 outer-train donor，始终隔离 fold 0 Geology。"""

    tasks: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        well_id = str(row.well_id)
        typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
        if not typewell_path.is_file():
            raise FileNotFoundError(f"P2-M01 找不到 Typewell：{typewell_path}")
        tasks.append(
            {
                "well_id": well_id,
                "fold": int(row.fold),
                "typewell_path": str(typewell_path),
                "marker_names": list(config["marker_names"]),
                "tail_rows": int(config["template_tail_rows"]),
                "validation_fold": int(config["fold_id"]),
            }
        )

    completed_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        futures = [executor.submit(_inspect_one_typewell, task) for task in tasks]
        for completed_number, future in enumerate(as_completed(futures), start=1):
            completed_results.append(future.result())
            if completed_number % 200 == 0 or completed_number == len(futures):
                print(
                    f"P2-M01 合法模板读取：{completed_number}/{len(futures)}",
                    flush=True,
                )

    fingerprint_rows = [result["fingerprint_row"] for result in completed_results]
    donor_rows = [
        donor_row
        for result in completed_results
        for donor_row in result["donor_rows"]
    ]
    fingerprints = pd.DataFrame(fingerprint_rows)
    fingerprints = fingerprints.sort_values("well_id", kind="stable").reset_index(
        drop=True
    )
    donors = pd.DataFrame(donor_rows)
    donors = donors.sort_values(
        ["template_fingerprint", "well_id", "marker_name"], kind="stable"
    ).reset_index(drop=True)

    if len(fingerprints) != len(registry):
        raise ValueError("P2-M01 没有为全部注册井生成模板指纹")
    if bool(fingerprints["well_id"].duplicated().any()):
        raise ValueError("P2-M01 模板指纹表含重复 well_id")
    if donors["fold"].astype(int).eq(int(config["fold_id"])).any():
        raise ValueError("P2-M01 donor 表意外含 fold 0 Geology")
    return fingerprints, donors


def build_legal_marker_table(
    selected_registry: pd.DataFrame,
    fingerprints: pd.DataFrame,
    donor_markers: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """逐口验证井用同指纹 outer-train donor 汇总六个 marker。"""

    fingerprint_by_well = fingerprints.set_index("well_id")[
        "template_fingerprint"
    ].astype(str)
    output_tables: list[pd.DataFrame] = []
    for row in selected_registry.itertuples(index=False):
        well_id = str(row.well_id)
        if well_id not in fingerprint_by_well.index:
            raise ValueError(f"P2-M01 指纹表缺少验证井 {well_id}")
        transferred = aggregate_outer_train_markers(
            query_well_id=well_id,
            query_fold=int(row.fold),
            query_template_fingerprint=str(fingerprint_by_well.loc[well_id]),
            donor_markers=donor_markers,
            marker_names=config["marker_names"],
            maximum_marker_range_ft=float(config["maximum_donor_marker_range_ft"]),
        )
        output_tables.append(transferred)

    if not output_tables:
        raise ValueError("P2-M01 没有选中验证井")
    legal_markers = pd.concat(output_tables, ignore_index=True)
    legal_markers = legal_markers.sort_values(
        ["well_id", "marker_name"], kind="stable"
    ).reset_index(drop=True)
    validate_legal_marker_table(
        legal_markers,
        expected_fold=int(config["fold_id"]),
    )
    return legal_markers


def build_withheld_marker_oracle(
    selected_registry: pd.DataFrame,
    raw_train_dir: Path,
    legal_markers: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """合法 marker 已落盘后，读取验证井自身 Geology 做独立核对。"""

    legal_index = legal_markers.set_index(["well_id", "marker_name"])
    output_rows: list[dict[str, Any]] = []
    for row in selected_registry.itertuples(index=False):
        well_id = str(row.well_id)
        typewell_path = raw_train_dir / f"{well_id}__typewell.csv"
        # 这是本脚本第一次读取验证井 Geology；调用位置位于合法落盘之后。
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "Geology"])
        extracted_markers = extract_typewell_markers(typewell)
        for marker_name in config["marker_names"]:
            marker_info = extracted_markers[marker_name]
            withheld_tvt = float(marker_info["marker_first_sample_tvt"])
            withheld_observed = bool(marker_info["marker_boundary_observed"])
            legal_row = legal_index.loc[(well_id, marker_name)]
            transferred_tvt = float(legal_row["marker_tvt"])
            comparable = bool(
                legal_row["marker_usable"]
                and withheld_observed
                and np.isfinite(transferred_tvt)
                and np.isfinite(withheld_tvt)
            )
            output_rows.append(
                {
                    "well_id": well_id,
                    "fold": int(row.fold),
                    "marker_name": marker_name,
                    "transferred_marker_tvt": transferred_tvt,
                    "transferred_marker_usable": bool(legal_row["marker_usable"]),
                    "withheld_marker_tvt": withheld_tvt,
                    "withheld_boundary_observed": withheld_observed,
                    "marker_comparable": comparable,
                    "absolute_marker_error_ft": (
                        abs(transferred_tvt - withheld_tvt)
                        if comparable
                        else float("nan")
                    ),
                }
            )
    withheld = pd.DataFrame(output_rows)
    return withheld.sort_values(
        ["well_id", "marker_name"], kind="stable"
    ).reset_index(drop=True)


def _complete_marker_profile(
    table: pd.DataFrame,
    marker_names: list[str],
    value_column: str,
    usable_column: str,
) -> np.ndarray | None:
    """按冻结顺序取六 marker；缺失、禁用或非递增时返回 None。"""

    indexed = table.set_index("marker_name")
    if any(marker_name not in indexed.index for marker_name in marker_names):
        return None
    ordered = indexed.loc[marker_names]
    if not ordered[usable_column].astype(bool).all():
        return None
    marker_positions = pd.to_numeric(
        ordered[value_column], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(marker_positions).all():
        return None
    if not np.all(np.diff(marker_positions) > 0.0):
        return None
    return marker_positions


def build_outer_template_profiles(
    donor_markers: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """把 outer-train donor 汇总成可用于错厚度负对照的模板剖面。"""

    observed = donor_markers.loc[donor_markers["marker_boundary_observed"].eq(True)].copy()
    observed["marker_tvt"] = pd.to_numeric(observed["marker_tvt"], errors="coerce")
    observed = observed.loc[
        np.isfinite(observed["marker_tvt"].to_numpy(dtype=np.float64))
    ]

    # 同一井-marker 只贡献一个中位数，避免重复标签行获得额外权重。
    one_per_well = (
        observed.groupby(
            ["template_fingerprint", "well_id", "marker_name"], sort=True
        )["marker_tvt"]
        .median()
        .reset_index()
    )
    profiles: dict[str, np.ndarray] = {}
    for template_fingerprint, template_rows in one_per_well.groupby(
        "template_fingerprint", sort=True
    ):
        summary = (
            template_rows.groupby("marker_name", sort=True)["marker_tvt"]
            .agg(["median", "min", "max"])
            .reset_index()
        )
        summary["usable"] = (
            summary["max"] - summary["min"]
        ) <= float(config["maximum_donor_marker_range_ft"])
        profile = _complete_marker_profile(
            summary.rename(
                columns={
                    "median": "profile_marker_tvt",
                    "usable": "profile_marker_usable",
                }
            ),
            marker_names=list(config["marker_names"]),
            value_column="profile_marker_tvt",
            usable_column="profile_marker_usable",
        )
        if profile is not None:
            profiles[str(template_fingerprint)] = profile
    return profiles


def choose_wrong_marker_profile(
    legal_markers: np.ndarray,
    legal_template_fingerprint: str,
    outer_profiles: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, str | None]:
    """选总厚度最接近的其他模板，只替换五段厚度并保留 ANCC 锚点。"""

    legal_total_thickness = float(legal_markers[-1] - legal_markers[0])
    candidates: list[tuple[float, str, np.ndarray]] = []
    for template_fingerprint, candidate_profile in outer_profiles.items():
        if template_fingerprint == legal_template_fingerprint:
            continue
        candidate_total = float(candidate_profile[-1] - candidate_profile[0])
        candidates.append(
            (
                abs(candidate_total - legal_total_thickness),
                template_fingerprint,
                candidate_profile,
            )
        )
    if not candidates:
        return None, None
    _, selected_fingerprint, selected_profile = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    wrong_markers = np.empty_like(legal_markers)
    wrong_markers[0] = legal_markers[0]
    wrong_markers[1:] = legal_markers[0] + np.cumsum(np.diff(selected_profile))
    return wrong_markers, selected_fingerprint


def _top_k_from_rank(rank_information: dict[str, Any], top_k: int) -> bool:
    """只有真实候选在门内且名次不大于 K 时，才算 top-K 命中。"""

    return bool(
        rank_information["true_included"]
        and int(rank_information["rank"]) <= int(top_k)
    )


def _best_eligible_position(costs: np.ndarray, eligible_mask: np.ndarray) -> int:
    """返回门内最低有限 cost 的位置；门内为空时返回 -1。"""

    comparable = np.asarray(eligible_mask, dtype=bool) & np.isfinite(costs)
    positions = np.flatnonzero(comparable)
    if len(positions) == 0:
        return -1
    local_best = int(np.argmin(costs[positions]))
    return int(positions[local_best])


def _load_f01b_well_inputs(
    well_id: str,
    f01b_artifact_dir: Path,
    expected_experiment_fingerprint: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """读取一口井冻结 F01b 块、PF 中心路径和五尺度合法 score。"""

    block_path = f01b_artifact_dir / "legal_blocks" / f"{well_id}.parquet"
    row_path = f01b_artifact_dir / "legal_paths" / f"{well_id}.parquet"
    score_path = f01b_artifact_dir / "legal_scores" / f"{well_id}.npz"
    runtime_path = f01b_artifact_dir / "legal_runtime" / f"{well_id}.json"
    for required_path in (block_path, row_path, score_path, runtime_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"P2-M01 找不到 F01b 单井产物：{required_path}")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("experiment_fingerprint") != expected_experiment_fingerprint:
        raise ValueError(f"{well_id} F01b 单井缓存指纹不一致")

    blocks = pd.read_parquet(block_path)
    row_path_frame = pd.read_parquet(
        row_path,
        columns=["well_id", "fold", "row_index", "f01b_pf_center_tvt"],
    )
    with np.load(score_path, allow_pickle=False) as score_file:
        score_arrays = {name: score_file[name].copy() for name in score_file.files}

    expected_arrays = {
        "row_index",
        "row_to_block",
        "offsets_ft",
        "scale_scores",
        "scale_pair_counts",
        "block_md_mid",
    }
    if set(score_arrays) != expected_arrays:
        raise ValueError(f"{well_id} F01b score 数组名不等于冻结集合")
    return blocks, row_path_frame, score_arrays


def score_one_well_blocks(
    well_id: str,
    fold_id: int,
    baseline_rows: pd.DataFrame,
    legal_marker_rows: pd.DataFrame,
    withheld_marker_rows: pd.DataFrame,
    outer_profiles: dict[str, np.ndarray],
    f01b_artifact_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """在一口验证井的冻结 41 候选上计算五种预注册排名。"""

    blocks, legal_path, score_arrays = _load_f01b_well_inputs(
        well_id=well_id,
        f01b_artifact_dir=f01b_artifact_dir,
        expected_experiment_fingerprint=str(config["f01b_experiment_fingerprint"]),
    )
    blocks = blocks.sort_values("block_position", kind="stable").reset_index(drop=True)
    legal_path = legal_path.sort_values("row_index", kind="stable").reset_index(drop=True)
    baseline_rows = baseline_rows.sort_values("row_index", kind="stable").reset_index(
        drop=True
    )

    score_row_index = score_arrays["row_index"].astype(np.int64)
    path_row_index = legal_path["row_index"].to_numpy(dtype=np.int64)
    truth_row_index = baseline_rows["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(score_row_index, path_row_index):
        raise ValueError(f"{well_id} F01b score 与 PF 中心行键不一致")
    if not np.array_equal(score_row_index, truth_row_index):
        raise ValueError(f"{well_id} F01b score 与 P2B00 真值行键不一致")
    if not legal_path["fold"].astype(int).eq(fold_id).all():
        raise ValueError(f"{well_id} F01b 路径 fold 不一致")
    if not baseline_rows["fold"].astype(int).eq(fold_id).all():
        raise ValueError(f"{well_id} P2B00 fold 不一致")

    offsets_ft = score_arrays["offsets_ft"].astype(np.float64)
    if not np.array_equal(offsets_ft, np.asarray(config["offset_grid_ft"])):
        raise ValueError(f"{well_id} F01b offset 网格与 M01 配置不一致")
    row_to_block = score_arrays["row_to_block"].astype(np.int64)
    number_of_blocks = int(score_arrays["scale_scores"].shape[1])
    if len(blocks) != number_of_blocks:
        raise ValueError(f"{well_id} F01b block 表与 score 块数不一致")
    if not np.array_equal(
        blocks["block_position"].to_numpy(dtype=np.int64),
        np.arange(number_of_blocks, dtype=np.int64),
    ):
        raise ValueError(f"{well_id} F01b block_position 不是从 0 连续编号")
    if row_to_block.shape != score_row_index.shape:
        raise ValueError(f"{well_id} F01b row_to_block 长度错误")
    if len(row_to_block) == 0 or row_to_block.min() < 0 or row_to_block.max() >= number_of_blocks:
        raise ValueError(f"{well_id} F01b row_to_block 超出范围")

    block_rows = np.bincount(row_to_block, minlength=number_of_blocks).astype(np.float64)
    emission_costs, support_fraction = build_emission_costs(
        scale_scores=score_arrays["scale_scores"],
        scale_pair_counts=score_arrays["scale_pair_counts"],
        block_rows=block_rows,
        offsets_ft=offsets_ft,
        minimum_valid_scales=F01B_MINIMUM_VALID_SCALES,
    )

    # 每个块只用块内中位数表示 PF 中心和真实 TVT，避免长块端点主导层位判断。
    row_values = pd.DataFrame(
        {
            "block_position": row_to_block,
            "pf_center_tvt": legal_path["f01b_pf_center_tvt"].to_numpy(
                dtype=np.float64
            ),
            "target_tvt": baseline_rows["target_tvt"].to_numpy(dtype=np.float64),
        }
    )
    block_centers = row_values.groupby("block_position", sort=True).median()
    if len(block_centers) != number_of_blocks:
        raise ValueError(f"{well_id} 存在没有隐藏行的 F01b 块")

    legal_markers = _complete_marker_profile(
        legal_marker_rows,
        marker_names=list(config["marker_names"]),
        value_column="marker_tvt",
        usable_column="marker_usable",
    )
    withheld_markers = _complete_marker_profile(
        withheld_marker_rows.rename(
            columns={"withheld_boundary_observed": "withheld_marker_usable"}
        ),
        marker_names=list(config["marker_names"]),
        value_column="withheld_marker_tvt",
        usable_column="withheld_marker_usable",
    )
    legal_template_fingerprint = str(
        legal_marker_rows["template_fingerprint"].astype(str).iloc[0]
    )
    wrong_markers: np.ndarray | None = None
    wrong_template_fingerprint: str | None = None
    if legal_markers is not None:
        wrong_markers, wrong_template_fingerprint = choose_wrong_marker_profile(
            legal_markers=legal_markers,
            legal_template_fingerprint=legal_template_fingerprint,
            outer_profiles=outer_profiles,
        )

    output_rows: list[dict[str, Any]] = []
    all_candidates_gate = np.ones(len(offsets_ft), dtype=bool)
    for block_position in range(number_of_blocks):
        costs = emission_costs[block_position]
        pf_center_tvt = float(block_centers.loc[block_position, "pf_center_tvt"])
        target_tvt = float(block_centers.loc[block_position, "target_tvt"])
        candidate_tvt = pf_center_tvt + offsets_ft
        true_candidate_position = int(np.argmin(np.abs(candidate_tvt - target_tvt)))
        true_offset_ft = float(target_tvt - pf_center_tvt)
        raw_rank = normalized_gated_rank(
            costs=costs,
            true_candidate_position=true_candidate_position,
            eligible_mask=all_candidates_gate,
        )
        raw_best_position = _best_eligible_position(costs, all_candidates_gate)

        oracle_rank: dict[str, Any] | None = None
        legal_rank: dict[str, Any] | None = None
        wrong_rank: dict[str, Any] | None = None
        random_rank: dict[str, Any] | None = None
        oracle_true_zone = -1
        raw_best_oracle_zone = -1
        legal_pf_zone = -1
        raw_best_cross_zone: bool | float = float("nan")
        pf_true_zone_agreement: bool | float = float("nan")
        raw_q_error = float("nan")
        legal_q_error = float("nan")

        # Oracle 区间只在验证自身六 marker 完整、真值位于 ANCC～BUDA 时定义。
        oracle_auditable = False
        if withheld_markers is not None:
            candidate_q_oracle, candidate_zone_oracle, _ = tvt_to_marker_coordinate(
                candidate_tvt,
                withheld_markers,
            )
            true_q_array, true_zone_array, true_valid_array = tvt_to_marker_coordinate(
                np.asarray([target_tvt]),
                withheld_markers,
            )
            if bool(true_valid_array[0]):
                oracle_true_zone = int(true_zone_array[0])
                oracle_gate = candidate_zone_oracle == oracle_true_zone
                oracle_rank = normalized_gated_rank(
                    costs=costs,
                    true_candidate_position=true_candidate_position,
                    eligible_mask=oracle_gate,
                )
                oracle_auditable = True
                if raw_best_position >= 0:
                    raw_best_oracle_zone = int(candidate_zone_oracle[raw_best_position])
                    raw_best_cross_zone = bool(
                        raw_best_oracle_zone != oracle_true_zone
                    )
                    if np.isfinite(candidate_q_oracle[raw_best_position]):
                        raw_q_error = float(
                            candidate_q_oracle[raw_best_position] - true_q_array[0]
                        )

        # 合法 gate 只由转移 marker 和 PF 中心决定；target 只参与事后排名。
        legal_marker_available = False
        legal_gate = np.zeros(len(offsets_ft), dtype=bool)
        if legal_markers is not None:
            candidate_q_legal, candidate_zone_legal, _ = tvt_to_marker_coordinate(
                candidate_tvt,
                legal_markers,
            )
            target_q_legal, _, target_valid_legal = tvt_to_marker_coordinate(
                np.asarray([target_tvt]),
                legal_markers,
            )
            _, pf_zone_array, pf_valid_array = tvt_to_marker_coordinate(
                np.asarray([pf_center_tvt]),
                legal_markers,
            )
            if bool(pf_valid_array[0]):
                legal_pf_zone = int(pf_zone_array[0])
                legal_gate = candidate_zone_legal == legal_pf_zone
            legal_rank = normalized_gated_rank(
                costs=costs,
                true_candidate_position=true_candidate_position,
                eligible_mask=legal_gate,
            )
            legal_marker_available = True
            if oracle_auditable:
                pf_true_zone_agreement = bool(legal_pf_zone == oracle_true_zone)
            legal_best_position = _best_eligible_position(costs, legal_gate)
            if (
                legal_best_position >= 0
                and bool(target_valid_legal[0])
                and np.isfinite(candidate_q_legal[legal_best_position])
            ):
                legal_q_error = float(
                    candidate_q_legal[legal_best_position] - target_q_legal[0]
                )

            # 等候选数随机连续 gate 使用块键和固定 seed，结果可跨进程复现。
            random_gate = deterministic_contiguous_gate(
                number_of_candidates=len(offsets_ft),
                gate_size=int(legal_rank["candidate_count"]),
                key=f"{well_id}:block_{block_position}",
                seed=int(config["random_gate_seed"]),
            )
            random_rank = normalized_gated_rank(
                costs=costs,
                true_candidate_position=true_candidate_position,
                eligible_mask=random_gate,
            )

        if wrong_markers is not None:
            _, wrong_candidate_zones, _ = tvt_to_marker_coordinate(
                candidate_tvt,
                wrong_markers,
            )
            _, wrong_pf_zone, wrong_pf_valid = tvt_to_marker_coordinate(
                np.asarray([pf_center_tvt]),
                wrong_markers,
            )
            wrong_gate = np.zeros(len(offsets_ft), dtype=bool)
            if bool(wrong_pf_valid[0]):
                wrong_gate = wrong_candidate_zones == int(wrong_pf_zone[0])
            wrong_rank = normalized_gated_rank(
                costs=costs,
                true_candidate_position=true_candidate_position,
                eligible_mask=wrong_gate,
            )

        auditable = bool(oracle_auditable and legal_marker_available)

        def rank_value(
            rank_information: dict[str, Any] | None,
            field: str,
            default: Any,
        ) -> Any:
            return default if rank_information is None else rank_information[field]

        output_rows.append(
            {
                "well_id": well_id,
                "fold": fold_id,
                "block_position": block_position,
                "block_md_mid": float(blocks.loc[block_position, "block_md_mid"]),
                "block_rows": int(block_rows[block_position]),
                "block_support_mean": float(np.mean(support_fraction[block_position])),
                "true_offset_ft": true_offset_ft,
                "true_candidate_position": true_candidate_position,
                "raw_best_position": raw_best_position,
                "raw_rank": int(raw_rank["rank"]),
                "raw_normalized_rank": float(raw_rank["normalized_rank"]),
                "raw_top1": _top_k_from_rank(raw_rank, 1),
                "raw_top3": _top_k_from_rank(raw_rank, 3),
                "raw_top5": _top_k_from_rank(raw_rank, 5),
                "oracle_zone_rank": rank_value(oracle_rank, "rank", np.nan),
                "oracle_zone_candidate_count": rank_value(
                    oracle_rank, "candidate_count", 0
                ),
                "oracle_zone_normalized_rank": rank_value(
                    oracle_rank, "normalized_rank", np.nan
                ),
                "oracle_zone_top1": (
                    _top_k_from_rank(oracle_rank, 1) if oracle_rank is not None else np.nan
                ),
                "oracle_zone_top3": (
                    _top_k_from_rank(oracle_rank, 3) if oracle_rank is not None else np.nan
                ),
                "oracle_zone_top5": (
                    _top_k_from_rank(oracle_rank, 5) if oracle_rank is not None else np.nan
                ),
                "legal_zone_rank": rank_value(legal_rank, "rank", np.nan),
                "legal_zone_candidate_count": rank_value(
                    legal_rank, "candidate_count", 0
                ),
                "legal_zone_true_included": rank_value(
                    legal_rank, "true_included", False
                ),
                "legal_zone_normalized_rank": rank_value(
                    legal_rank, "normalized_rank", np.nan
                ),
                "legal_zone_top1": (
                    _top_k_from_rank(legal_rank, 1) if legal_rank is not None else np.nan
                ),
                "legal_zone_top3": (
                    _top_k_from_rank(legal_rank, 3) if legal_rank is not None else np.nan
                ),
                "legal_zone_top5": (
                    _top_k_from_rank(legal_rank, 5) if legal_rank is not None else np.nan
                ),
                "wrong_marker_normalized_rank": rank_value(
                    wrong_rank, "normalized_rank", np.nan
                ),
                "random_count_normalized_rank": rank_value(
                    random_rank, "normalized_rank", np.nan
                ),
                "oracle_true_zone": oracle_true_zone,
                "raw_best_oracle_zone": raw_best_oracle_zone,
                "raw_best_is_wrong": bool(raw_best_position != true_candidate_position),
                "raw_best_cross_zone": raw_best_cross_zone,
                "legal_pf_zone": legal_pf_zone,
                "pf_true_zone_agreement": pf_true_zone_agreement,
                "raw_best_q_minus_true_q": raw_q_error,
                "legal_best_q_minus_true_q": legal_q_error,
                "marker_auditable": auditable,
                "wrong_marker_available": wrong_rank is not None,
                "legal_template_fingerprint": legal_template_fingerprint,
                "wrong_template_fingerprint": wrong_template_fingerprint,
            }
        )
    return pd.DataFrame(output_rows)


def load_selected_baseline(
    selected_registry: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """在合法 marker 落盘后，读取所选验证井的冻结隐藏真值。"""

    baseline_path = CLEAN_ROOT / str(config["baseline_predictions"])
    baseline = pd.read_parquet(
        baseline_path,
        columns=["well_id", "fold", "row_index", "target_tvt"],
        filters=[("fold", "==", int(config["fold_id"]))],
    )
    selected_wells = set(selected_registry["well_id"].astype(str))
    baseline["well_id"] = baseline["well_id"].astype(str)
    baseline = baseline.loc[baseline["well_id"].isin(selected_wells)].copy()
    baseline = baseline.sort_values(["well_id", "row_index"], kind="stable")
    expected_rows = int(selected_registry["hidden_rows"].sum())
    if len(baseline) != expected_rows:
        raise ValueError("P2-M01 P2B00 所选隐藏行数与 fold 表不一致")
    if bool(baseline.duplicated(["well_id", "row_index"]).any()):
        raise ValueError("P2-M01 P2B00 含重复井-行键")
    return baseline.reset_index(drop=True)


def score_selected_wells(
    selected_registry: pd.DataFrame,
    baseline: pd.DataFrame,
    legal_markers: pd.DataFrame,
    withheld_markers: pd.DataFrame,
    donor_markers: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """读取冻结 F01b score，按井计算 block 排名并合并。"""

    outer_profiles = build_outer_template_profiles(donor_markers, config)
    f01b_artifact_dir = CLEAN_ROOT / str(config["f01b_artifact_dir"])
    baseline_groups = {
        str(well_id): group.copy()
        for well_id, group in baseline.groupby("well_id", sort=False)
    }
    legal_groups = {
        str(well_id): group.copy()
        for well_id, group in legal_markers.groupby("well_id", sort=False)
    }
    withheld_groups = {
        str(well_id): group.copy()
        for well_id, group in withheld_markers.groupby("well_id", sort=False)
    }

    block_tables: list[pd.DataFrame] = []
    for completed_number, row in enumerate(
        selected_registry.itertuples(index=False), start=1
    ):
        well_id = str(row.well_id)
        block_tables.append(
            score_one_well_blocks(
                well_id=well_id,
                fold_id=int(row.fold),
                baseline_rows=baseline_groups[well_id],
                legal_marker_rows=legal_groups[well_id],
                withheld_marker_rows=withheld_groups[well_id],
                outer_profiles=outer_profiles,
                f01b_artifact_dir=f01b_artifact_dir,
                config=config,
            )
        )
        if completed_number % 25 == 0 or completed_number == len(selected_registry):
            print(
                f"P2-M01 oracle block 排名：{completed_number}/{len(selected_registry)}",
                flush=True,
            )
    return pd.concat(block_tables, ignore_index=True)


def cluster_bootstrap_interval(
    block_ranks: pd.DataFrame,
    new_column: str,
    baseline_column: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    """按井成簇重采样，返回 ``新排名-原排名`` 均值的 95% 区间。"""

    valid = block_ranks.loc[
        np.isfinite(block_ranks[new_column].to_numpy(dtype=np.float64))
        & np.isfinite(block_ranks[baseline_column].to_numpy(dtype=np.float64))
    ].copy()
    if valid.empty:
        return float("nan"), float("nan")
    valid["paired_difference"] = (
        valid[new_column].to_numpy(dtype=np.float64)
        - valid[baseline_column].to_numpy(dtype=np.float64)
    )
    per_well = valid.groupby("well_id", sort=True)["paired_difference"].agg(
        ["sum", "count"]
    )
    sums = per_well["sum"].to_numpy(dtype=np.float64)
    counts = per_well["count"].to_numpy(dtype=np.float64)
    number_of_wells = len(per_well)
    random_generator = np.random.default_rng(int(seed))
    bootstrap_means = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        sampled_positions = random_generator.integers(
            0, number_of_wells, size=number_of_wells
        )
        bootstrap_means[repetition] = float(
            np.sum(sums[sampled_positions]) / np.sum(counts[sampled_positions])
        )
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def build_per_well_metrics(block_ranks: pd.DataFrame) -> pd.DataFrame:
    """保存逐井覆盖率和五种平均 normalized rank，便于定位收益来源。"""

    output_rows: list[dict[str, Any]] = []
    for well_id, well_rows in block_ranks.groupby("well_id", sort=True):
        auditable = well_rows.loc[well_rows["marker_auditable"].eq(True)]
        output_rows.append(
            {
                "well_id": str(well_id),
                "fold": int(well_rows["fold"].iloc[0]),
                "blocks": int(len(well_rows)),
                "auditable_blocks": int(len(auditable)),
                "auditable_block_fraction": float(len(auditable) / len(well_rows)),
                "raw_mean_normalized_rank": float(
                    auditable["raw_normalized_rank"].mean()
                ),
                "oracle_mean_normalized_rank": float(
                    auditable["oracle_zone_normalized_rank"].mean()
                ),
                "legal_mean_normalized_rank": float(
                    auditable["legal_zone_normalized_rank"].mean()
                ),
                "wrong_marker_mean_normalized_rank": float(
                    auditable["wrong_marker_normalized_rank"].mean()
                ),
                "random_count_mean_normalized_rank": float(
                    auditable["random_count_normalized_rank"].mean()
                ),
                "pf_true_zone_agreement": float(
                    auditable["pf_true_zone_agreement"].mean()
                ),
            }
        )
    return pd.DataFrame(output_rows)


def build_summary(
    mode: str,
    selected_registry: pd.DataFrame,
    legal_markers: pd.DataFrame,
    withheld_markers: pd.DataFrame,
    block_ranks: pd.DataFrame,
    program_controls: dict[str, Any],
    config: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """汇总预注册的全部覆盖、排名、负对照和井级 bootstrap 指标。"""

    marker_names = list(config["marker_names"])
    matched_wells = 0
    ambiguous_wells = 0
    for _, well_rows in legal_markers.groupby("well_id", sort=False):
        profile = _complete_marker_profile(
            well_rows,
            marker_names=marker_names,
            value_column="marker_tvt",
            usable_column="marker_usable",
        )
        matched_wells += int(profile is not None)
        ambiguous_wells += int(bool(well_rows["marker_ambiguous"].astype(bool).any()))

    comparable_markers = withheld_markers.loc[
        withheld_markers["marker_comparable"].eq(True)
    ]
    marker_errors = comparable_markers["absolute_marker_error_ft"].to_numpy(
        dtype=np.float64
    )
    auditable = block_ranks.loc[block_ranks["marker_auditable"].eq(True)].copy()
    raw_wrong = auditable.loc[auditable["raw_best_is_wrong"].eq(True)]

    oracle_ci_lower, oracle_ci_upper = cluster_bootstrap_interval(
        auditable,
        new_column="oracle_zone_normalized_rank",
        baseline_column="raw_normalized_rank",
        repetitions=int(config["cluster_bootstrap_repetitions"]),
        seed=int(config["cluster_bootstrap_seed"]),
    )
    legal_ci_lower, legal_ci_upper = cluster_bootstrap_interval(
        auditable,
        new_column="legal_zone_normalized_rank",
        baseline_column="raw_normalized_rank",
        repetitions=int(config["cluster_bootstrap_repetitions"]),
        seed=int(config["cluster_bootstrap_seed"]),
    )

    def column_mean(column: str) -> float:
        values = pd.to_numeric(auditable[column], errors="coerce")
        return float(values.mean())

    raw_mean = column_mean("raw_normalized_rank")
    oracle_mean = column_mean("oracle_zone_normalized_rank")
    legal_mean = column_mean("legal_zone_normalized_rank")
    wrong_mean = column_mean("wrong_marker_normalized_rank")
    random_mean = column_mean("random_count_normalized_rank")
    number_of_selected_wells = len(selected_registry)

    metrics: dict[str, Any] = {
        "template_match_well_coverage": float(
            matched_wells / number_of_selected_wells
        ),
        "auditable_block_coverage": float(len(auditable) / len(block_ranks)),
        "withheld_marker_p90_error_ft": (
            float(np.quantile(marker_errors, 0.90)) if len(marker_errors) else float("nan")
        ),
        "ambiguous_well_fraction": float(
            ambiguous_wells / number_of_selected_wells
        ),
        "cross_zone_fraction_among_wrong_top1": float(
            pd.to_numeric(raw_wrong["raw_best_cross_zone"], errors="coerce").mean()
        ),
        "raw_mean_normalized_rank": raw_mean,
        "oracle_zone_mean_normalized_rank": oracle_mean,
        "legal_zone_mean_normalized_rank": legal_mean,
        "wrong_marker_mean_normalized_rank": wrong_mean,
        "random_count_mean_normalized_rank": random_mean,
        "oracle_zone_normalized_rank_improvement": raw_mean - oracle_mean,
        "legal_zone_normalized_rank_improvement": raw_mean - legal_mean,
        "oracle_zone_top5_gain": float(
            auditable["oracle_zone_top5"].mean() - auditable["raw_top5"].mean()
        ),
        "pf_true_zone_agreement": float(
            pd.to_numeric(auditable["pf_true_zone_agreement"], errors="coerce").mean()
        ),
        "legal_gain_vs_wrong_marker_gate": wrong_mean - legal_mean,
        "legal_gain_vs_random_count_gate": random_mean - legal_mean,
        "oracle_zone_bootstrap_ci_lower": oracle_ci_lower,
        "oracle_zone_bootstrap_ci_upper": oracle_ci_upper,
        "legal_zone_bootstrap_ci_lower": legal_ci_lower,
        "legal_zone_bootstrap_ci_upper": legal_ci_upper,
        "maximum_q_roundtrip_error": float(
            program_controls["maximum_q_roundtrip_error"]
        ),
        "maximum_crop_recovery_marker_error_ft": float(
            program_controls["maximum_crop_recovery_marker_error_ft"]
        ),
    }
    checks = evaluate_success_checks(metrics, config)
    full_stage_supported = bool(
        mode == "all"
        and checks["marker_zone_rank_supported"]
        and program_controls["cropped_template_fingerprint_equal"]
        and program_controls["query_well_excluded_from_donors"]
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "wells": int(number_of_selected_wells),
        "hidden_rows": int(selected_registry["hidden_rows"].sum()),
        "blocks": int(len(block_ranks)),
        "auditable_blocks": int(len(auditable)),
        "comparable_markers": int(len(comparable_markers)),
        **metrics,
        "checks": checks,
        "stage_supported": full_stage_supported,
        "wall_seconds": float(wall_seconds),
    }


def write_conclusion(summary: dict[str, Any], output_path: Path) -> None:
    """按项目规定的五段式结论生成简明中文记录。"""

    supported = bool(summary["stage_supported"])
    decision = "通过全部门槛" if supported else "没有通过全部门槛"
    text = f"""# P2-M01 结论

事实：本次模式为 `{summary['mode']}`，处理 {summary['wells']} 口井、{summary['blocks']} 个块；模板覆盖率为 {summary['template_match_well_coverage']:.4f}，可审计块覆盖率为 {summary['auditable_block_coverage']:.4f}。合法区间相对原始 normalized rank 改善 {summary['legal_zone_normalized_rank_improvement']:.4f}，实验{decision}。

推断：marker 区间是否能减少 F01b 的跨层重复峰，应以 `summary.json` 中逐项冻结门槛为准。

仍未验证：本诊断没有实现 q-state 连续路径，也没有训练 LightGBM，因此不能说明最终 CV 会提高。

当前只能否定：若失败，只否定精确尾段指纹、六 marker 和 PF 所在区间这一固定实现。

下一步：只有 `stage_supported=true` 才实现 q-state 连续路径；否则返回路线图选择新的最小表示。
"""
    output_path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 smoke/all、配置和独立产物目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """按合法落盘边界运行 M01；不训练模型也不生成 TVT 预测。"""

    arguments = parse_args(argv)
    started = time.perf_counter()
    config = json.loads(Path(arguments.config).read_text(encoding="utf-8"))
    validate_config(config)
    input_hashes = validate_external_inputs(config)
    registry = load_registry(config)
    full_validation_registry = select_validation_registry(registry, config)
    if arguments.mode == "all":
        selected_registry = full_validation_registry
    else:
        # smoke 固定选 fold 0 注册表最前的三口井，不根据任何诊断结果挑井。
        selected_registry = full_validation_registry.iloc[:3].copy().reset_index(drop=True)

    artifact_dir = Path(arguments.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_train_dir = (CLEAN_ROOT / str(config["raw_train_dir"])).resolve()
    fingerprint = experiment_fingerprint(config)
    program_controls = run_program_controls(config)
    conditions = config["success_conditions"]
    if not program_controls["cropped_template_fingerprint_equal"]:
        raise ValueError("P2-M01 裁剪模板指纹正对照失败")
    if not program_controls["query_well_excluded_from_donors"]:
        raise ValueError("P2-M01 query donor 排除正对照失败")
    if program_controls["maximum_crop_recovery_marker_error_ft"] > float(
        conditions["maximum_crop_recovery_marker_error_ft"]
    ):
        raise ValueError("P2-M01 裁剪 marker 恢复正对照失败")
    if program_controls["maximum_q_roundtrip_error"] > float(
        conditions["maximum_q_roundtrip_error"]
    ):
        raise ValueError("P2-M01 q 往返正对照失败")
    write_json_atomic(artifact_dir / "program_controls.json", program_controls)

    print(
        f"P2-M01 开始合法阶段：全体 {len(registry)} 口井模板，"
        f"验证 {len(selected_registry)} 口井；fold 0 Geology 暂不读取。",
        flush=True,
    )
    fingerprints, donor_markers = build_legal_template_inputs(
        registry=registry,
        raw_train_dir=raw_train_dir,
        config=config,
    )
    legal_markers = build_legal_marker_table(
        selected_registry=selected_registry,
        fingerprints=fingerprints,
        donor_markers=donor_markers,
        config=config,
    )
    write_csv_atomic(artifact_dir / "template_fingerprints.csv", fingerprints)
    write_csv_atomic(artifact_dir / "legal_markers.csv", legal_markers)

    # 从磁盘重新读取并验证，形成合法/oracle 两阶段之间的硬边界。
    saved_legal_markers = pd.read_csv(
        artifact_dir / "legal_markers.csv", dtype={"well_id": str}
    )
    validate_legal_marker_table(saved_legal_markers, expected_fold=int(config["fold_id"]))
    legal_phase = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "all_template_wells": int(len(fingerprints)),
        "outer_train_geology_wells": int(
            donor_markers["well_id"].astype(str).nunique()
        ),
        "validation_geology_read": False,
        "legal_marker_rows": int(len(saved_legal_markers)),
        "legal_marker_sha256": file_sha256(artifact_dir / "legal_markers.csv"),
    }
    write_json_atomic(artifact_dir / "legal_phase.json", legal_phase)

    print(
        "P2-M01 合法 marker 已落盘并通过禁止列检查；现在才进入 oracle 评分阶段。",
        flush=True,
    )
    withheld_markers = build_withheld_marker_oracle(
        selected_registry=selected_registry,
        raw_train_dir=raw_train_dir,
        legal_markers=saved_legal_markers,
        config=config,
    )
    write_csv_atomic(artifact_dir / "withheld_marker_oracle.csv", withheld_markers)
    baseline = load_selected_baseline(selected_registry, config)
    block_ranks = score_selected_wells(
        selected_registry=selected_registry,
        baseline=baseline,
        legal_markers=saved_legal_markers,
        withheld_markers=withheld_markers,
        donor_markers=donor_markers,
        config=config,
    )

    if arguments.mode == "all":
        if len(selected_registry) != int(config["expected_fold_wells"]):
            raise ValueError("P2-M01 full fold 0 井数不等于冻结值")
        if int(selected_registry["hidden_rows"].sum()) != int(
            config["expected_hidden_rows"]
        ):
            raise ValueError("P2-M01 full fold 0 隐藏行数不等于冻结值")
        if len(block_ranks) != int(config["expected_f01b_blocks"]):
            raise ValueError("P2-M01 full F01b 块数不等于冻结值")

    write_parquet_atomic(artifact_dir / "block_ranks.parquet", block_ranks)
    per_well = build_per_well_metrics(block_ranks)
    write_csv_atomic(artifact_dir / "per_well.csv", per_well)
    wall_seconds = float(time.perf_counter() - started)
    summary = build_summary(
        mode=arguments.mode,
        selected_registry=selected_registry,
        legal_markers=saved_legal_markers,
        withheld_markers=withheld_markers,
        block_ranks=block_ranks,
        program_controls=program_controls,
        config=config,
        wall_seconds=wall_seconds,
    )
    summary["experiment_fingerprint"] = fingerprint
    write_json_atomic(artifact_dir / "summary.json", summary)
    lineage = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "input_hashes": input_hashes,
        "template_fingerprints_sha256": file_sha256(
            artifact_dir / "template_fingerprints.csv"
        ),
        "legal_markers_sha256": file_sha256(artifact_dir / "legal_markers.csv"),
        "legal_markers_saved_before_validation_geology": True,
        "legal_markers_saved_before_hidden_target": True,
        "validation_fold_geology_used_only_for_withheld_oracle": True,
        "f01b_score_used_only_after_legal_marker_save": True,
    }
    write_json_atomic(artifact_dir / "lineage.json", lineage)
    runtime = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": fingerprint,
        "mode": arguments.mode,
        "wells": int(len(selected_registry)),
        "blocks": int(len(block_ranks)),
        "wall_seconds": wall_seconds,
    }
    write_json_atomic(artifact_dir / "runtime.json", runtime)
    (artifact_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_conclusion(summary, artifact_dir / "conclusion.md")
    print(
        f"P2-M01 完成：stage_supported={summary['stage_supported']}，"
        f"合法 rank 改善={summary['legal_zone_normalized_rank_improvement']:.4f}，"
        f"输出={artifact_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
