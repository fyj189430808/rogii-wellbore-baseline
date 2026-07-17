"""运行 RF03-D0：773 井可见前缀的真实 TVT 与 ±10/20 ft typewell 对齐诊断。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


# 当前脚本位于 rogii_clean/scripts，因此父目录是干净项目根目录。
CLEAN_ROOT = Path(__file__).resolve().parents[1]

# 原始比赛数据默认位于干净项目的上一级 input 目录。
PROJECT_ROOT = CLEAN_ROOT.parent

# 允许直接运行脚本时导入 rogii_clean/src。
sys.path.insert(0, str(CLEAN_ROOT))

from src.lgbm_data import file_sha256, load_and_validate_registry
from src.rf03_prefix_alignment import (
    build_prefix_alignment_margins,
    score_prefix_offsets,
)


# Task 1 把所有诊断参数冻结在一个 JSON 合同中；这里的副本只用于拒绝静默改值。
FROZEN_DIAGNOSTIC_CONFIG: dict[str, object] = {
    "experiment_id": "RF03_D0_prefix_alignment_v1",
    "experiment_type": "diagnostic_only",
    "baseline_id": "B00_simple_lgbm_v1",
    "fold_registry": "artifacts/folds/spatial_pad_1000_v1.csv",
    "fold_registry_sha256": (
        "0c217c417c6f62a2105c4056e92a23dfbaefa8b1d1fa8f5a11c13637b9cd99ab"
    ),
    "expected_wells": 773,
    "train_dir": "../input/data/raw/train",
    "smoke_output_dir": "artifacts/_smoke/RF03_D0_prefix_alignment_v1",
    "full_output_dir": "artifacts/RF03_D0_prefix_alignment_v1",
    "offsets_ft": [-20.0, -10.0, 0.0, 10.0, 20.0],
    "scopes": {"all_visible": None, "tail_1000ft": 1000.0},
    "minimum_points": 50,
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 42,
    "hidden_tvt_loaded": False,
}


def resolve_configured_path(path_value: str | Path) -> Path:
    """把配置相对路径固定锚定到 rogii_clean，完全不读取当前工作目录。"""

    configured_path = Path(path_value)
    if configured_path.is_absolute():
        return configured_path.resolve()

    # 若调用者显式写 rogii_clean/...，就从项目根解析；其余路径从干净项目根解析。
    if configured_path.parts and configured_path.parts[0] == CLEAN_ROOT.name:
        return (PROJECT_ROOT / configured_path).resolve()
    return (CLEAN_ROOT / configured_path).resolve()


def _validate_frozen_value(
    actual: object,
    expected: object,
    field_path: str,
) -> None:
    """递归比较冻结值和精确类型；发现第一个合同差异立即报错。"""

    # Python 中 bool 是 int 的子类，因此必须使用 type is，不能使用 isinstance。
    if type(actual) is not type(expected):
        raise ValueError(
            f"配置字段 {field_path} 类型不匹配："
            f"actual={type(actual).__name__}, expected={type(expected).__name__}"
        )

    if isinstance(expected, dict):
        actual_dict = actual
        missing_keys = sorted(set(expected) - set(actual_dict))
        unknown_keys = sorted(set(actual_dict) - set(expected))
        if missing_keys or unknown_keys:
            raise ValueError(
                f"配置字段 {field_path} 的字段不匹配："
                f"缺少={missing_keys}，未知={unknown_keys}"
            )
        for key, expected_value in expected.items():
            _validate_frozen_value(
                actual_dict[key],
                expected_value,
                f"{field_path}.{key}",
            )
        return

    if isinstance(expected, list):
        actual_list = actual
        if len(actual_list) != len(expected):
            raise ValueError(
                f"配置字段 {field_path} 长度不匹配："
                f"actual={len(actual_list)}, expected={len(expected)}"
            )
        for index, expected_value in enumerate(expected):
            _validate_frozen_value(
                actual_list[index],
                expected_value,
                f"{field_path}[{index}]",
            )
        return

    if actual != expected:
        raise ValueError(
            f"配置字段 {field_path} 值不匹配：actual={actual!r}, expected={expected!r}"
        )


def load_and_validate_diagnostic_config(path: Path) -> dict:
    """从固定根读取 RF03-D0 JSON，并逐字段验证冻结值与 JSON 类型。"""

    config_path = resolve_configured_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少 RF03-D0 配置：{config_path}")

    loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_frozen_value(loaded_config, FROZEN_DIAGNOSTIC_CONFIG, "config")
    return loaded_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """只接受冻结配置路径和 smoke/full 模式，不提供参数覆盖入口。"""

    parser = argparse.ArgumentParser(description="诊断可见前缀 typewell GR offset")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        required=True,
    )
    return parser.parse_args(argv)


def select_registry_for_mode(
    registry: pd.DataFrame,
    mode: str,
    expected_wells: int,
) -> pd.DataFrame:
    """先验证完整注册表井数，再按字符串 well_id 排序选择 smoke/full 井。"""

    required_columns = {"well_id", "fold"}
    missing_columns = required_columns - set(registry.columns)
    if missing_columns:
        raise ValueError(f"fold 注册表缺少列：{sorted(missing_columns)}")
    if len(registry) != int(expected_wells):
        raise ValueError(
            f"fold 注册表井数不匹配：actual={len(registry)}, expected={expected_wells}"
        )
    if registry["well_id"].astype(str).duplicated().any():
        raise ValueError("fold 注册表含重复 well_id")
    if mode not in {"smoke", "full"}:
        raise ValueError(f"未知运行模式：{mode}")

    sorted_registry = registry.copy()
    sorted_registry["well_id"] = sorted_registry["well_id"].astype(str)
    sorted_registry = sorted_registry.sort_values("well_id").reset_index(drop=True)
    if mode == "smoke":
        return sorted_registry.iloc[:3].copy().reset_index(drop=True)
    return sorted_registry


def _selected_input_paths(
    registry: pd.DataFrame,
    train_dir: Path,
) -> list[tuple[Path, Path]]:
    """返回选定井的水平井/typewell 路径，并在读取任何一井前完成全量预检。"""

    if not train_dir.is_dir():
        raise FileNotFoundError(f"缺少训练数据目录：{train_dir}")

    input_paths: list[tuple[Path, Path]] = []
    missing_paths: list[Path] = []
    for registry_row in registry.itertuples(index=False):
        well_id = str(registry_row.well_id)
        horizontal_path = train_dir / f"{well_id}__horizontal_well.csv"
        typewell_path = train_dir / f"{well_id}__typewell.csv"
        input_paths.append((horizontal_path, typewell_path))
        if not horizontal_path.is_file():
            missing_paths.append(horizontal_path)
        if not typewell_path.is_file():
            missing_paths.append(typewell_path)

    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths[:10])
        raise FileNotFoundError(f"缺少选定井输入文件：{missing_text}")
    return input_paths


def validate_run_inputs(
    config: dict,
    mode: str,
) -> tuple[pd.DataFrame, Path, Path]:
    """验证冻结 registry/hash/井数与全部选定文件；不创建任何输出目录。"""

    _validate_frozen_value(config, FROZEN_DIAGNOSTIC_CONFIG, "config")
    registry_path = resolve_configured_path(config["fold_registry"])
    train_dir = resolve_configured_path(config["train_dir"])
    output_key = "smoke_output_dir" if mode == "smoke" else "full_output_dir"
    if mode not in {"smoke", "full"}:
        raise ValueError(f"未知运行模式：{mode}")
    output_dir = resolve_configured_path(config[output_key])

    if not registry_path.is_file():
        raise FileNotFoundError(f"缺少 fold 注册表：{registry_path}")
    actual_registry_hash = file_sha256(registry_path)
    expected_registry_hash = str(config["fold_registry_sha256"])
    if actual_registry_hash != expected_registry_hash:
        raise ValueError(
            "fold registry SHA-256/hash 不匹配："
            f"actual={actual_registry_hash}, expected={expected_registry_hash}"
        )

    # smoke 也先加载并验证完整 773 井注册表，之后才固定取字符串排序前三井。
    full_registry = load_and_validate_registry(
        registry_path,
        expected_wells=int(config["expected_wells"]),
        expected_rows=None,
    )
    selected_registry = select_registry_for_mode(
        full_registry,
        mode,
        expected_wells=int(config["expected_wells"]),
    )
    _selected_input_paths(selected_registry, train_dir)
    return selected_registry, train_dir, output_dir


def run_alignment_for_registry(
    registry: pd.DataFrame,
    train_dir: Path,
    minimum_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """为注册表中的井计算两个固定 scope×五个固定 offset，并返回分数和 margin。"""

    if type(minimum_points) is not int or minimum_points != 50:
        raise ValueError("minimum_points 必须是冻结整数 50")
    required_columns = {"well_id", "fold"}
    missing_columns = required_columns - set(registry.columns)
    if missing_columns:
        raise ValueError(f"运行注册表缺少列：{sorted(missing_columns)}")
    if registry.empty:
        raise ValueError("运行注册表不能为空")

    sorted_registry = registry.copy()
    sorted_registry["well_id"] = sorted_registry["well_id"].astype(str)
    sorted_registry = sorted_registry.sort_values("well_id").reset_index(drop=True)
    input_paths = _selected_input_paths(sorted_registry, train_dir.resolve())

    score_tables: list[pd.DataFrame] = []
    total_wells = len(sorted_registry)
    registry_rows = list(sorted_registry.itertuples(index=False))
    for well_number, (registry_row, input_pair) in enumerate(
        zip(registry_rows, input_paths, strict=True),
        start=1,
    ):
        horizontal_path, typewell_path = input_pair

        # usecols 是防泄漏合同：水平井隐藏 TVT 列即使存在也不会进入内存。
        horizontal_df = pd.read_csv(
            horizontal_path,
            usecols=["MD", "GR", "TVT_input"],
        )
        typewell_df = pd.read_csv(typewell_path, usecols=["TVT", "GR"])

        score_tables.append(
            score_prefix_offsets(
                horizontal_df,
                typewell_df,
                str(registry_row.well_id),
                int(registry_row.fold),
                "all_visible",
                None,
                minimum_points,
            )
        )
        score_tables.append(
            score_prefix_offsets(
                horizontal_df,
                typewell_df,
                str(registry_row.well_id),
                int(registry_row.fold),
                "tail_1000ft",
                1000.0,
                minimum_points,
            )
        )

        if well_number % 50 == 0 or well_number == total_wells:
            print(f"RF03-D0进度：{well_number}/{total_wells}井", flush=True)

    offset_scores = pd.concat(score_tables, ignore_index=True)
    margins = build_prefix_alignment_margins(offset_scores)
    return offset_scores, margins


def summarize_margin_group(margin_df: pd.DataFrame) -> dict[str, float | int]:
    """汇总一个 scope/fold 的有效井数、正对照胜率和两个 margin。"""

    # eligible 只保留 NCC 与 affine-MAE 都能计算的井。
    eligible = margin_df.loc[
        np.isfinite(margin_df["ncc_margin_vs_best_wrong"])
        & np.isfinite(margin_df["mae_margin_vs_best_wrong"])
    ]

    # 空分组仍返回明确的 0 和 NaN，避免 JSON 缺字段。
    if eligible.empty:
        return {
            "wells": 0,
            "ncc_zero_best_rate": float("nan"),
            "ncc_median_margin": float("nan"),
            "mae_zero_best_rate": float("nan"),
            "mae_median_margin": float("nan"),
            "median_points": float("nan"),
        }

    return {
        "wells": int(len(eligible)),
        "ncc_zero_best_rate": float(eligible["zero_ncc_is_best"].mean()),
        "ncc_median_margin": float(
            eligible["ncc_margin_vs_best_wrong"].median()
        ),
        "mae_zero_best_rate": float(eligible["zero_mae_is_best"].mean()),
        "mae_median_margin": float(
            eligible["mae_margin_vs_best_wrong"].median()
        ),
        "median_points": float(eligible["n_points"].median()),
    }


def build_summary(
    offset_scores: pd.DataFrame,
    margins: pd.DataFrame,
) -> dict[str, object]:
    """构造 overall、逐fold和逐offset的完整诊断 JSON。"""

    scope_summaries: list[dict[str, object]] = []
    for scope in sorted(margins["scope"].unique()):
        scope_df = margins.loc[margins["scope"] == scope]

        # fold=-1 明确表示全部固定 folds 合并，不与真实 fold 0 混淆。
        overall = summarize_margin_group(scope_df)
        overall.update({"scope": str(scope), "fold": -1})
        scope_summaries.append(overall)

        for fold in range(5):
            fold_summary = summarize_margin_group(
                scope_df.loc[scope_df["fold"] == fold]
            )
            fold_summary.update({"scope": str(scope), "fold": int(fold)})
            scope_summaries.append(fold_summary)

    offset_summaries: list[dict[str, object]] = []
    for (scope, offset_ft), group_df in offset_scores.groupby(
        ["scope", "offset_ft"],
        sort=True,
    ):
        valid_ncc = group_df["raw_ncc"].dropna()
        valid_mae = group_df["affine_median_ae"].dropna()
        offset_summaries.append(
            {
                "scope": str(scope),
                "offset_ft": float(offset_ft),
                "wells": int(min(len(valid_ncc), len(valid_mae))),
                "median_raw_ncc": float(valid_ncc.median()),
                "median_affine_median_ae": float(valid_mae.median()),
            }
        )

    # 严格判据：某一指标必须在两个scope的overall和全部五折都超过50%胜率。
    summary_df = pd.DataFrame(scope_summaries)
    eligible_summaries = summary_df.loc[summary_df["wells"] > 0]
    ncc_passes = bool(
        len(eligible_summaries) == 12
        and (eligible_summaries["ncc_zero_best_rate"] > 0.5).all()
    )
    mae_passes = bool(
        len(eligible_summaries) == 12
        and (eligible_summaries["mae_zero_best_rate"] > 0.5).all()
    )

    return {
        "scope_fold_summaries": scope_summaries,
        "offset_summaries": offset_summaries,
        "ncc_passes_all_scopes_and_folds": ncc_passes,
        "mae_passes_all_scopes_and_folds": mae_passes,
        "diagnostic_passes": bool(ncc_passes or mae_passes),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """以 UTF-8、排序键和固定缩进写入可复核 JSON。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """读取冻结 JSON，验证全部输入后运行；Task 1 仅保留现有最小诊断输出。"""

    args = parse_args()
    config = load_and_validate_diagnostic_config(args.config)

    # 此函数只做读和验证；registry/hash/井数/全部选定文件失败时输出目录仍不存在。
    registry, train_dir, output_dir = validate_run_inputs(config, args.mode)
    offset_scores, margins = run_alignment_for_registry(
        registry,
        train_dir,
        int(config["minimum_points"]),
    )
    summary = build_summary(offset_scores, margins)

    # Task 1 不运行 main；即使未来调用，也在全部计算成功后才建立目录。
    output_dir.mkdir(parents=True, exist_ok=False)
    offset_scores.to_csv(
        output_dir / "offset_scores.csv",
        index=False,
        lineterminator="\n",
    )
    margins.to_csv(
        output_dir / "per_well_margins.csv",
        index=False,
        lineterminator="\n",
    )

    write_json(output_dir / "config.json", config)
    write_json(output_dir / "summary.json", summary)

    conclusion_lines = [
        "# RF03-D0 可见前缀 typewell 对齐结论",
        "",
        f"- 运行模式：`{args.mode}`",
        "- baseline_id：`B00_simple_lgbm_v1 = B0`",
        f"- 诊断是否通过：`{summary['diagnostic_passes']}`",
        f"- raw NCC 是否在两个scope及全部fold过半：`{summary['ncc_passes_all_scopes_and_folds']}`",
        f"- affine-MAE 是否在两个scope及全部fold过半：`{summary['mae_passes_all_scopes_and_folds']}`",
        "- 本诊断未读取隐藏 TVT，也未训练模型。",
        "- 诊断通过只允许进入循环平移负对照/F03a，不代表 CV 已改善。",
        "- 详细数字见 `summary.json`、`per_well_margins.csv` 和 `offset_scores.csv`。",
        "",
    ]
    (output_dir / "conclusion.md").write_text(
        "\n".join(conclusion_lines),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"输出目录：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
