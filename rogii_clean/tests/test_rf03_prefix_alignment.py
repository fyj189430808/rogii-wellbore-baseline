"""RF03-D0：可见前缀真实 TVT 与固定错位的 typewell GR 对齐测试。"""

import copy
import json
import sys
import tempfile
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


# 把 rogii_clean 加入模块路径，测试项目中的真实实现。
CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

from src.rf03_prefix_alignment import (
    ALIGNMENT_OFFSETS_FT,
    build_prefix_alignment_margins,
    score_prefix_offsets,
)
from scripts import diagnose_rf03_prefix_alignment as diagnostic_runner
from scripts.diagnose_rf03_prefix_alignment import (
    load_and_validate_diagnostic_config,
    parse_args,
    resolve_configured_path,
    run_alignment_for_registry,
    select_registry_for_mode,
    validate_run_inputs,
)


CONFIG_PATH = CLEAN_ROOT / "configs" / "rf03_d0_prefix_alignment_v1.json"

# 这份字典是 Task 1 的完整冻结合同；测试同时检查值、顺序和 JSON 原生类型。
EXPECTED_DIAGNOSTIC_CONFIG = {
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


@pytest.fixture
def local_tmp_path() -> Iterator[Path]:
    """在工作区内创建临时目录，绕开当前机器系统 TEMP 的权限异常。"""

    with tempfile.TemporaryDirectory(
        prefix=".pytest_rf03_task1_",
        dir=CLEAN_ROOT.parent,
    ) as temporary_directory:
        yield Path(temporary_directory)


def assert_same_json_types(actual: object, expected: object) -> None:
    """递归检查 JSON 值类型，防止 bool 或 50.0 冒充冻结整数。"""

    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, expected_value in expected.items():
            assert_same_json_types(actual[key], expected_value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            assert_same_json_types(actual_value, expected_value)


def make_aligned_prefix_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 horizontal_GR=2×typewell_GR(true_TVT)+5 的可见前缀。"""

    typewell_tvt = np.arange(0.0, 501.0, 1.0)
    typewell_gr = (
        np.sin(typewell_tvt / 7.0)
        + 0.4 * np.sin(typewell_tvt / 19.0)
        + 0.002 * typewell_tvt
    )
    typewell_df = pd.DataFrame({"TVT": typewell_tvt, "GR": typewell_gr})

    md = np.arange(400, dtype=np.float64)
    true_tvt = 80.0 + md
    horizontal_gr = 2.0 * np.interp(true_tvt, typewell_tvt, typewell_gr) + 5.0
    tvt_input = true_tvt.copy()
    tvt_input[300:] = np.nan
    horizontal_df = pd.DataFrame(
        {
            "MD": md,
            "GR": horizontal_gr,
            "TVT_input": tvt_input,
            "TVT": true_tvt,
        }
    )
    return horizontal_df, typewell_df


def write_config(path: Path, config: dict[str, object]) -> None:
    """把测试配置写成 UTF-8 JSON，不依赖生产代码。"""

    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_registry(path: Path, number_of_wells: int) -> pd.DataFrame:
    """写一份结构合法的小注册表；井号故意倒序以测试字符串排序。"""

    well_ids = [f"well_{index:04d}" for index in range(number_of_wells)]
    registry = pd.DataFrame(
        {
            "well_id": list(reversed(well_ids)),
            "pad_id": [f"pad_{index:04d}" for index in range(number_of_wells)],
            "fold": [index % 5 for index in range(number_of_wells)],
            "hidden_rows": [1] * number_of_wells,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False, lineterminator="\n")
    return registry


def write_three_well_alignment_inputs(
    train_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """写三井真实 CSV；水平井刻意不含隐藏真值 TVT 列。"""

    train_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.DataFrame(
        {
            "well_id": ["well_c", "well_a", "well_b"],
            "fold": [2, 0, 1],
        }
    )
    horizontal_df, typewell_df = make_aligned_prefix_pair()
    horizontal_df = horizontal_df.drop(columns=["TVT"])
    horizontal_df["forbidden_horizontal"] = "must_not_be_loaded"
    typewell_df = typewell_df.assign(forbidden_typewell="must_not_be_loaded")

    for well_id in registry["well_id"]:
        horizontal_df.to_csv(
            train_dir / f"{well_id}__horizontal_well.csv",
            index=False,
            lineterminator="\n",
        )
        typewell_df.to_csv(
            train_dir / f"{well_id}__typewell.csv",
            index=False,
            lineterminator="\n",
        )
    return registry, sorted(registry["well_id"].tolist())


def test_frozen_diagnostic_config_loads_with_exact_values_and_types() -> None:
    """唯一冻结 JSON 必须原样加载，且不能发生隐式类型宽化。"""

    config = load_and_validate_diagnostic_config(CONFIG_PATH)

    assert config == EXPECTED_DIAGNOSTIC_CONFIG
    assert_same_json_types(config, EXPECTED_DIAGNOSTIC_CONFIG)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("experiment_id", "RF03_D0_changed"),
        ("experiment_type", "training"),
        ("baseline_id", "another_baseline"),
        ("fold_registry", "artifacts/folds/other.csv"),
        ("fold_registry_sha256", "0" * 64),
        ("expected_wells", 772),
        ("expected_wells", False),
        ("train_dir", "input/elsewhere"),
        ("smoke_output_dir", "artifacts/wrong_smoke"),
        ("full_output_dir", "artifacts/wrong_full"),
        ("offsets_ft", [-20.0, 0.0, -10.0, 10.0, 20.0]),
        ("offsets_ft", [-30.0, -10.0, 0.0, 10.0, 20.0]),
        ("scopes", {"all_visible": None, "tail_500ft": 500.0}),
        ("scopes", {"all_visible": None, "tail_1000ft": 500.0}),
        ("minimum_points", 49),
        ("minimum_points", 50.0),
        ("minimum_points", True),
        ("bootstrap_resamples", 1000),
        ("bootstrap_resamples", 2000.0),
        ("bootstrap_seed", 29),
        ("bootstrap_seed", False),
        ("hidden_tvt_loaded", True),
        ("hidden_tvt_loaded", 0),
    ],
)
def test_frozen_diagnostic_config_rejects_mutated_fields(
    local_tmp_path: Path,
    field_name: str,
    replacement: object,
) -> None:
    """任一冻结值或类型被改动时都必须立即拒绝。"""

    changed_config = copy.deepcopy(EXPECTED_DIAGNOSTIC_CONFIG)
    changed_config[field_name] = replacement
    changed_path = local_tmp_path / "changed.json"
    write_config(changed_path, changed_config)

    with pytest.raises(ValueError, match=field_name):
        load_and_validate_diagnostic_config(changed_path)


def test_frozen_diagnostic_config_rejects_unknown_key(local_tmp_path: Path) -> None:
    """额外键可能形成静默覆盖入口，因此也属于合同异常。"""

    changed_config = copy.deepcopy(EXPECTED_DIAGNOSTIC_CONFIG)
    changed_config["minimum_points_override"] = 10
    changed_path = local_tmp_path / "unknown-key.json"
    write_config(changed_path, changed_config)

    with pytest.raises(ValueError, match="未知|字段"):
        load_and_validate_diagnostic_config(changed_path)


def test_config_and_declared_paths_do_not_depend_on_current_directory(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """相对配置路径和配置内路径都固定锚定 rogii_clean，而不是调用者 CWD。"""

    monkeypatch.chdir(local_tmp_path)
    config = load_and_validate_diagnostic_config(
        Path("configs/rf03_d0_prefix_alignment_v1.json")
    )

    assert config == EXPECTED_DIAGNOSTIC_CONFIG
    assert resolve_configured_path(config["fold_registry"]) == (
        CLEAN_ROOT / config["fold_registry"]
    ).resolve()
    assert resolve_configured_path(config["train_dir"]) == (
        CLEAN_ROOT / config["train_dir"]
    ).resolve()


def test_cli_accepts_only_config_and_smoke_or_full_mode() -> None:
    """CLI 只暴露冻结配置和运行规模，不暴露任何诊断参数覆盖口。"""

    smoke_args = parse_args(
        ["--config", "configs/rf03_d0_prefix_alignment_v1.json", "--mode", "smoke"]
    )
    full_args = parse_args(
        ["--config", "configs/rf03_d0_prefix_alignment_v1.json", "--mode", "full"]
    )

    assert smoke_args.mode == "smoke"
    assert full_args.mode == "full"
    assert smoke_args.config == Path("configs/rf03_d0_prefix_alignment_v1.json")


@pytest.mark.parametrize(
    "forbidden_arguments",
    [
        ["--train-dir", "somewhere"],
        ["--registry", "other.csv"],
        ["--output-dir", "somewhere"],
        ["--minimum-points", "10"],
        ["--offset", "5"],
        ["--scope", "tail_500ft"],
    ],
)
def test_cli_rejects_parameter_and_path_overrides(
    forbidden_arguments: list[str],
) -> None:
    """路径、offset、scope 和最低点数都只能来自冻结 JSON。"""

    valid_arguments = [
        "--config",
        "configs/rf03_d0_prefix_alignment_v1.json",
        "--mode",
        "smoke",
    ]
    with pytest.raises(SystemExit):
        parse_args(valid_arguments + forbidden_arguments)


def test_smoke_selects_first_three_after_string_sort_and_full_requires_count() -> None:
    """smoke 先看到完整注册表再固定取排序前三井；full 不允许少井。"""

    registry = pd.DataFrame(
        {
            "well_id": ["well_10", "well_2", "well_01", "well_1"],
            "fold": [0, 1, 2, 3],
        }
    )

    smoke_registry = select_registry_for_mode(registry, "smoke", expected_wells=4)
    full_registry = select_registry_for_mode(registry, "full", expected_wells=4)

    assert smoke_registry["well_id"].tolist() == ["well_01", "well_1", "well_10"]
    assert full_registry["well_id"].tolist() == [
        "well_01",
        "well_1",
        "well_10",
        "well_2",
    ]
    with pytest.raises(ValueError, match="井数"):
        select_registry_for_mode(registry.iloc[:3], "smoke", expected_wells=4)
    with pytest.raises(ValueError, match="井数"):
        select_registry_for_mode(registry.iloc[:3], "full", expected_wells=4)


def test_run_alignment_reads_exact_column_whitelists_and_no_hidden_tvt(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """三井运行只把五个合法源列读入内存，水平井没有 TVT 仍可运行。"""

    train_dir = local_tmp_path / "train"
    registry, expected_well_ids = write_three_well_alignment_inputs(train_dir)
    original_read_csv = pd.read_csv
    read_calls: list[tuple[str, list[str] | None]] = []

    def tracking_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        usecols = kwargs.get("usecols")
        read_calls.append((Path(path).name, usecols))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(diagnostic_runner.pd, "read_csv", tracking_read_csv)
    offset_scores, margins = run_alignment_for_registry(registry, train_dir, 50)

    assert len(offset_scores) == 3 * 2 * 5
    assert len(margins) == 3 * 2
    assert sorted(offset_scores["well_id"].unique().tolist()) == expected_well_ids
    assert set(offset_scores["scope"]) == {"all_visible", "tail_1000ft"}
    assert offset_scores.groupby(["well_id", "scope"]).size().eq(5).all()
    assert margins.groupby("well_id").size().eq(2).all()

    horizontal_calls = [call for call in read_calls if "horizontal_well" in call[0]]
    typewell_calls = [call for call in read_calls if "typewell" in call[0]]
    assert len(horizontal_calls) == 3
    assert len(typewell_calls) == 3
    assert all(call[1] == ["MD", "GR", "TVT_input"] for call in horizontal_calls)
    assert all(call[1] == ["TVT", "GR"] for call in typewell_calls)


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_registry_count_is_validated_before_any_output_directory(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """smoke/full 都必须先核对完整 773 井注册表，失败不留输出目录。"""

    clean_root = local_tmp_path / "rogii_clean"
    monkeypatch.setattr(diagnostic_runner, "CLEAN_ROOT", clean_root)
    monkeypatch.setattr(diagnostic_runner, "PROJECT_ROOT", local_tmp_path)
    registry_path = clean_root / EXPECTED_DIAGNOSTIC_CONFIG["fold_registry"]
    write_registry(registry_path, 772)
    monkeypatch.setattr(
        diagnostic_runner,
        "file_sha256",
        lambda path: EXPECTED_DIAGNOSTIC_CONFIG["fold_registry_sha256"],
    )
    output_key = "smoke_output_dir" if mode == "smoke" else "full_output_dir"
    output_dir = clean_root / EXPECTED_DIAGNOSTIC_CONFIG[output_key]

    with pytest.raises(ValueError, match="井数"):
        validate_run_inputs(EXPECTED_DIAGNOSTIC_CONFIG, mode)

    assert not output_dir.exists()


def test_fold_hash_is_validated_before_any_output_directory(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注册表内容指纹不符时，在 smoke 正式目录出现前立即停止。"""

    clean_root = local_tmp_path / "rogii_clean"
    monkeypatch.setattr(diagnostic_runner, "CLEAN_ROOT", clean_root)
    monkeypatch.setattr(diagnostic_runner, "PROJECT_ROOT", local_tmp_path)
    registry_path = clean_root / EXPECTED_DIAGNOSTIC_CONFIG["fold_registry"]
    write_registry(registry_path, 773)
    output_dir = clean_root / EXPECTED_DIAGNOSTIC_CONFIG["smoke_output_dir"]

    with pytest.raises(ValueError, match="hash|SHA-256"):
        validate_run_inputs(EXPECTED_DIAGNOSTIC_CONFIG, "smoke")

    assert not output_dir.exists()


def test_all_selected_input_files_are_prechecked_before_computation(
    local_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一选定井文件缺失时，不得读取较早井，更不得创建 smoke 目录。"""

    clean_root = local_tmp_path / "rogii_clean"
    monkeypatch.setattr(diagnostic_runner, "CLEAN_ROOT", clean_root)
    monkeypatch.setattr(diagnostic_runner, "PROJECT_ROOT", local_tmp_path)
    registry_path = clean_root / EXPECTED_DIAGNOSTIC_CONFIG["fold_registry"]
    write_registry(registry_path, 773)
    monkeypatch.setattr(
        diagnostic_runner,
        "file_sha256",
        lambda path: EXPECTED_DIAGNOSTIC_CONFIG["fold_registry_sha256"],
    )
    train_dir = (clean_root / EXPECTED_DIAGNOSTIC_CONFIG["train_dir"]).resolve()
    train_dir.mkdir(parents=True)
    # 只给排序第一井创建两个文件；第二井缺文件必须在任何 pd.read_csv 前发现。
    (train_dir / "well_0000__horizontal_well.csv").touch()
    (train_dir / "well_0000__typewell.csv").touch()
    original_read_csv = pd.read_csv
    read_calls: list[Path] = []

    def track_data_reads(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        current_path = Path(path)
        if current_path == registry_path:
            return original_read_csv(path, *args, **kwargs)
        read_calls.append(current_path)
        raise AssertionError("全部输入预检完成前不应读取逐井 CSV")

    monkeypatch.setattr(diagnostic_runner.pd, "read_csv", track_data_reads)
    output_dir = clean_root / EXPECTED_DIAGNOSTIC_CONFIG["smoke_output_dir"]

    with pytest.raises(FileNotFoundError, match="well_0001"):
        validate_run_inputs(EXPECTED_DIAGNOSTIC_CONFIG, "smoke")

    assert read_calls == []
    assert not output_dir.exists()


def test_true_offset_wins_ncc_and_affine_mae() -> None:
    """在精确仿射匹配数据上，0 ft 必须同时胜过 ±10/20 ft。"""

    horizontal_df, typewell_df = make_aligned_prefix_pair()
    scores = score_prefix_offsets(
        horizontal_df=horizontal_df,
        typewell_df=typewell_df,
        well_id="aligned",
        fold=0,
        scope="all_visible",
        tail_window_ft=None,
        minimum_points=50,
    )
    margins = build_prefix_alignment_margins(scores)

    assert scores["offset_ft"].tolist() == list(ALIGNMENT_OFFSETS_FT)
    zero_score = scores.loc[scores["offset_ft"] == 0.0].iloc[0]
    np.testing.assert_allclose(zero_score["raw_ncc"], 1.0, atol=1e-12)
    np.testing.assert_allclose(zero_score["affine_median_ae"], 0.0, atol=1e-12)
    assert bool(margins.loc[0, "zero_ncc_is_best"])
    assert bool(margins.loc[0, "zero_mae_is_best"])
    assert float(margins.loc[0, "ncc_margin_vs_best_wrong"]) > 0.0
    assert float(margins.loc[0, "mae_margin_vs_best_wrong"]) > 0.0


def test_alignment_uses_common_typewell_support() -> None:
    """五个 offset 必须在同一批公共有效行上评分，避免不同样本量造成假优势。"""

    horizontal_df, typewell_df = make_aligned_prefix_pair()
    scores = score_prefix_offsets(
        horizontal_df=horizontal_df,
        typewell_df=typewell_df,
        well_id="aligned",
        fold=2,
        scope="all_visible",
        tail_window_ft=None,
        minimum_points=50,
    )

    assert scores["n_points"].nunique() == 1
    assert int(scores["n_points"].iloc[0]) == 300
    assert scores["fold"].eq(2).all()


def test_hidden_tvt_does_not_change_prefix_alignment() -> None:
    """修改自然隐藏段 TVT 真值时，全部可见前缀 offset 分数必须逐位不变。"""

    original, typewell_df = make_aligned_prefix_pair()
    changed = original.copy()
    changed.loc[changed["TVT_input"].isna(), "TVT"] += 1000.0

    original_scores = score_prefix_offsets(
        original,
        typewell_df,
        "aligned",
        0,
        "tail_1000ft",
        1000.0,
        50,
    )
    changed_scores = score_prefix_offsets(
        changed,
        typewell_df,
        "aligned",
        0,
        "tail_1000ft",
        1000.0,
        50,
    )

    assert_frame_equal(original_scores, changed_scores, check_exact=True)
