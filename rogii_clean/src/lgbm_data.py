"""读取冻结 fold，并把逐井 B00 特征组装成完整行级表。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.lgbm_features import build_simple_lgbm_rows


# 输入是需要记录指纹的文件路径；输出是只由文件内容决定的 SHA-256 字符串，不修改文件。
def file_sha256(path: Path) -> str:
    """返回文件内容的 SHA-256 十六进制字符串。"""

    # 一次读取文件的原始字节，确保文本编码和换行方式不会改变哈希结果。
    file_bytes = path.read_bytes()

    # digest 是 64 个十六进制字符，可用于确认缓存文件内容是否完全一致。
    digest = hashlib.sha256(file_bytes).hexdigest()

    # 返回内容指纹；相同字节一定得到相同结果。
    return digest


# 输入是冻结注册表路径及可选的全局期望数量；输出是一井一行、按井号排序的注册表。
def load_and_validate_registry(
    registry_path: Path,
    expected_wells: int | None = 773,
    expected_rows: int | None = 3_783_989,
) -> pd.DataFrame:
    """读取固定注册表，并检查井、行数和 pad 不跨折。"""

    # registry 形状为 [井数量, 注册表列数]；井号和 pad 强制按字符串读取，避免前导零丢失。
    registry = pd.read_csv(registry_path, dtype={"well_id": str, "pad_id": str})

    # required_columns 是后续定位文件、分 fold 和核对隐藏行数所需的最小字段集合。
    required_columns = {"well_id", "pad_id", "fold", "hidden_rows"}

    # missing_columns 保存注册表实际没有提供的字段名。
    missing_columns = required_columns - set(registry.columns)

    # 缺列时立即停止，防止后续把不完整注册表解释成有效 fold。
    if missing_columns:
        raise ValueError(f"fold 注册表缺少列：{sorted(missing_columns)}")

    # 一口井必须只出现一次，否则它可能被重复训练或分到多个 fold。
    if registry["well_id"].duplicated().any():
        raise ValueError("fold 注册表含重复 well_id")

    # pad_fold_counts 的每个值表示同一个 pad 实际出现了多少个不同 fold。
    pad_fold_counts = registry.groupby("pad_id")["fold"].nunique()

    # 同一 pad 跨 fold 会让空间相邻井同时进入训练和验证，因此必须拒绝。
    if int(pad_fold_counts.max()) != 1:
        raise ValueError("同一个 pad 出现在多个 fold")

    # 默认要求与冻结基线完全相同的 773 口井；小测试可传 None 关闭这一项。
    if expected_wells is not None and len(registry) != expected_wells:
        raise ValueError("fold 注册表井数不匹配")

    # total_hidden_rows 是注册表承诺的全部自然隐藏评价行数。
    total_hidden_rows = int(registry["hidden_rows"].sum())

    # 默认要求冻结基线的评价行总数一致；小测试可传 None 关闭这一项。
    if expected_rows is not None and total_hidden_rows != expected_rows:
        raise ValueError("fold 注册表隐藏行数不匹配")

    # 排序并重建连续索引，使下游组装顺序不受 CSV 原始行序影响。
    sorted_registry = registry.sort_values("well_id").reset_index(drop=True)

    # 返回已经通过完整性与防泄漏约束检查的注册表。
    return sorted_registry


# 输入是一井一行的注册表和逐井 CSV 目录；输出是所有自然隐藏行拼接后的特征表。
def build_feature_table(
    registry_df: pd.DataFrame,
    train_dir: Path,
    progress_interval: int = 50,
) -> pd.DataFrame:
    """逐井构造特征；返回顺序固定为 well_id、row_index。"""

    # well_tables 中每个元素形状为 [该井隐藏行数, 特征及元数据列数]。
    well_tables: list[pd.DataFrame] = []

    # total_wells 是本次需要读取和组装的井数量，仅用于进度提示。
    total_wells = len(registry_df)

    # 按注册表逐井处理；well_number 从 1 开始，便于人类阅读运行进度。
    for well_number, registry_row in enumerate(
        registry_df.itertuples(index=False), start=1
    ):
        # well_path 指向当前井的原始水平井 CSV，不会读取注册表之外的井。
        well_path = train_dir / f"{registry_row.well_id}__horizontal_well.csv"

        # 缺少任意注册井文件时立即停止，避免静默生成不完整缓存。
        if not well_path.is_file():
            raise FileNotFoundError(f"缺少水平井文件：{well_path}")

        # horizontal_df 形状为 [当前井总采样行数, 原始字段数]，同时包含可见前缀和隐藏后缀。
        horizontal_df = pd.read_csv(well_path)

        # well_rows 只包含当前井的自然隐藏后缀；特征公式由 Task 1 的唯一接口负责。
        well_rows = build_simple_lgbm_rows(
            horizontal_df,
            str(registry_row.well_id),
            int(registry_row.fold),
        )

        # actual_hidden_rows 是依据 TVT_input 实际构造出的隐藏行数量。
        actual_hidden_rows = len(well_rows)

        # 实际行数必须与冻结注册表一致，防止数据文件或 fold 清单版本错配。
        if actual_hidden_rows != int(registry_row.hidden_rows):
            raise ValueError(
                f"井 {registry_row.well_id} 的隐藏行数与注册表不一致"
            )

        # 保存当前井表，待全部井校验通过后一次性纵向拼接。
        well_tables.append(well_rows)

        # progress_interval 为 0 时关闭提示；否则按固定间隔并在最后一口井打印。
        should_print_progress = progress_interval > 0 and (
            well_number % progress_interval == 0 or well_number == total_wells
        )

        # 进度输出包含已完成井数和总井数，不改变返回数据。
        if should_print_progress:
            print(f"特征进度：{well_number}/{total_wells} 井", flush=True)

    # result 形状为 [全部隐藏行数, 特征及元数据列数]，先按注册表处理顺序拼接。
    result = pd.concat(well_tables, ignore_index=True)

    # 最终排序消除注册表输入顺序差异，使相同井文件总能生成相同行序。
    sorted_result = result.sort_values(["well_id", "row_index"]).reset_index(drop=True)

    # 返回可直接用于固定 fold 训练和后续缓存写出的完整行级表。
    return sorted_result
