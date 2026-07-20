"""P3-CFGR-D01：先落盘无标签 GR 候选评分，再独立读取隐藏 TVT 评价。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

CLEAN_ROOT = Path(__file__).resolve().parents[1]
if str(CLEAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_ROOT))

from src.p3_cfgr_d01_observed_block_scoring import (  # noqa: E402
    assign_hidden_blocks, block_candidate_costs, candidate_ranks, deterministic_hidden_gr_roll,
)
from src.p3_mdp01_dynamic_mode_path import fit_prefix_calibration  # noqa: E402

EXPERIMENT_ID = "P3-CFGR-D01_observed_block_candidate_scoring"
CORE_NAMES = ["P2", "low", "middle", "high"]
SHIFT_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
CACHE_VERSION = "cfgr_d01_v2_shared_sigma_rank_tie_safe"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shadow_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    flag = frame.get("is_shadow", pd.Series(True, index=frame.index))
    return set(frame.loc[flag.astype(str).str.lower().isin(["true", "1", "yes"]), "well_id"].astype(str))


def load_development_registry(config: dict, mode: str) -> pd.DataFrame:
    registry = pd.read_csv(CLEAN_ROOT / config["fold_registry"])
    shadow = _shadow_ids(CLEAN_ROOT / config["shadow_registry"])
    registry = registry.loc[~registry["well_id"].astype(str).isin(shadow), ["well_id", "fold"]].drop_duplicates()
    registry["well_id"] = registry["well_id"].astype(str)
    if mode == "smoke":
        registry = registry.sort_values(["fold", "well_id"]).head(3)
    if len(registry) == 0 or set(registry["well_id"]).intersection(shadow):
        raise ValueError("开发井筛选失败或影子井泄漏")
    return registry.reset_index(drop=True)


def assert_complete_mode_cache(config: dict) -> None:
    """正式 core4 的模式路径必须来自完整 657 井 PFM02 合法缓存。"""
    development = load_development_registry(config, "all")
    directory = CLEAN_ROOT / config["mode_cache_dir"]
    missing = [well for well in development.well_id if not (directory / f"{well}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"PFM02 mode cache 不完整，缺少 {len(missing)} 口开发井，例如 {missing[:3]}")


def _read_horizontal(raw_dir: Path, well_id: str, include_target: bool = False) -> pd.DataFrame:
    columns = ["MD", "GR", "TVT_input"] + (["TVT"] if include_target else [])
    return pd.read_csv(raw_dir / f"{well_id}__horizontal_well.csv", usecols=columns)


def _read_typewell(raw_dir: Path, well_id: str) -> pd.DataFrame:
    return pd.read_csv(raw_dir / f"{well_id}__typewell.csv", usecols=["TVT", "GR"])


def load_p2_legal_predictions(config: dict) -> dict[tuple[str, int], pd.DataFrame]:
    """每个阶段一次性读取 P2 的合法列，永久不读取 target_tvt。"""
    frame = pd.read_parquet(CLEAN_ROOT / config["p2_predictions"], columns=["well_id", "fold", "row_index", "md", "pred_tvt"])
    frame.well_id = frame.well_id.astype(str)
    return {(well, int(fold)): group.sort_values("row_index").reset_index(drop=True) for (well, fold), group in frame.groupby(["well_id", "fold"], sort=False)}


def _load_candidates(config: dict, well_id: str, fold: int, p2_by_well: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    # 明确排除 P2 预测中的 target_tvt，法律评分阶段绝不接触它。
    p2 = p2_by_well.get((well_id, fold))
    if p2 is None: raise ValueError(f"P2 合法预测缺少 {well_id}/{fold}")
    modes = pd.read_parquet(CLEAN_ROOT / config["mode_cache_dir"] / f"{well_id}.parquet")
    modes = modes.sort_values("row_index")
    key = ["row_index"]
    table = p2.merge(modes[key + ["last_visible_tvt", "pf_mode_low_delta", "pf_mode_middle_delta", "pf_mode_high_delta"]], on=key, validate="one_to_one")
    if len(table) != len(p2) or not np.array_equal(table.row_index.to_numpy(), p2.row_index.to_numpy()):
        raise ValueError(f"{well_id} P2/PFM01 行键不严格对齐")
    core = np.column_stack([table.pred_tvt, table.last_visible_tvt + table.pf_mode_low_delta,
                            table.last_visible_tvt + table.pf_mode_middle_delta, table.last_visible_tvt + table.pf_mode_high_delta])
    seed = np.load(CLEAN_ROOT / config["seed_cache_dir"] / f"{well_id}.npz")
    if not np.array_equal(seed["row_index"], table.row_index.to_numpy()):
        raise ValueError(f"{well_id} seed 路径行键不严格对齐")
    all132 = np.column_stack([core, float(seed["last_tvt"][0]) + np.asarray(seed["seed_delta"], dtype=float).T])
    return table, all132, CORE_NAMES + [f"seed_{i:03d}" for i in range(128)]


def _expected_gr(typewell: pd.DataFrame, candidates: np.ndarray, slope: float, intercept: float) -> tuple[np.ndarray, np.ndarray]:
    reference = typewell.dropna().sort_values("TVT")
    tvt = reference.TVT.to_numpy(float); gr = reference.GR.to_numpy(float)
    outside = (candidates < tvt[0]) | (candidates > tvt[-1])
    # np.interp 仅为数值计算便利；越界行被明确加罚，绝不作为真正 Typewell 观测外推。
    result = slope * np.interp(candidates, tvt, gr) + intercept
    return result, outside


def _pool_scores(observed_gr: np.ndarray, expected: np.ndarray, outside: np.ndarray, blocks: np.ndarray, common_sigma: float, config: dict) -> dict[str, list[float] | int | float]:
    result: dict[str, list[float] | int | float] = {}
    for parity, selector in {"all": np.ones(len(blocks), bool), "odd": blocks % 2 == 1, "even": blocks % 2 == 0}.items():
        costs, valid, counts = block_candidate_costs(observed_gr[selector], expected[selector], blocks[selector],
            minimum_observed_points=int(config["minimum_observed_gr_points"]), common_sigma=common_sigma,
            out_of_range=outside[selector], out_of_range_penalty=float(config["out_of_range_penalty"]))
        result[f"score_{parity}"] = costs.tolist(); result[f"blocks_{parity}"] = int(len(valid)); result[f"observed_{parity}"] = int(np.sum(counts))
    return result


def legal_score_well(config: dict, well_id: str, fold: int, donor_id: str, p2_by_well: dict[tuple[str, int], pd.DataFrame], fingerprint: str) -> dict:
    raw_dir = (CLEAN_ROOT / config["raw_train_dir"]).resolve()
    horizontal = _read_horizontal(raw_dir, well_id, include_target=False)
    typewell = _read_typewell(raw_dir, well_id)
    donor = _read_typewell(raw_dir, donor_id)
    table, all132, names = _load_candidates(config, well_id, fold, p2_by_well)
    hidden = horizontal.TVT_input.isna().to_numpy()
    # P2 行索引是自然隐藏行索引；严格验证其与当前原始井的隐藏位置相同。
    hidden_index = np.flatnonzero(hidden)
    if not np.array_equal(table.row_index.to_numpy(), hidden_index):
        raise ValueError(f"{well_id} 原始井隐藏行与 P2 row_index 不一致")
    calibration = fit_prefix_calibration(horizontal, typewell)
    donor_calibration = fit_prefix_calibration(horizontal, donor)
    observed = horizontal.loc[hidden, "GR"].to_numpy(float)
    blocks = assign_hidden_blocks(table.md.to_numpy(float), min_hidden_md=float(table.md.min()))
    payload: dict = {"cache_version": CACHE_VERSION, "fingerprint": fingerprint, "well_id": well_id, "fold": int(fold), "donor_id": donor_id, "candidate_names": names,
                     "rows": int(len(table)), "raw_observed_gr_rows": int(np.isfinite(observed).sum()), "calibration": calibration.__dict__, "pools": {}}
    controls: dict[str, np.ndarray] = {"normal": observed}
    controls.update({f"shift_{fraction:.2f}": deterministic_hidden_gr_roll(observed, fraction) for fraction in SHIFT_FRACTIONS})
    for pool_name, candidates in {"core4": all132[:, :4], "all132": all132}.items():
        normal_gr, outside = _expected_gr(typewell, candidates, calibration.slope, calibration.intercept)
        donor_gr, donor_outside = _expected_gr(donor, candidates, donor_calibration.slope, donor_calibration.intercept)
        pool: dict = {"out_of_range_rate": float(outside.mean()), "scores": {}}
        # 在成本后显式加越界比例惩罚（每块等权平均已由核心处理）。
        for label, source_gr in controls.items():
            scores = _pool_scores(source_gr, normal_gr, outside, blocks, calibration.sigma_level, config)
            pool["scores"][label] = scores
        # 错配模板也允许使用目标井可见前缀对该 donor 的合法最佳标定及其共同尺度。
        mismatch = _pool_scores(observed, donor_gr, donor_outside, blocks, donor_calibration.sigma_level, config)
        pool["scores"]["cross_mismatch"] = mismatch; pool["out_of_range_rate"] = float(outside.mean())
        payload["pools"][pool_name] = pool
    return payload


def _fingerprint(config: dict) -> str:
    code = Path(__file__).read_bytes() + (CLEAN_ROOT / "src/p3_cfgr_d01_observed_block_scoring.py").read_bytes()
    return hashlib.sha256(CACHE_VERSION.encode() + json.dumps(config, sort_keys=True).encode() + code).hexdigest()


def legal_cache_dir(artifact: Path, scope: str) -> Path:
    """smoke 与 all 的 donor 图不同，必须物理隔离缓存。"""
    if scope not in {"smoke", "all"}: raise ValueError("非法 legal cache scope")
    return artifact / "legal_scores" / scope


def donor_map(registry: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for _, group in registry.sort_values(["fold", "well_id"]).groupby("fold"):
        ids = group.well_id.tolist()
        result.update({target: ids[(index + 1) % len(ids)] for index, target in enumerate(ids)})
    if any(key == value for key, value in result.items()): raise ValueError("跨井错配 donor 不得等于目标井")
    return result


def run_legal(config: dict, mode: str, artifact: Path, force: bool = False) -> None:
    assert_complete_mode_cache(config)
    registry = load_development_registry(config, mode)
    artifact.mkdir(parents=True, exist_ok=True); legal_dir = legal_cache_dir(artifact, mode); legal_dir.mkdir(parents=True, exist_ok=True)
    donors = donor_map(registry)
    fingerprint = _fingerprint(config); p2_by_well = load_p2_legal_predictions(config)
    for index, row in enumerate(registry.itertuples(index=False), 1):
        path = legal_dir / f"{row.well_id}.json"
        if path.exists() and not force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("cache_version") == CACHE_VERSION: continue
        _json(path, legal_score_well(config, row.well_id, int(row.fold), donors[row.well_id], p2_by_well, fingerprint))
        if index % 10 == 0: print(f"CFGR legal 已完成 {index}/{len(registry)} 井", flush=True)
    _json(artifact / "runtime.json", {"experiment_id": EXPERIMENT_ID, "mode": mode, "wells": len(registry), "hidden_tvt_read": False, "legal_cache_scope": mode, "legal_cache_complete": True, "resume": "fingerprint_validated_per_well_json", "fingerprint": fingerprint})


def _rank_metrics(score: np.ndarray, truth: np.ndarray) -> dict:
    score = np.asarray(score, float); truth = np.asarray(truth, float)
    usable = np.isfinite(score) & np.isfinite(truth)
    if usable.sum() == 0:
        empty = np.full(score.size, np.nan)
        return {"spearman": float("nan"), "top1_hit": float("nan"), "top3_contains_true_best": float("nan"), "top3_jaccard": float("nan"), "score_rank": empty, "truth_rank": empty.copy()}
    score_rank = np.full(score.size, np.nan); truth_rank = np.full(truth.size, np.nan)
    score_rank[usable] = rankdata(score[usable], method="average"); truth_rank[usable] = rankdata(truth[usable], method="average")
    score_order = np.flatnonzero(usable)[np.argsort(score[usable], kind="mergesort")]
    truth_order = np.flatnonzero(usable)[np.argsort(truth[usable], kind="mergesort")]
    correlation = float("nan") if (np.unique(score_rank[usable]).size < 2 or np.unique(truth_rank[usable]).size < 2) else float(spearmanr(score_rank[usable], truth_rank[usable]).statistic)
    return {"spearman": correlation,
            "top1_hit": bool(score_order[0] == truth_order[0]),
            "top3_contains_true_best": bool(truth_order[0] in score_order[:3]),
            "top3_jaccard": float(len(set(score_order[:3]) & set(truth_order[:3])) / len(set(score_order[:3]) | set(truth_order[:3]))),
            "score_rank": score_rank, "truth_rank": truth_rank}


def _bootstrap_ci(values: np.ndarray, *, seed: int = 42, draws: int = 2000) -> list[float]:
    """逐井重采样的 normal-minus-control 95% CI。"""
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if values.size == 0: return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sampled = np.array([rng.choice(values, size=values.size, replace=True).mean() for _ in range(draws)])
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


def build_control_contrasts(frame: pd.DataFrame, metric_names: list[str]) -> dict:
    """normal 减去逐井五 shift 平均、及 normal 减 cross；绝不混合 shift 行。"""
    output: dict = {}
    normal = frame.loc[frame.control == "normal"].set_index("well_id")
    shifts = frame.loc[frame.control.str.startswith("shift_")].groupby("well_id")[metric_names].mean()
    cross = frame.loc[frame.control == "cross_mismatch"].set_index("well_id")
    for label, comparison in {"normal_minus_shift_mean": shifts, "normal_minus_cross_mismatch": cross}.items():
        details = {}
        for metric in metric_names:
            delta = (normal[metric].astype(float) - comparison[metric].astype(float)).dropna()
            details[metric] = {"mean_delta": float(delta.mean()) if len(delta) else float("nan"), "paired_well_bootstrap_ci95": _bootstrap_ci(delta.to_numpy()), "valid_wells": int(len(delta))}
        output[label] = details
    return output


def evaluate(config: dict, mode: str, artifact: Path, *, smoke: bool = False) -> None:
    assert_complete_mode_cache(config)
    registry = load_development_registry(config, mode); legal_dir = legal_cache_dir(artifact, mode); fingerprint = _fingerprint(config); donors = donor_map(registry)
    # 所有身份、缓存版本、配置/代码指纹及 scope donor 均须在读取真值之前通过。
    for row in registry.itertuples(index=False):
        path = legal_dir / f"{row.well_id}.json"
        if not path.exists(): raise RuntimeError("必须先完成当前 scope 全部 legal_scores 才能读取隐藏 TVT")
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (cached.get("cache_version") != CACHE_VERSION or cached.get("fingerprint") != fingerprint or cached.get("well_id") != row.well_id or int(cached.get("fold", -1)) != int(row.fold) or cached.get("donor_id") != donors[row.well_id]):
            raise RuntimeError(f"legal score 合同/血缘不匹配：{row.well_id}")
    raw_dir = (CLEAN_ROOT / config["raw_train_dir"]).resolve(); records: list[dict] = []; candidate_records: list[dict] = []; p2_by_well = load_p2_legal_predictions(config)
    for well_number, row in enumerate(registry.itertuples(index=False), 1):
        legal = json.loads((legal_dir / f"{row.well_id}.json").read_text(encoding="utf-8"))
        horizontal = _read_horizontal(raw_dir, row.well_id, include_target=True)  # 仅此评价阶段读取目标。
        truth = horizontal.loc[horizontal.TVT_input.isna(), "TVT"].to_numpy(float)
        candidate_table, candidates, _ = _load_candidates(config, row.well_id, int(row.fold), p2_by_well)
        hidden_md = candidate_table.md.to_numpy(float)
        block_ids = assign_hidden_blocks(hidden_md, min_hidden_md=float(hidden_md.min()))
        rmse = np.sqrt(np.nanmean((candidates - truth[:, None]) ** 2, axis=0))
        for pool, width in {"core4": 4, "all132": 132}.items():
            for label, scores in legal["pools"][pool]["scores"].items():
                metrics = _rank_metrics(np.asarray(scores["score_all"]), rmse[:width])
                score_ranks = metrics.pop("score_rank")
                truth_ranks = metrics.pop("truth_rank")
                odd = np.asarray(scores["score_odd"]); even = np.asarray(scores["score_even"])
                pair = _rank_metrics(odd, even)
                metrics.update({"odd_even_rank_spearman": pair["spearman"], "odd_even_top1_same": pair["top1_hit"], "odd_even_top3_jaccard": pair["top3_jaccard"]})
                odd_truth = np.sqrt(np.nanmean((candidates[block_ids % 2 == 1, :width] - truth[block_ids % 2 == 1, None]) ** 2, axis=0))
                even_truth = np.sqrt(np.nanmean((candidates[block_ids % 2 == 0, :width] - truth[block_ids % 2 == 0, None]) ** 2, axis=0))
                metrics["odd_score_even_truth_spearman"] = float(spearmanr(candidate_ranks(odd), candidate_ranks(even_truth)).statistic)
                metrics["even_score_odd_truth_spearman"] = float(spearmanr(candidate_ranks(even), candidate_ranks(odd_truth)).statistic)
                records.append({"well_id": row.well_id, "fold": row.fold, "pool": pool, "control": label, **metrics,
                                **{key: scores[key] for key in ("blocks_all", "blocks_odd", "blocks_even", "observed_all", "observed_odd", "observed_even")}, "out_of_range_rate": legal["pools"][pool]["out_of_range_rate"]})
                for index, name in enumerate(legal["candidate_names"][:width]):
                    candidate_records.append({"well_id": row.well_id, "fold": row.fold, "pool": pool, "control": label,
                        "candidate": name, "score_all": scores["score_all"][index], "score_odd": odd[index], "score_even": even[index], "true_rmse": rmse[index], "score_rank": score_ranks[index], "truth_rank": truth_ranks[index]})
        if well_number % 10 == 0: print(f"CFGR evaluate 已完成 {well_number}/{len(registry)} 井", flush=True)
    frame = pd.DataFrame(records); suffix = "smoke_" if smoke else ""; frame.to_csv(artifact / f"{suffix}per_well.csv", index=False)
    pd.DataFrame(candidate_records).to_parquet(artifact / f"{suffix}per_candidate.parquet", index=False)
    metric_names = ["spearman", "top1_hit", "top3_contains_true_best", "odd_even_rank_spearman", "odd_even_top1_same", "odd_even_top3_jaccard"]
    summary = []
    for (pool, control), group in frame.groupby(["pool", "control"]):
        item = {"pool": pool, "control": control, "scorable_wells": int(group.spearman.notna().sum())}
        for name in metric_names:
            values = pd.to_numeric(group[name], errors="coerce").astype(float).dropna(); item[name] = {"median": float(values.median()) if len(values) else float("nan"), "q25": float(values.quantile(.25)) if len(values) else float("nan"), "q75": float(values.quantile(.75)) if len(values) else float("nan"), "mean": float(values.mean()) if len(values) else float("nan")}
        # pooled within-well ranks：拼接每井已处理并列的候选 rank。
        candidates = pd.DataFrame(candidate_records); subset = candidates[(candidates.pool == pool) & (candidates.control == control)].dropna(subset=["score_rank", "truth_rank"])
        item["pooled_within_well_rank_spearman"] = float(spearmanr(subset.score_rank, subset.truth_rank).statistic) if len(subset) else float("nan")
        coverage = {}
        for part in ("all", "odd", "even"):
            coverage[part] = {}
            for prefix in ("blocks", "observed"):
                values = group[f"{prefix}_{part}"].dropna(); coverage[part][prefix] = {"median": float(values.median()), "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75)), "zero_block_share": float((values == 0).mean()) if prefix == "blocks" else None}
        coverage["out_of_range_rate"] = float(group.out_of_range_rate.median()); item["coverage"] = coverage
        summary.append(item)
    contrasts = {pool: build_control_contrasts(group, metric_names) for pool, group in frame.groupby("pool")}
    metrics = {"experiment_id": EXPERIMENT_ID, "wells": len(registry), "hidden_tvt_read_in_evaluation_only": True, "summary": summary, "paired_control_contrasts": contrasts}
    _json(artifact / ("smoke_metrics.json" if smoke else "metrics.json"), metrics)
    _json(artifact / "runtime.json", {"experiment_id": EXPERIMENT_ID, "mode": "smoke-evaluate" if smoke else "evaluate", "wells": len(registry), "hidden_tvt_read": True, "hidden_tvt_read_stage": "evaluation_after_legal_cache_validation"})
    if not smoke: (artifact / "conclusion.md").write_text("# P3-CFGR-D01 结论\n\n数据直接证明的事实：见 metrics.json；本诊断不训练模型，也不把 hidden TVT 用作特征。\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["smoke", "all", "evaluate", "smoke-evaluate"], required=True); parser.add_argument("--force", action="store_true"); parser.add_argument("--config", default=str(CLEAN_ROOT / "configs/p3_cfgr_d01_observed_block_candidate_scoring_v1.json")); args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8")); artifact = CLEAN_ROOT / config["artifact_dir"]
    _json(artifact / "config.json", config)
    if args.mode in {"smoke", "all"}: run_legal(config, args.mode, artifact, args.force)
    if args.mode == "evaluate": evaluate(config, "all", artifact)
    if args.mode == "smoke-evaluate": evaluate(config, "smoke", artifact, smoke=True)

if __name__ == "__main__": main()
