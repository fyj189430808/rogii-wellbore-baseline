# P3 Baseline, Shadow Holdout and D00 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for each production-code task. The user handles Git; do not create worktrees, commits, pushes, or PRs.

**Goal:** Freeze P2-P02 as P3B00, create a deterministic label-free 116-well shadow holdout, and run the P3-D00 low-dimensional OOF residual audit on development wells only.

**Architecture:** Keep P2 artifacts immutable and create pointer-and-hash lineage for P3B00. Build shadow metadata from raw legal columns, then filter P2-P02 OOF by `well_id` before exposing target columns. Fit each development well independently with polynomial and piecewise-linear bases and aggregate SSE without saving row-level oracle features.

**Tech Stack:** Python 3.11, pandas, NumPy, PyArrow, pytest, JSON/CSV/Parquet.

## Global Constraints

- CV is exactly `balanced_well_5fold_v1`, 773 wells and 3,783,989 natural-hidden rows.
- Baseline is exactly `P2_P02_multiscale_pf_paths_v1`: 41 features, one LightGBM, 1,734 trees, seed 29, micro RMSE 10.305704992073148.
- Do not modify or overwrite any phase-2 artifact.
- Shadow selection must not read TVT, surfaces, residuals, OOF scores, or model errors.
- P3-D00 must not return or fit any shadow-well target row.
- Oracle outputs are diagnostics only and must never enter a formal feature cache.
- Do not run PF01, tune LightGBM, or change the target in this plan.

---

### Task 1: Freeze the P3 contract and P3B00 lineage

**Files:**
- Create: `rogii_clean/configs/p3_b00_group5_p2p02_v1.json`
- Create: `rogii_clean/scripts/freeze_p3_b00.py`
- Create: `rogii_clean/tests/test_freeze_p3_b00.py`
- Create at runtime: `rogii_clean/artifacts/P3B00_group5_p2p02_v1/`

**Interfaces:**
- Consumes: P2-P02 `config.json`, `metrics.json`, `feature_list.json`, `predictions.parquet`.
- Produces: `freeze_p3_baseline(clean_root: Path, output_dir: Path) -> dict[str, object]`.

- [ ] Write a failing test that builds four tiny source files and asserts that the freeze result records source paths, SHA-256 values, 41 features, five exact fold RMSE values and `10.305704992073148` without copying predictions.
- [ ] Run `python -m pytest rogii_clean/tests/test_freeze_p3_b00.py -q`; expect failure because `freeze_p3_b00` does not exist.
- [ ] Implement strict source validation and write `config.json`, `metrics.json`, `feature_list.json`, `lineage.json`, and `conclusion.md` to the new artifact directory.
- [ ] Re-run the focused test; expect all assertions to pass.

### Task 2: Build the label-free P3 shadow holdout

**Files:**
- Create: `rogii_clean/src/p3_shadow_holdout.py`
- Create: `rogii_clean/scripts/build_p3_shadow_holdout.py`
- Create: `rogii_clean/configs/p3_shadow_holdout_v1.json`
- Create: `rogii_clean/tests/test_p3_shadow_holdout.py`
- Create at runtime: `rogii_clean/artifacts/P3_shadow_holdout_v1/`

**Interfaces:**
- Produces: `collect_legal_well_metadata(raw_train_dir, registry) -> pd.DataFrame`.
- Produces: `select_balanced_shadow(metadata, number_of_shadow_wells, salt) -> pd.DataFrame`.

- [ ] Write failing tests for deterministic selection, exact count, fold coverage, stable Typewell fingerprinting, and rejection of forbidden columns matching `TVT`, `target`, six surfaces, `residual`, `error`, `rmse`, or `oracle`.
- [ ] Run `python -m pytest rogii_clean/tests/test_p3_shadow_holdout.py -q`; expect failure because the module is missing.
- [ ] Implement legal metadata extraction, label-free bins, deterministic SHA-256 tie breaking and marginal-balance greedy selection.
- [ ] Re-run the focused test; expect all assertions to pass.
- [ ] Implement the runner to save `shadow_holdout.csv`, `metadata.csv`, `balance_report.json`, `config.json`, `runtime.json`, and `conclusion.md`.

### Task 3: Implement low-dimensional residual bases

**Files:**
- Create: `rogii_clean/src/p3_d00_residual_structure.py`
- Create: `rogii_clean/tests/test_p3_d00_residual_structure.py`

**Interfaces:**
- Produces: `normalized_progress(md: np.ndarray) -> np.ndarray`.
- Produces: `fit_polynomial_residual(t, residual, degree, anchored) -> ResidualFit`.
- Produces: `fit_control_point_residual(t, residual, number_of_points, anchored) -> ResidualFit`.
- Produces: `audit_well_residuals(well_frame: pd.DataFrame) -> dict[str, object]`.

- [ ] Write failing synthetic tests proving exact recovery for constant/linear/quadratic/cubic curves, anchored start behavior, control-basis partition of unity, finite output for a one-row well, and monotonically non-increasing SSE as an unanchored basis gains dimensions.
- [ ] Run `python -m pytest rogii_clean/tests/test_p3_d00_residual_structure.py -q`; expect failure because the module is missing.
- [ ] Implement NumPy least-squares fits with explicit basis matrices and named coefficients.
- [ ] Re-run the focused test; expect all assertions to pass.

### Task 4: Build and run the P3-D00 audit

**Files:**
- Create: `rogii_clean/configs/p3_d00_residual_structure_v1.json`
- Create: `rogii_clean/experiments/P3_D00_residual_structure_v1_card.md`
- Create: `rogii_clean/scripts/diagnose_p3_d00_residual_structure.py`
- Create: `rogii_clean/tests/test_diagnose_p3_d00_residual_structure.py`
- Create at runtime: `rogii_clean/artifacts/P3_D00_residual_structure_v1/`

**Interfaces:**
- Consumes: P3B00 lineage, shadow CSV and P2-P02 OOF.
- Produces: `load_development_predictions(...) -> pd.DataFrame` with a hard disjointness assertion.
- Produces: `run_audit(predictions, source_metrics) -> tuple[pd.DataFrame, pd.DataFrame, dict]`.

- [ ] Write a failing test ensuring shadow wells are absent before target use, duplicate `(well_id,row_index)` keys fail, and aggregate pooled RMSE equals `sqrt(sum(SSE)/sum(n))` rather than the mean well RMSE.
- [ ] Run `python -m pytest rogii_clean/tests/test_diagnose_p3_d00_residual_structure.py -q`; expect failure because the runner is missing.
- [ ] Implement filtered PyArrow loading, per-well fits, per-fold and overall aggregation, SSE shares, decision text, runtime and conclusion output.
- [ ] Re-run all four focused test files; expect zero failures.
- [ ] Run the baseline freeze and shadow builder, then run P3-D00 once on all non-shadow development wells.
- [ ] Verify artifact row counts, well counts, source hashes, shadow disjointness, and JSON parsing.

### Task 5: Register the phase transition and result

**Files:**
- Modify: `AGENTS.md`
- Create: `rogii_clean/experiments/phase3_roadmap.md`
- Modify: `rogii_clean/experiments/current_state.json`
- Modify: `rogii_clean/experiments/registry.jsonl`
- Create: `docs/三阶段突破路线与实验执行说明.md`

**Interfaces:**
- Current state points to the actual last completed artifact and names `P3-D01` as next only after P3-D00 succeeds.

- [ ] Save the user-provided three-stage route verbatim in `docs/三阶段突破路线与实验执行说明.md`.
- [ ] Replace the phase-2 AGENTS contract with the phase-3 fixed baseline, A/B boundary, unified gates, shadow rule and `P3-D00 → P3-D01 → P3-PF01` immediate order.
- [ ] Append one registry record for P3B00, one for shadow creation and one for P3-D00 without editing earlier lines.
- [ ] Update `current_state.json` using actual measured P3-D00 outputs; do not pre-write results.
- [ ] Run `python -m pytest rogii_clean/tests/test_freeze_p3_b00.py rogii_clean/tests/test_p3_shadow_holdout.py rogii_clean/tests/test_p3_d00_residual_structure.py rogii_clean/tests/test_diagnose_p3_d00_residual_structure.py -q` and parse every newly written JSON/JSONL file.

