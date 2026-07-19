# P2-D01 PF-Centered GR Audit Implementation Plan

> **For agentic workers:** Use test-driven development. Work on `codex/phase2-experiments`; do not create worktrees or commits because the user keeps Git operations separate from experiments.

**Goal:** Produce a legal PF-centered multiscale GR score landscape and a separately attached hidden-truth identifiability audit without training a model.

**Architecture:** A pure scorer receives legal arrays and returns block × scale × offset scores. A separate oracle helper attaches truth-derived offset/rank columns. A thin resumable runner loads each well and frozen PF cache, writes per-well results, then aggregates positive, real, and circular-shift controls.

**Tech Stack:** Python, NumPy, pandas, Parquet, pytest.

## Global Constraints

- No LightGBM training in P2-D01.
- Legal scorer cannot accept hidden TVT.
- Use frozen `pf_ancc_delta`; do not rerun or tune PF.
- Fixed offsets, block widths, smoothing scales and controls come only from the design spec.
- Oracle and legal outputs remain distinguishable and cannot be used as a formal feature cache.

### Task 1: Core block scorer

- Create `rogii_clean/src/p2_d01_pf_centered_gr.py`.
- Test in `rogii_clean/tests/test_p2_d01_pf_centered_gr.py` with synthetic Typewell GR and a known +8 ft shift.
- RED: require best ensemble offset within 2 ft of +8 and a shifted-GR control to score worse.
- GREEN: implement physical-width smoothing, stable block IDs, grouped Pearson NCC, peak/second peak, soft/scale summaries and support counts.
- Verify invalid MD, key length mismatch and fewer than 30 pairs are handled explicitly.

### Task 2: Oracle isolation and summaries

- Add an oracle helper whose signature receives an already-built legal block table plus separate true-TVT/PF-center arrays.
- Test that changing true TVT changes only oracle columns while legal scores remain byte-for-byte equal.
- Add nearest-grid true rank, range coverage, top-5 hit, best absolute error and true-offset IQR.
- Add positive-control and circular-shift summary functions.

### Task 3: Resumable runner

- Create `rogii_clean/configs/p2_d01_pf_centered_multiscale_gr_v1.json`.
- Create `rogii_clean/scripts/diagnose_p2_d01_pf_centered_gr.py` with `smoke` and `all` modes.
- Load per-well PF cache by `(well_id,row_index)`; reconstruct PF center from the last visible TVT.
- Read legal horizontal columns separately from oracle TVT.
- Save one per-well Parquet plus runtime row so `all` can resume.
- Aggregate real, negative and prefix-positive results; join only the final per-well diagnostic with P2B00 `per_well.csv`.

### Task 4: Run and conclude

- Run the focused tests.
- Run fixed 3-well smoke and inspect recovered offsets, support and runtime.
- Run all 773 wells only if smoke produces finite, non-empty score landscapes.
- Evaluate the five preregistered evidence conditions without changing thresholds.
- Save conclusion, append registry and set `current_state.next_experiment` to `P2-S01` regardless of whether P2-F01 evidence passes.

