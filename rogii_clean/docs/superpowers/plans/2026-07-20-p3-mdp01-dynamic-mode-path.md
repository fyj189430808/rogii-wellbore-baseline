# P3-MDP01 Dynamic Mode Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and execute inline; this workspace explicitly forbids Git/worktrees for this task.

**Goal:** Build a target-free four-state dynamic path generator, deterministic negative controls, and a physically separate hidden-target scorer for a three-well smoke.

**Architecture:** `src/p3_mdp01_dynamic_mode_path.py` contains only legal array transforms and DP. `scripts/run_p3_mdp01_dynamic_mode_path.py` reads target-free sources and writes per-well legal caches. `scripts/score_p3_mdp01_dynamic_mode_path.py` refuses to run until all requested legal caches exist, then loads hidden targets only for scoring.

**Tech Stack:** Python, NumPy, pandas, PyArrow, pytest.

## Global Constraints

- Fixed experiment id: `P3_MDP01_dynamic_mode_path_v1`.
- Fixed 250 ft windows, 125 ft centers, states `P2/low/middle/high`.
- Fixed costs, DP penalties, minimum three-block non-P2 run, safe10/safe25, GR roll and block-cost permutation.
- Legal generation must never load horizontal `TVT` or P2 `target_tvt`.
- Only unit tests and three-well smoke; no formal folds.

---

### Task 1: Core legal path engine

**Files:**
- Create: `rogii_clean/tests/test_p3_mdp01_dynamic_mode_path.py`
- Create: `rogii_clean/src/p3_mdp01_dynamic_mode_path.py`

- [ ] Write failing tests for prefix calibration, observed-only costs, P2 all-missing fallback, minimum non-P2 run, jump/second-order penalties, exact global margin, row cross-fade, safe paths and deterministic controls.
- [ ] Run the focused test file and confirm failures are caused by the missing module/API.
- [ ] Implement the minimal core APIs and rerun until green.

### Task 2: Target-free legal runner

**Files:**
- Create: `rogii_clean/tests/test_run_p3_mdp01_dynamic_mode_path.py`
- Create: `rogii_clean/scripts/run_p3_mdp01_dynamic_mode_path.py`
- Create: `rogii_clean/configs/p3_mdp01_dynamic_mode_path_v1.json`

- [ ] Write failing tests proving exact legal input columns, alignment rejection, physical smoke directory separation, cache fingerprint checks and target-reader independence.
- [ ] Run the runner tests and confirm the missing API failure.
- [ ] Implement per-well legal cache generation, atomic writes and checkpoint resume; rerun until green.

### Task 3: Independent scorer and smoke

**Files:**
- Create: `rogii_clean/tests/test_score_p3_mdp01_dynamic_mode_path.py`
- Create: `rogii_clean/scripts/score_p3_mdp01_dynamic_mode_path.py`

- [ ] Write failing tests that scoring refuses incomplete caches and loads targets only after completeness validation.
- [ ] Implement pooled/per-well/per-fold scoring for P2, dynamic, safe10, safe25 and both controls.
- [ ] Run all three focused test files.
- [ ] Generate three target-free legal caches, inspect audit, then run the independent smoke scorer.

