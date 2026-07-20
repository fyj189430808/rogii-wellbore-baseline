import numpy as np


def test_build_up03_candidate_applies_fixed_pfs_correction_to_up01_path():
    from src.p3_up03_up01_plus_pfs_correction import build_up03_candidate

    up01 = np.array([100.0, 110.0])
    p3b00 = np.array([90.0, 120.0])
    pfs_lag1000 = np.array([110.0, 100.0])

    result = build_up03_candidate(up01, p3b00, pfs_lag1000, correction_fraction=0.25)

    np.testing.assert_allclose(result, [105.0, 105.0])


def test_confirmation_gate_uses_arithmetic_mean_fold_improvement_and_worst_fold():
    from src.p3_up03_up01_plus_pfs_correction import evaluate_confirmation_gate

    result = evaluate_confirmation_gate(
        fold_improvements_ft=[0.31, -0.01],
        minimum_mean_improvement_ft=0.15,
        maximum_any_fold_degradation_ft=0.15,
    )

    assert result["arithmetic_mean_fold_improvement_ft"] == 0.15
    assert result["maximum_fold_degradation_ft"] == 0.01
    assert result["passed"] is True


def test_conclusion_keeps_confirmation_metrics_separate_from_full_summary():
    from scripts.run_p3_up03_up01_plus_pfs_correction import conclusion

    confirmation = {
        "up01_pooled_rmse": 11.0,
        "candidate_pooled_rmse": 10.5,
        "pooled_improvement_ft": 0.5,
        "fold_improvements_ft": [0.4, 0.6],
    }
    full = {
        "folds": [0, 1, 2, 3, 4],
        "up01_pooled_rmse": 10.0,
        "candidate_pooled_rmse": 9.8,
        "pooled_improvement_ft": 0.2,
        "fold_improvements_ft": [0.1, 0.1, 0.1, 0.4, 0.6],
        "confirmation_folds": [3, 4],
        "confirmation_gate": {
            "arithmetic_mean_fold_improvement_ft": 0.5,
            "passed": True,
        },
        "independent_confirmation_metrics": confirmation,
    }

    text = conclusion(full)

    assert "确认折 UP01 RMSE 为 11.000000 ft" in text
    assert "完整开发集 UP01 RMSE 为 10.000000 ft" in text
