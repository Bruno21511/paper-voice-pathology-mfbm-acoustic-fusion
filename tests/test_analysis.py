# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from src.analysis.experiment_types import ExperimentConfig, ExperimentResult
from src.analysis.run_cv_experiment import (
    _apply_block_feature_selection,
    _derive_ova_accuracy,
    run_cv_experiment,
)
from src.analysis.run_experiment_grid import run_experiment_grid


# =====================================================================
# 1. GRID RUNNER TESTS (ORCHESTRATION & COMBINATORICS)
# =====================================================================
@patch("src.analysis.run_experiment_grid.build_X")
@patch("src.analysis.run_experiment_grid.run_cv_experiment")
def test_run_experiment_grid_combinatorics(mock_run_cv, mock_build_X):
    """Test if the grid orchestrator executes the exact number of combinations

    and filters data correctly without triggering real ML or feature engineering.
    """
    # 1. Setup mock data with the structure expected by the pipeline
    mock_array = np.zeros(20)
    df_mock = pd.DataFrame(
        {
            "group": ["control", "pathology", "control"],
            "mean_MFBM": [mock_array, mock_array, mock_array],
            "std_MFBM": [mock_array, mock_array, mock_array],
        }
    )

    tasks_mock = {
        "Control_vs_Pathology": ["control", "pathology"],
        "All_classes": None,  # None means no filtering, use all classes
    }
    feature_sets_mock = ["acoustic", "spectral", "all"]

    # 2. Configure Mock behaviors
    mock_build_X.return_value = np.array([[1], [2], [3]])

    # Mock run_cv_experiment to return a dummy result object
    mock_result_obj = MagicMock()
    mock_run_cv.return_value = mock_result_obj

    # 3. Execute the Orchestrator Grid
    results = run_experiment_grid(
        df=df_mock,
        tasks=tasks_mock,
        feature_sets=feature_sets_mock,
        num_iters=10,
    )

    # 4. Assertions
    # 2 tasks x 3 feature sets = exactly 6 cross-validation runs
    assert mock_run_cv.call_count == 6
    # build_X should be called once per task (2 tasks total)
    assert mock_build_X.call_count == 2

    # Verify that the output is a list containing all 6 results
    assert isinstance(results, list)
    assert len(results) == 6


# =====================================================================
# 2. SUPPORT FUNCTIONS TESTS (FEATURE SELECTION & METRICS)
# =====================================================================
def test_apply_block_feature_selection_acoustic():
    """Ensure the acoustic subset extracts only the correct column indices."""
    # Create a dummy matrix with 30 columns
    X = np.arange(30).reshape(1, 30)
    train_idx = np.array([0])
    test_idx = np.array([0])

    X_tr, X_te = _apply_block_feature_selection(
        X, train_idx, test_idx, "acoustic"
    )

    # Expected columns are 24, 25, 26 -> [24, 25, 26]
    np.testing.assert_array_equal(X_tr, [[24, 25, 26]])
    np.testing.assert_array_equal(X_te, [[24, 25, 26]])


def test_apply_block_feature_selection_invalid():
    """Ensure a ValueError is raised when an invalid feature set key is provided."""
    X = np.zeros((2, 30))
    with pytest.raises(ValueError, match="feature_set must be"):
        _apply_block_feature_selection(X, np.array([0]), np.array([1]), "bad_key")


def test_derive_ova_accuracy_square_check():
    """Ensure the function validates that the confusion matrix is square."""
    bad_cm = np.zeros((2, 3))
    with pytest.raises(ValueError, match="must be square"):
        _derive_ova_accuracy(bad_cm)


def test_derive_ova_accuracy_exact_math():
    """Validate the mathematical correctness of the One-vs-All (OvA) calculation."""
    # Simulated 3x3 confusion matrix
    # Class 0: TP=5, FN=1, FP=1, TN=(18-5-1-1)=11 -> Acc = 16/18 = 0.8888...
    cm = np.array([[5, 1, 0], [1, 4, 2], [0, 0, 5]])

    ova_accs = _derive_ova_accuracy(cm)

    assert len(ova_accs) == 3
    # Total samples = 5+1+0+1+4+2+0+0+5 = 18
    # For Class 0: TP=5, TN=4+2+0+5=11 -> Accuracy = 16/18
    assert pytest.approx(ova_accs[0]) == 16.0 / 18.0


# =====================================================================
# 3. CORE PIPELINE TESTS (CROSS-VALIDATION EXECUTION)
# =====================================================================
@patch("src.analysis.run_cv_experiment._apply_block_feature_selection")
@patch("src.analysis.run_cv_experiment.SVC")
def test_run_cv_experiment_binary_execution(mock_svc_cls, mock_feat_sel):
    """Test the complete binary pipeline execution using ML mocks."""
    # 1. Setup synthetic binary classification data (2 classes)
    X_mock = np.random.rand(10, 27)
    y_mock = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    cfg = ExperimentConfig(
        task_name="Binary_Test",
        groups=["control", "pathology"],
        feature_set="all",
        num_iters=2,  # Run only 2 iterations for a fast unit test
    )
    # Force a static n_splits inside the mock config for deterministic behavior
    cfg.n_splits = 2

    # 2. Configure Mock behaviors
    # Simulate that block feature selection returns a simplified matrix
    mock_feat_sel.return_value = (np.ones((5, 3)), np.ones((5, 3)))

    # Simulate scikit-learn's SVC classifier behavior
    mock_svc_instance = MagicMock()
    mock_svc_cls.return_value = mock_svc_instance
    mock_svc_instance.fit.return_value = mock_svc_instance

    # Force predictions to return classes that simulate a perfectly accurate model (100% acc)
    # Since the loop flattens predictions per fold (5 samples per fold in kfold=2), pass blocks of 5
    mock_svc_instance.predict.side_effect = [
        np.array([0, 0, 0, 1, 1]),  # Iter 0 - Fold 1
        np.array([0, 0, 1, 1, 1]),  # Iter 0 - Fold 2
        np.array([0, 0, 0, 1, 1]),  # Iter 1 - Fold 1
        np.array([0, 0, 1, 1, 1]),  # Iter 1 - Fold 2
    ]

    # Simulated continuous decision function values required for AUC calculations
    mock_svc_instance.decision_function.side_effect = [
        np.array([-1.0, -1.0, -1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, 1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, 1.0, 1.0, 1.0]),
    ]

    # 3. Run the cross-validation experiment pipeline
    result = run_cv_experiment(X_mock, y_mock, config=cfg, print_report=False)

    # 4. Assertions
    assert isinstance(result, ExperimentResult)
    assert result.config == cfg
    assert result.accuracy_mean == 1.0  # Perfect model simulation
    assert result.auc_mean == 1.0

    # Ensure that feature selection and model fitting were triggered the correct amount of times
    # 2 iterations x 2 folds = 4 calls total
    assert mock_feat_sel.call_count == 4
    assert mock_svc_instance.fit.call_count == 4