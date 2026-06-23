# src/analysis/flatten_results.py
"""
Conversion from a list of ExperimentResult objects (rich structured
results, one per experiment) into two complementary tidy output DataFrames.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from src.analysis.experiment_types import ExperimentResult


def flatten_results(
    results: List[ExperimentResult]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flatten a list of ExperimentResult objects into two tidy DataFrames:
    one for scalar metrics and another for confusion matrices.

    Parameters
    ----------
    results : list of ExperimentResult
        Results from all experiments in the grid, as produced by
        run_experiment_grid.

    Returns
    -------
    tidy_metrics_df : pd.DataFrame
        Long-format metrics table. One row per (task, features, metric),
        with columns 'task', 'features', 'metric', 'value', 'std'.
    tidy_cm_df : pd.DataFrame
        Long-format confusion matrices table. One row per matrix cell,
        with columns 'task', 'features', 'true_index', 'pred_index', 'count', 'norm_value'.
    """
    all_metric_rows = []
    all_cm_rows = []

    for res in results:
        task = res.config.task_name
        features = res.config.feature_set
        
        # Base dictionary to avoid repetition
        base = {
            "task": task,
            "features": features
        }

        # --- 1. Flatten Confusion Matrices dynamically (Supports 2x2 and 3x3)
        cm = np.array(res.confusion_matrix)
        cm_norm = np.array(res.confusion_matrix_norm)
        n_classes = cm.shape[0]

        for i in range(n_classes):
            for j in range(n_classes):
                all_cm_rows.append({
                    **base,
                    "true_index": i,
                    "pred_index": j,
                    "count": cm[i, j],
                    "norm_value": cm_norm[i, j]
                })

        # --- 2. Flatten Scalar Metrics
        rows = [
            {**base, "metric": "accuracy", "value": res.accuracy_mean, "std": res.accuracy_std},
            {**base, "metric": "precision", "value": res.precision_mean, "std": res.precision_std},
            {**base, "metric": "recall", "value": res.recall_mean, "std": res.recall_std},
            {**base, "metric": "f1", "value": res.f1_mean, "std": res.f1_std},
            {**base, "metric": "auc", "value": res.auc_mean, "std": res.auc_std},
        ]

        # One-vs-All accuracy rows
        for idx, (mean, std) in enumerate(zip(res.ova_mean, res.ova_std)):
            rows.append({**base, "metric": f"ova_class_{idx}", "value": mean, "std": std})

        all_metric_rows.extend(rows)

    # Convert both lists to DataFrames
    tidy_metrics_df = pd.DataFrame(all_metric_rows)
    tidy_cm_df = pd.DataFrame(all_cm_rows)

    return tidy_metrics_df, tidy_cm_df
