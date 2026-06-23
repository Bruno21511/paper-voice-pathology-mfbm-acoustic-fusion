# src/analysis/run_experiment_grid.py
import pandas as pd
from typing import Dict, List, Optional
from src.analysis.experiment_types import ExperimentConfig, ExperimentResult
from src.analysis.run_cv_experiment import run_cv_experiment
from src.analysis.build_X import build_X  # presumindo que já existe algures

def run_experiment_grid(
    df: pd.DataFrame,
    tasks: Dict[str, Optional[List[str]]],
    feature_sets: List[str],
    num_iters: int = 100,
    print_report: bool = False,
) -> List[ExperimentResult]:
    """Run the full experiment grid (all tasks × all feature sets).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'group' column and features.
    tasks : dict
        Mapping from task name to list of groups (None = all classes).
    feature_sets : list of str
        Feature sets to evaluate, e.g. ['acoustic', 'spectral', 'all'].
    num_iters : int, optional
        Number of CV repetitions per experiment, by default 100.
    print_report : bool, optional
        Whether to print a summary after each experiment.

    Returns
    -------
    list of ExperimentResult
        A list containing the rich structured results for each experiment
        executed in the grid.
    """
    
    all_results = []

    for task_name, groups in tasks.items():
        df_task = df if groups is None else df[df["group"].isin(groups)].copy()
        df_task["class"] = pd.Categorical(df_task["group"]).codes
        X_task = build_X(df_task)
        y_task = df_task["class"].values

        for feat in feature_sets:
            cfg = ExperimentConfig(
                task_name=task_name,
                groups=groups,
                feature_set=feat,
                num_iters=num_iters
            )
            result = run_cv_experiment(X_task, y_task, config=cfg, print_report=print_report)
            all_results.append(result)

    return all_results