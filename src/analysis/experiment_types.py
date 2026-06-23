# src/analysis/experiment_types.py
"""
Typed data structures representing configurations and results of
classification experiments.

These dataclasses replace loose dictionaries, ensuring that each
experiment always contains the expected fields (failing early if
something is missing). They also serve as a common base that can be
extended via inheritance in future repositories with additional needs
(e.g., PCA variants, cross corpus comparisons).
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ExperimentConfig:
    """
    Configuration for a single classification experiment.

    Parameters
    ----------
    task_name : str
        Identifier for the classification task
        (e.g., "Control_vs_PhLP").
    groups : list of str or None
        Groups/classes included in this task. None indicates that all
        dataset classes are used (multi‑class classification).
    feature_set : str
        Feature set to use (e.g., "acoustic", "spectral", "all").
    n_splits : int, optional
        Number of folds in stratified cross‑validation (default: 5).
    num_iters : int, optional
        Number of cross‑validation repetitions (each repetition uses a
        different random_state), default 100.
    """
    task_name: str
    groups: Optional[List[str]]
    feature_set: str
    n_splits: int = 5
    num_iters: int = 100


@dataclass
class ExperimentResult:
    """
    Aggregated result of a classification experiment after all
    cross validation repetitions.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration that produced this result — maintains traceability
        between results and the parameters used.
    accuracy_mean : float
        Mean accuracy across repetitions.
    accuracy_std : float
        Standard deviation of accuracy across repetitions.
    precision : float
        Mean precision (positive class).
    recall : float
        Mean recall (positive class).
    f1 : float
        Mean F1‑score (positive class).
    auc : float
        Mean ROC AUC across repetitions. Set to 0.0 for multi‑class
        problems where AUC is not computed.
    confusion_matrix : np.ndarray
        Confusion matrix averaged across repetitions (non‑normalised).
    confusion_matrix_norm : np.ndarray
        Confusion matrix normalised per row (true label), averaged
        across repetitions.
    ova_mean : np.ndarray
        Mean One‑vs‑All accuracy per class, shape (n_classes,).
    ova_std : np.ndarray
        Standard deviation of OvA accuracy per class, shape (n_classes,).
    """
    config: ExperimentConfig
    accuracy_mean: float
    accuracy_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    auc_mean: float
    auc_std: float
    confusion_matrix: np.ndarray
    confusion_matrix_norm: np.ndarray
    ova_mean: np.ndarray
    ova_std: np.ndarray
