import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from src.analysis.experiment_types import ExperimentConfig, ExperimentResult



# ------------------------------------------------------------------
# Support functions
# ------------------------------------------------------------------
def _apply_block_feature_selection(
    X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    """Applies PCA per feature block and filters columns based on the feature set.

    Parameters
    ----------
    X : np.ndarray
        The complete feature matrix of shape (n_samples, n_features).
    train_idx : np.ndarray
        Array of indices corresponding to the training split.
    test_idx : np.ndarray
        Array of indices corresponding to the testing/validation split.
    feature_set : str
        The target feature subset to extract. Must be one of 'all',
        'spectral', or 'acoustic'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - X_tr : Formatted and transformed training feature matrix.
        - X_te : Formatted and transformed testing feature matrix.

    Raises
    ------
    ValueError
        If `feature_set` is not one of 'all', 'spectral', or 'acoustic'.
    """
    # 1. Quick exit for acoustic-only features (no PCA required)
    if feature_set == "acoustic":
        X_tr = X[train_idx, 24:27]  # Jitter, Shimmer, HNR
        X_te = X[test_idx, 24:27]
        return X_tr, X_te

    if feature_set not in ["all", "spectral"]:
        raise ValueError(
            "feature_set must be 'all', 'spectral' or 'acoustic'"
        )

    # 2. Define block slicing boundaries and target PCA components
    blocks = {
        "mfbm_mean1-6": (0, 6, 2),
        "mfbm_mean7-12": (6, 12, 1),
        "mfbm_std1-6": (12, 18, 2),
        "mfbm_std7-12": (18, 24, 1),
    }

    Xtr_out = []
    Xte_out = []

    # 3. Apply PCA processing per spectral block
    for start, end, n_comp in blocks.values():
        pca = PCA(n_components=n_comp)
        Xtr_block = pca.fit_transform(X[train_idx, start:end])
        Xte_block = pca.transform(X[test_idx, start:end])

        Xtr_out.append(Xtr_block)
        Xte_out.append(Xte_block)

    # 4. Append acoustic features if the 'all' set is selected
    if feature_set == "all":
        Xtr_out.append(X[train_idx, 24:])  # Jitter, Shimmer, HNR
        Xte_out.append(X[test_idx, 24:])

    return np.hstack(Xtr_out), np.hstack(Xte_out)


import numpy as np


def _derive_ova_accuracy(cm: np.ndarray) -> np.ndarray:
    """Derive One-vs-All (OvA) accuracy for each class from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        A square confusion matrix of shape (n_classes, n_classes) where
        rows represent true labels and columns represent predicted labels.

    Returns
    -------
    np.ndarray
        A 1D array of shape (n_classes,) containing the One-vs-All accuracy
        score for each respective class.

    Raises
    ------
    ValueError
        If the input matrix `cm` is not square.
    """
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("The confusion matrix must be square.")

    total_samples = np.sum(cm)
    n_classes = cm.shape[0]
    ova_accuracies = []

    for i in range(n_classes):
        # True Positives: diagonal element for the current class
        tp = cm[i, i]

        # False Negatives: sum of the row minus True Positives
        fn = np.sum(cm[i, :]) - tp

        # False Positives: sum of the column minus True Positives
        fp = np.sum(cm[:, i]) - tp

        # True Negatives: everything else in the matrix
        tn = total_samples - tp - fn - fp

        # Calculate OvA accuracy for the current class
        class_accuracy = (tp + tn) / total_samples
        ova_accuracies.append(class_accuracy)

    return np.array(ova_accuracies)


# ------------------------------------------------------------------
# Pipeline Core
# ------------------------------------------------------------------
import numpy as np


def run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
    print_report: bool = False,
) -> ExperimentResult:
    """Runs a repeated stratified k-fold cross-validation experiment using

    configurations.

    This function handles the complete pipeline for a single experiment task,
    including splitting the data, applying block-based feature selection (PCA),
    training the classifier, evaluating performance, and aggregating confusion
    matrices and metrics across all folds and repetitions.

    Parameters
    ----------
    X : np.ndarray
        The complete feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        The target labels array of shape (n_samples,).
    config : ExperimentConfig
        An instance containing configuration parameters such as the number of
        iterations, selected feature set, and classifier hyperparameters.
    print_report : bool, default=False
        If True, prints a detailed classification report and metrics summary
        to the console after completion.

    Returns
    -------
    ExperimentResult
        An object containing the aggregated evaluation metrics, raw predictions,
        and final confusion matrices for the cross-validation run.
    """
    
    acc = np.zeros(config.num_iters)
    pre = np.zeros(config.num_iters)
    rec = np.zeros(config.num_iters)
    f1s = np.zeros(config.num_iters)
    aucs = np.zeros(config.num_iters)

    n_classes = len(np.unique(y))
    ova_per_iter = np.zeros((config.num_iters, n_classes))

    conf_total = None
    conf_norm_total = None

    for ii in range(config.num_iters):
        cv = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=ii
        )

        y_true_all = []
        y_pred_all = []
        y_score_all = []

        for train_idx, test_idx in cv.split(X, y):
        
            # Feature selection via blocks
            X_tr, X_te = _apply_block_feature_selection(
                X, 
                train_idx, 
                test_idx, 
                config.feature_set
            )

            # Scaling
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_te = scaler.transform(X_te)

            # Model Execution
            clf = SVC(kernel='rbf').fit(X_tr, y[train_idx])
            y_pred = clf.predict(X_te)
            
            y_true_all.append(y[test_idx])
            y_pred_all.append(y_pred)

            if n_classes == 2:
                y_score_all.append(clf.decision_function(X_te))

        y_true_all = np.hstack(y_true_all)
        y_pred_all = np.hstack(y_pred_all)

        conf = confusion_matrix(y_true_all, y_pred_all)
        acc[ii] = np.mean(y_true_all == y_pred_all)

        # Hardcoded logic for binary metrics tracking safely
        if n_classes == 2:
            pre[ii] = precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
            rec[ii] = recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
            f1s[ii] = f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0)
            
            y_score_all = np.hstack(y_score_all)
            fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
            aucs[ii] = auc(fpr, tpr)
        else:
            pre[ii], rec[ii], f1s[ii], aucs[ii] = 0.0, 0.0, 0.0, np.nan

        ova_per_iter[ii] = _derive_ova_accuracy(conf)

        if conf_total is None:
            conf_total = conf
            conf_norm_total = confusion_matrix(y_true_all, y_pred_all, normalize="true")
        else:
            conf_total += conf
            conf_norm_total += confusion_matrix(y_true_all, y_pred_all, normalize="true")

    # Construct the strictly typed output
    result = ExperimentResult(
        config=config,
        accuracy_mean=float(acc.mean()),
        accuracy_std=float(acc.std()),
        precision_mean=float(pre.mean()),
        precision_std=float(pre.std()),
        recall_mean=float(rec.mean()),
        recall_std=float(rec.std()),
        f1_mean=float(f1s.mean()),
        f1_std=float(f1s.std()),
        auc_mean=float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else 0.0,
        auc_std=float(np.nanstd(aucs)) if not np.all(np.isnan(aucs)) else 0.0,
        confusion_matrix=conf_total / config.num_iters,
        confusion_matrix_norm=conf_norm_total / config.num_iters,
        ova_mean=np.mean(ova_per_iter, axis=0),
        ova_std=np.std(ova_per_iter, axis=0)
    )

    if print_report:
        print(f"\nTask      : {config.task_name} | Features: {config.feature_set}")
        print(f"Accuracy  : {result.accuracy_mean:.4f} ± {result.accuracy_std:.4f}")

    return result