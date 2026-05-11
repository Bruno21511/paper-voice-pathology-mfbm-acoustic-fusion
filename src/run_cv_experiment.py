# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _apply_block_pca(X, train_idx, test_idx):
    """
    Applies PCA per feature block, replicating original paper logic
    using explicit structure.
    """

    # --------------------------
    # define blocks explicitly
    # --------------------------
    blocks = {
        "mfbm_mean1-6": (0, 6, 2),
        "mfbm_mean7-12": (6, 12, 1),
        "mfbm_std1-6": (0, 6, 2),
        "mfbm_std7-12": (6, 12, 1),
    }

    Xtr_out = []
    Xte_out = []

    # --------------------------
    # PCA blocks
    # --------------------------
    for start, end, n_comp in blocks.values():

        pca = PCA(n_components=n_comp)

        Xtr_block = pca.fit_transform(X[train_idx, start:end])
        Xte_block = pca.transform(X[test_idx, start:end])

        Xtr_out.append(Xtr_block)
        Xte_out.append(Xte_block)

    # --------------------------
    # append non-PCA features
    # --------------------------
    Xtr_out.append(X[train_idx, 24:])  # jitter, shimmer, HNR
    Xte_out.append(X[test_idx, 24:])

    # --------------------------
    # final concat
    # --------------------------
    Xtr_final = np.hstack(Xtr_out)
    Xte_final = np.hstack(Xte_out)

    return Xtr_final, Xte_final


def _ova_from_cm(cm):
    """
    Derive One-vs-All (OvA) accuracy for each class from a multiclass confusion matrix.

    For each class i, all other classes are merged into a single negative class,
    and accuracy is computed as (TP + TN) / total, where:
        - TP: correct predictions for class i (diagonal element cm[i, i])
        - FN: class i instances misclassified as another class (row sum minus TP)
        - FP: other classes misclassified as class i (column sum minus TP)
        - TN: all remaining correctly handled instances

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
        Confusion matrix where cm[i, j] is the number of instances
        of true class i predicted as class j.

    Returns
    -------
    ova : list of float, length n_classes
        OvA accuracy for each class, in the same order as the confusion matrix.
    """

    total = np.sum(cm)
    n_classes = cm.shape[0]
    ova = []

    for i in range(n_classes):
        tp = cm[i, i]                    # correctly classified as class i
        fn = np.sum(cm[i, :]) - tp       # class i instances missed
        fp = np.sum(cm[:, i]) - tp       # other classes wrongly predicted as i
        tn = total - tp - fn - fp        # everything else, correctly not class i

        ova.append((tp + tn) / total)

    return ova


def run_cv_experiment(X, y, feature_set="all", print_report=False, num_iters=1000, n_splits=5, kernel="rbf"):
    """
    Run a repeated stratified k-fold cross-validation experiment with an SVM classifier.

    For each iteration, a fresh StratifiedKFold split is generated using the iteration
    index as the random state, ensuring distinct fold assignments across iterations.
    Features are selected and transformed inside the fold loop to prevent data leakage:
    PCA, scaling, and the classifier are all fitted on the training fold only and then
    applied to the test fold.

    Precision, recall, F1, and AUC are only meaningful for binary classification tasks.
    In multiclass settings these values will be unreliable and should be ignored;
    use accuracy and the OvA metrics derived from the confusion matrix instead.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 27)
        Feature matrix. Columns 0–23 are spectral features (mean and std of 12 Mel
        filterbank bands, prior to PCA); columns 24–26 are acoustic features
        (Jitter, Shimmer, HNR).
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    feature_set : str, optional
        Which features to use. One of:
            - "all"      : spectral (after block PCA) + acoustic (default)
            - "spectral" : spectral only (after block PCA); acoustic columns discarded
            - "acoustic" : acoustic only (Jitter, Shimmer, HNR); no PCA applied
    print_report : bool, optional
        If True, prints a summary of all metrics and confusion matrices to stdout
        after all iterations are complete (default: False).
    num_iters : int, optional
        Number of cross-validation repetitions (default: 1000).
    n_splits : int, optional
        Number of folds in each stratified k-fold split (default: 5).
    kernel : str, optional
        SVM kernel passed to sklearn.svm.SVC (default: "rbf").

    Returns
    -------
    dict
        Summary metrics averaged across all iterations:
            - accuracy_mean / accuracy_std : mean and std of per-iteration accuracy
            - precision, recall, f1        : binary classification only (see note above)
            - auc                          : ROC AUC, binary classification only
            - ova_class_{i}_mean / _std    : One-vs-All accuracy mean and std per class,
                                             derived from the per-iteration confusion matrix
    """

    # per-iteration metric arrays
    acc  = np.zeros(num_iters)
    pre  = np.zeros(num_iters)
    rec  = np.zeros(num_iters)
    f1   = np.zeros(num_iters)
    aucs = np.zeros(num_iters)

    n_classes = len(np.unique(y))
    ova_per_iter = np.zeros((num_iters, n_classes))  # OvA accuracy per iteration per class

    conf_total      = None  # accumulated raw confusion matrix (for reporting)
    conf_norm_total = None  # accumulated normalised confusion matrix (for reporting)

    for ii in range(num_iters):

        # unique random state per iteration ensures distinct fold assignments
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ii)

        y_true_all  = []
        y_pred_all  = []
        y_score_all = []  # decision function scores, binary case only

        for train_idx, test_idx in cv.split(X, y):

            # ------------------------------------------------------------------
            # Feature selection and PCA — fitted on training fold only
            # ------------------------------------------------------------------
            if feature_set == "all":
                # apply block PCA to spectral features, then append acoustic
                X_tr, X_te = _apply_block_pca(X, train_idx, test_idx)

            elif feature_set == "spectral":
                # apply block PCA to spectral features, then discard acoustic columns
                X_tr, X_te = _apply_block_pca(X, train_idx, test_idx)
                X_tr = X_tr[:, :-3]  # drop Jitter, Shimmer, HNR
                X_te = X_te[:, :-3]

            elif feature_set == "acoustic":
                # use raw acoustic features only — no PCA needed
                X_tr = X[train_idx, 24:27]  # columns: Jitter, Shimmer, HNR
                X_te = X[test_idx,  24:27]

            else:
                raise ValueError("feature_set must be 'all', 'spectral' or 'acoustic'")

            # ------------------------------------------------------------------
            # Scaling — fitted on training fold, applied to both
            # ------------------------------------------------------------------
            scaler = StandardScaler().fit(X_tr)
            X_tr   = scaler.transform(X_tr)
            X_te   = scaler.transform(X_te)

            # ------------------------------------------------------------------
            # SVM — fitted on training fold, evaluated on test fold
            # ------------------------------------------------------------------
            clf    = SVC(kernel=kernel).fit(X_tr, y[train_idx])
            y_pred = clf.predict(X_te)
            y_true = y[test_idx]

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)

            # decision function scores needed for ROC AUC (binary only)
            if n_classes == 2:
                y_score_all.append(clf.decision_function(X_te))

        # ------------------------------------------------------------------
        # Aggregate predictions across folds for this iteration
        # ------------------------------------------------------------------
        y_true_all = np.hstack(y_true_all)
        y_pred_all = np.hstack(y_pred_all)

        conf = confusion_matrix(y_true_all, y_pred_all)

        acc[ii] = np.mean(y_true_all == y_pred_all)

        # precision, recall, F1 — hardcoded for binary; unreliable in multiclass
        pre[ii] = conf[0, 0] / (conf[0, 0] + conf[1, 0] + 1e-10)
        rec[ii] = conf[0, 0] / (conf[0, 0] + conf[0, 1] + 1e-10)
        f1[ii]  = 2 * pre[ii] * rec[ii] / (pre[ii] + rec[ii] + 1e-10)

        # OvA accuracy per class for this iteration
        ova_per_iter[ii] = _ova_from_cm(conf)

        # accumulate confusion matrices for final reporting
        if conf_total is None:
            conf_total      = conf
            conf_norm_total = confusion_matrix(y_true_all, y_pred_all, normalize="true")
        else:
            conf_total      += conf
            conf_norm_total += confusion_matrix(y_true_all, y_pred_all, normalize="true")

        # ROC AUC — binary case only
        if n_classes == 2:
            y_score_all = np.hstack(y_score_all)
            fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
            aucs[ii]    = auc(fpr, tpr)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    if print_report:
        print("Accuracy :", np.mean(acc))
        print("Std Acc  :", np.std(acc))
        print("Precision:", np.mean(pre))
        print("Recall   :", np.mean(rec))
        print("F1       :", np.mean(f1))
        print("AUC      :", np.mean(aucs))
        print("\nConfusion Matrix (average over iterations)\n",            conf_total      / num_iters)
        print("\nNormalised Confusion Matrix (average over iterations)\n", conf_norm_total / num_iters)
        print()

    # OvA mean and std per class, derived from per-iteration values
    ova_mean = np.mean(ova_per_iter, axis=0)
    ova_std  = np.std(ova_per_iter,  axis=0)

    ova_mean_dict = {f"ova_class_{i}_mean": ova_mean[i] for i in range(n_classes)}
    ova_std_dict  = {f"ova_class_{i}_std":  ova_std[i]  for i in range(n_classes)}

    return {
        "accuracy_mean": np.mean(acc),
        "accuracy_std":  np.std(acc),
        "precision":     np.mean(pre),
        "recall":        np.mean(rec),
        "f1":            np.mean(f1),
        "auc":           np.mean(aucs),
        
        "confusion_matrix": conf_total / num_iters,
        "confusion_matrix_norm": conf_norm_total / num_iters,
    
        **dict(sorted(ova_mean_dict.items())),
        **dict(sorted(ova_std_dict.items())),
    }
