# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_3_confusion_matrices(tidy_cm_df, task_name, class_names, save_path=None):
    """
    Plot the confusion matrices (acoustic, spectral, combined) for a given task,
    using the new tidy_cm_df DataFrame produced by flatten_results().

    Parameters
    ----------
    tidy_cm_df : pd.DataFrame
        Long-format confusion matrices table containing columns:
        'task', 'features', 'true_index', 'pred_index', 'count', 'norm_value'.
    task_name : str
        Name of the task (e.g., "Control_vs_PhLP" or "All_classes").
    class_names : list of str
        Names of the classes for axis labels.
    save_path : str or None
        If provided, saves the figure to this path.
    """

    feature_order = ["acoustic", "spectral", "all"]
    pretty_names = {
        "acoustic": "Acoustic",
        "spectral": "Spectral",
        "all": "Combined"
    }    
    
    # === Fontsize configs ===
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    CELL_SIZE = 12  # square numbers size
    # ==========================================

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    task_df = tidy_cm_df[tidy_cm_df["task"] == task_name]

    for ax, feat in zip(axes, feature_order):
        
        feat_df = task_df[task_df["features"] == feat]

        if feat_df.empty:
            ax.set_title(f"{pretty_names[feat]} (missing)", fontsize=TITLE_SIZE)
            ax.axis("off")
            continue

        cm = feat_df.pivot(index="true_index", columns="pred_index", values="count").values
        cm_norm = feat_df.pivot(index="true_index", columns="pred_index", values="norm_value").values

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        # Subplot and axis adjustment
        ax.set_title(pretty_names[feat], fontsize=TITLE_SIZE)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        
        # (X e Y) classes characters adjustment
        ax.set_xticklabels(class_names, fontsize=TICK_SIZE)
        ax.set_yticklabels(class_names, fontsize=TICK_SIZE)

        # "Predicted" and "True" adjustment
        ax.set_xlabel("Predicted", fontsize=LABEL_SIZE)
        ax.set_ylabel("True", fontsize=LABEL_SIZE)

        # Matriz numbers adjustment
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center",
                        color=color, 
                        fontsize=CELL_SIZE) # <--- Added here

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Main title adjustment
    plt.suptitle(task_name, fontsize=TITLE_SIZE + 4, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()