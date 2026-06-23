# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_accuracy_bars(tidy_df: pd.DataFrame, save_path=None, dpi=300):
    """
    Plot grouped bar chart of accuracy across tasks and feature sets,
    using the tidy long-format dataframe.

    Parameters
    ----------
    tidy_df : pd.DataFrame
        Long-format metrics table with columns:
        ['task', 'features', 'metric', 'value', 'std']
    save_path : str or None
        If provided, save the figure.
    dpi : int
        Resolution for saving.
    """

    # ---------------------------------------------------------
    # Pretty names
    # ---------------------------------------------------------
    feature_map = {
        "acoustic": "Acoustic",
        "spectral": "Spectral",
        "all":      "Combined"
    }
    task_map = {
        "Control_vs_PhLP": "Control vs. PhLP",
        "Control_vs_UVFP": "Control vs. UVFP",
        "PhLP_vs_UVFP":    "PhLP vs. UVFP",
        "All_classes":     "3-Class"
    }

    # ---------------------------------------------------------
    # Filter accuracy rows only
    # ---------------------------------------------------------
    df_acc = tidy_df[tidy_df["metric"] == "accuracy"].copy()

    # Convert to %
    df_acc["acc"] = df_acc["value"] * 100
    df_acc["std"] = df_acc["std"] * 100

    # Pivot to shape (features × tasks)
    acc_matrix = df_acc.pivot(index="features", columns="task", values="acc")
    std_matrix = df_acc.pivot(index="features", columns="task", values="std")

    # Rename pretty labels
    acc_matrix = acc_matrix.rename(index=feature_map, columns=task_map)
    std_matrix = std_matrix.rename(index=feature_map, columns=task_map)
    
    # ---------------------------------------------------------
    # Force feature order
    # ---------------------------------------------------------
    ordem_desejada = ["Acoustic", "Spectral", "Combined"]
    acc_matrix = acc_matrix.reindex(ordem_desejada)
    std_matrix = std_matrix.reindex(ordem_desejada)
    
    # Force group order in X axis
    ordem_colunas = ["Control vs. PhLP", "Control vs. UVFP", "PhLP vs. UVFP", "3-Class"]
    acc_matrix = acc_matrix.reindex(columns=ordem_colunas)
    std_matrix = std_matrix.reindex(columns=ordem_colunas)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    features = acc_matrix.index.tolist()
    tasks = acc_matrix.columns.tolist()

    x = np.arange(len(tasks))
    width = 0.2

    colors  = ["darkorange", "lightcyan", "darkgreen"]
    hatches = ["//", "\\\\", "oo"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, feat in enumerate(features):
        pos = x + (i - 1) * width
        ax.bar(
            pos,
            acc_matrix.loc[feat],
            width,
            yerr=std_matrix.loc[feat],
            capsize=5,
            color=colors[i],
            edgecolor="black",
            hatch=hatches[i],
            label=feat
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("Classification Performance")

    ax.grid(which="major", linewidth=0.6, axis="y", color="black")
    ax.grid(which="minor", linewidth=0.3, axis="y", color="black")
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
