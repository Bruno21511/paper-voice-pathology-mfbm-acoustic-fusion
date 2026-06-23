# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional


def plot_acoustic_features(
    df: pd.DataFrame,
    features: list[tuple[str, str]],
    class_col: str = "group",
    log_transform: tuple[str, ...] = (),
    palette: str | list = "Set2",
    figsize: tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Plot boxplots of acoustic features per class.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing acoustic features and class labels.

    features : list[tuple[str, str]]
        Features to plot as:
            [(column_name, display_name)]

    class_col : str, optional
        Column containing class labels.

    log_transform : tuple[str, ...], optional
        Features converted to logarithmic scale before visualization.
        Uses 20*log10 transformation.

    palette : str or list, optional
        Seaborn color palette.

    figsize : tuple[int, int], optional
        Figure size.

    save_path : str or None, optional
        Output path for saving the figure.

    dpi : int, optional
        Figure resolution.

    Returns
    -------
    None
    """

    # -----------------------------------------
    # Validate columns
    # -----------------------------------------

    required_columns = (
        [class_col] +
        [feature[0] for feature in features]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in dataframe: {missing}"
        )


    # -----------------------------------------
    # Prepare data
    # -----------------------------------------

    df_plot = df.copy()


    for feature, _ in features:

        if feature in log_transform:

            df_plot[feature] = (
                20 *
                np.log10(
                    df_plot[feature] + 1e-10
                )
            )


    # -----------------------------------------
    # Plot
    # -----------------------------------------

    with sns.axes_style("whitegrid"):

        fig, axes = plt.subplots(
            1,
            len(features),
            figsize=figsize
        )


    if len(features) == 1:
        axes = [axes]


    for ax, (feature, title) in zip(
        axes,
        features
    ):

        sns.boxplot(
            data=df_plot,
            x=class_col,
            y=feature,
            hue=class_col,
            palette=palette,
            dodge=False,
            legend=False,
            ax=ax
        )


        ax.set_title(title)
        ax.set_xlabel("Class")


    plt.tight_layout()


    if save_path:

        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight"
        )


    plt.show()

    plt.close(fig)