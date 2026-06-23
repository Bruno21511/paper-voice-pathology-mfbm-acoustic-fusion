# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict


def plot_mfbm_statistics_per_class(
    mean_dict: Dict[str, np.ndarray],
    std_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot mean and standard deviation of MFBM per class from precomputed aggregates.

    Parameters
    ----------
    mean_dict : dict
        Mapping from class label to mean MFBM per band.

    std_dict : dict
        Mapping from class label to standard deviation MFBM per band.

    save_path : str or None, optional
        If provided, saves the figure to this path.
    """

    classes = list(mean_dict.keys())

    # Generate line styles automatically
    available_styles = ['-', '--', '-.', ':']

    linestyles = {
        cls: available_styles[i % len(available_styles)]
        for i, cls in enumerate(classes)
    }


    n_bands = len(
        mean_dict[classes[0]]
    )

    bands = np.arange(n_bands) + 1


    plt.figure(figsize=(12, 6))


    # -------------------------
    # Mean
    # -------------------------

    plt.subplot(1, 2, 1)

    plt.title(
        "Mean MFBM magnitude per class"
    )

    for cls in classes:

        plt.plot(
            bands,
            mean_dict[cls],
            linestyle=linestyles[cls],
            linewidth=2,
            label=cls
        )

    plt.xlabel("Band")
    plt.ylabel("Mean magnitude")
    plt.grid(True)
    plt.xticks(bands)
    plt.legend()


    # -------------------------
    # Std
    # -------------------------

    plt.subplot(1, 2, 2)

    plt.title(
        "Std MFBM magnitude per class"
    )

    for cls in classes:

        plt.plot(
            bands,
            std_dict[cls],
            linestyle=linestyles[cls],
            linewidth=2,
            label=cls
        )

    plt.xlabel("Band")
    plt.ylabel("Standard deviation magnitude")
    plt.grid(True)
    plt.xticks(bands)
    plt.legend()


    plt.tight_layout()


    if save_path:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )


    plt.show()