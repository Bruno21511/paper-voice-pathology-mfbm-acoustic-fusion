# -*- coding: utf-8 -*-
import numpy as np

def build_X(df):
    """
    Builds feature matrix X aligned with article setup.

    Uses only first 12 MFBM bands (out of 20), discarding last 8.
    """

    # --------------------------
    # Convert to arrays
    # --------------------------
    mean_mfbm = np.vstack(df["mean_MFBM"].values)[:, :12]
    std_mfbm  = np.vstack(df["std_MFBM"].values)[:, :12]

    # --------------------------
    # Scalar features
    # --------------------------
    jitter = df["localJitter"].values.reshape(-1, 1)
    shimmer = df["localShimmer"].values.reshape(-1, 1)
    hnr = df["HNR"].values.reshape(-1, 1)

    # --------------------------
    # Final X
    # --------------------------
    X = np.hstack([
        mean_mfbm,   # (N, 12)
        std_mfbm,    # (N, 12)
        jitter,
        shimmer,
        hnr
    ])

    return X