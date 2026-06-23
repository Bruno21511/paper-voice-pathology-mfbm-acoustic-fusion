import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple

def aggregate_band_statistics_per_class(
    df: pd.DataFrame,
    classes: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Aggregate mean and std MFBM statistics per class.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'group', 'mean_MFBM', and 'std_MFBM' 
        (as produced by compute_band_statistics).
    classes : list of str or None, optional
        List of classes to aggregate. If None, uses unique values in 
        df['group'].

    Returns
    -------
    mean_dict : dict
        Mapping from class label to mean MFBM per band, shape (n_bands,).
    std_dict : dict
        Mapping from class label to mean of per-sample std MFBM per 
        band, shape (n_bands,).
    """
    if classes is None:
        classes = sorted(df['group'].unique())

    mean_dict = {}
    std_dict = {}
    for c in classes:
        subset = df[df['group'] == c]
        mean_stack = np.vstack(subset['mean_MFBM'].values)
        std_stack = np.vstack(subset['std_MFBM'].values)
        mean_dict[c] = np.mean(mean_stack, axis=0)
        std_dict[c] = np.mean(std_stack, axis=0)

    return mean_dict, std_dict