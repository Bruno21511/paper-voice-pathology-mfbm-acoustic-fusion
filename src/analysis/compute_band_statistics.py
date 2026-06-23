import numpy as np
import pandas as pd

def compute_band_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-band mean and standard deviation from MFBM matrices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'MFBM' column with matrices of shape 
        (n_filters, n_frames).

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with two new columns added:
        'mean_MFBM' and 'std_MFBM', each containing arrays of shape 
        (n_filters,).
    """
    df_out = df.copy()
    
    mean_list = []
    std_list = []
    for mfbm in df_out['mfbm']:
        mean_bands = np.mean(mfbm, axis=1)
        std_bands = np.std(mfbm, axis=1)
        mean_list.append(mean_bands)
        std_list.append(std_bands)
    
    df_out['mean_MFBM'] = mean_list
    df_out['std_MFBM'] = std_list
    
    return df_out