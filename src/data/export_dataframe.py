# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def export_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    output_root: str,
    expand_mfbm: bool = True,
    drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Export processed DataFrame to parquet.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    dataset_name : str
        Name used for output file.
    output_root : str
        Base directory to save file (absolute or correctly resolved 
        relative path; e.g. PROJECT_ROOT / "data/processed").
    expand_mfbm : bool, optional
        Whether to split MFBM into per-band columns, by default True.
    drop_columns : list of str or None, optional
        Columns to remove before saving, by default None.

    Returns
    -------
    pd.DataFrame
        The exported dataframe (after expansion and column dropping).
    """
    df_out = df.copy()

    # --- 1. Expand MFBM if requested
    if expand_mfbm and 'mfbm' in df_out.columns:
        n_bands = df_out['mfbm'].iloc[0].shape[0]
        for i in range(n_bands):
            df_out[f'mfbm_{i}'] = df_out['mfbm'].apply(
                lambda x: x[i, :].tolist()
            )

    # --- 2. Drop columns
    if drop_columns is not None:
        df_out = df_out.drop(columns=drop_columns, errors='ignore')

    # --- 3. Reset index
    df_out = df_out.reset_index(drop=True)

    # --- 4. Save
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}.parquet"
    df_out.to_parquet(output_path, index=False)
    logger.info(f"Saved to: {output_path}")

    return df_out