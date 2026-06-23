# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def import_dataframe(
    dataset_name: str,
    input_root: str,
    rebuild_mfbm: bool = True
) -> pd.DataFrame:
    """
    Load parquet dataset and optionally rebuild MFBM matrices.

    Parameters
    ----------
    dataset_name : str
        Name of dataset file (without extension).
    input_root : str
        Base directory containing the parquet file (absolute or 
        correctly resolved relative path; e.g. PROJECT_ROOT / 
        "data/processed").
    rebuild_mfbm : bool, optional
        If True, reconstruct MFBM (n_bands x n_frames) from expanded 
        per-band columns, by default True.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe, with MFBM reconstructed if requested.
    """
    # --- 1. Load parquet
    input_path = Path(input_root) / f"{dataset_name}.parquet"
    df = pd.read_parquet(input_path)

    # --- 2. Normalize class labels

    if "group" in df.columns:

        df["group"] = (
            df["group"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # --- 3. Rebuild MFBM
    if rebuild_mfbm:
        # find all MFBM columns
        mfbm_cols = [col for col in df.columns if col.startswith("mfbm_")]
        # sort to guarantee order
        mfbm_cols = sorted(mfbm_cols, key=lambda x: int(x.split("_")[1]))

        def rebuild(row):
            bands = [np.array(row[col]) for col in mfbm_cols]
            return np.vstack(bands)

        df['mfbm'] = df.apply(rebuild, axis=1)

        # remove duplicate columns
        df = df.drop(columns=mfbm_cols)

    return df