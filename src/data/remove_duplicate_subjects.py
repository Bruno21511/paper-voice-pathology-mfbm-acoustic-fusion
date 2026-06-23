# -*- coding: utf-8 -*-
import logging
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

def remove_duplicate_subjects(
    df: pd.DataFrame, 
    duplicate_subjects: list
    ) -> pd.DataFrame:
    """
    Remove known duplicate subject entries from the corpus.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'file' and 'group' columns.
    duplicate_subjects : list of dict
        Each dict has keys 'file' (str) and 'groups' (list of str), 
        specifying which (file, group) combinations to remove.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    df_out = df.copy()
    for entry in duplicate_subjects:
        filename = entry["file"]
        groups = entry["groups"]
        mask = (df_out["file"] == filename) & (df_out["group"].isin(groups))
        n_removed = mask.sum()
        df_out = df_out[~mask]
        logger.info(f"Removed {n_removed} duplicate entries for {filename} (groups: {groups})")
    return df_out