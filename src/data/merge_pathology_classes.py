import pandas as pd
from typing import List

def merge_pathology_classes(
    df: pd.DataFrame,
    classes_to_merge: List[str],
    merged_label: str
) -> pd.DataFrame:
    """
    Merge specified pathology classes into a single group label and 
    recompute integer class codes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'group' column with class labels.
    classes_to_merge : list of str
        Group labels to merge (e.g. ['edema', 'nodulo']).
    merged_label : str
        New label to assign to the merged classes (e.g. 'physio').

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with 'group' updated and 'class' 
        recomputed from scratch.
    """
    df_out = df.copy()
    df_out.loc[df_out['group'].isin(classes_to_merge), 'group'] = merged_label
    df_out['class'] = pd.Categorical(df_out['group']).codes
    return df_out