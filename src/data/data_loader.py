# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import soundfile as sf
import os
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def data_loader(
    dataset_name: str,
    data_root: str
) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Load metadata and audio signals from a dataset.

    Expected structure:
    <data_root>/<dataset_name>/<dataset_name>.csv
    <data_root>/<dataset_name>/<class>/<audio_file>

    CSV format (no header):
    [file, age, gender, class]

    Parameters
    ----------
    dataset_name : str
        Name of the dataset folder (e.g., "myUSP")
    data_root : str
        Root folder containing datasets (absolute path recommended; 
        e.g., CORPORA_ROOT from the calling notebook/script)

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing metadata and loaded audio signals:
        columns = [file, age, gender, group, path, class, signal, fs]
    """

    # -----------------------------
    # 1. Paths
    # -----------------------------
    dataset_path = os.path.join(data_root, dataset_name)
    csv_path = os.path.join(dataset_path, f"{dataset_name}.csv")

    # -----------------------------
    # 2. Load CSV
    # -----------------------------
    columns = ['file', 'age', 'gender', 'group']

    df = pd.read_csv(
        csv_path,
        delimiter=';',
        header=None,
        names=columns
    )

    # -----------------------------
    # 3. Build file paths
    # -----------------------------
    df['path'] = df.apply(
        lambda row: os.path.join(dataset_path, row['group'], row['file']),
        axis=1
    )

    # -----------------------------
    # 4. Encode group labels as integer class codes
    #    Note: encoding is alphabetical
    # -----------------------------
    df['class'] = pd.Categorical(df['group']).codes

    # -----------------------------
    # 5. Load signals
    # -----------------------------
    signals = []
    samplerates = []

    for _, row in df.iterrows():
    
        signal, fs = sf.read(row['path'])
        
        # in case it's stereo
        signal = np.mean(signal, axis=1) if signal.ndim > 1 else signal        

        signals.append(signal)
        samplerates.append(fs)

    df['signal'] = signals
    df['fs'] = samplerates

    # -----------------------------
    # 6. Check sampling rate
    # -----------------------------
    if df['fs'].nunique() == 1:
        fs_global = int(df['fs'].iloc[0])
        logger.info(f"All signals have same sampling rate: {fs_global} Hz")
    else:
        fs_global = None
        logger.warning(f"Inconsistent sampling rates detected: {df['fs'].unique()}"
)

    return df