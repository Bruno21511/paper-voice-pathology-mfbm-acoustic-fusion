# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import parselmouth
from parselmouth.praat import call
from typing import Dict, Any

logger = logging.getLogger(__name__)


def extract_voice_features(
    df: pd.DataFrame,
    audio_col: str = "signal",
    fs_col: str = "fs",
    f0_min: float = 75,
    f0_max: float = 400,
    jitter_params: tuple = (0.0001, 0.02, 1.3),
    shimmer_window: float = 0.03,
    shimmer_params: tuple = (1.3, 1.6),
    hnr_params: tuple = (0.01, 75, 0.1, 1.0),
    print_report: bool = False
) -> pd.DataFrame:
    """
    Extract acoustic voice features (F0, jitter, shimmer, HNR) using 
    Praat/Parselmouth, appending them as new DataFrame columns.

    Rows where extraction fails are assigned NaN for all feature 
    columns, with a warning logged including the row index/file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing audio signals and sampling rates.
    audio_col : str, optional
        Column name containing the audio signal arrays, by default "signal".
    fs_col : str, optional
        Column name containing the sampling rate, by default "fs".
    f0_min : float, optional
        Minimum F0 for pitch detection (Hz), by default 75.
    f0_max : float, optional
        Maximum F0 for pitch detection (Hz), by default 400.
    jitter_params : tuple, optional
        Parameters for Praat's "Get jitter (local)" call.
    shimmer_window : float, optional
        Window size (s) for shimmer extraction, by default 0.03.
    shimmer_params : tuple, optional
        Parameters for Praat's "Get shimmer (local)" call.
    hnr_params : tuple, optional
        Parameters for Praat's "To Harmonicity (cc)" call.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: meanF0, stddevF0, localJitter, 
        localShimmer, HNR.
    """
    unidade = "Hertz"
    inicio = 0.0

    meanF0_list, stdF0_list = [], []
    jitter_list, shimmer_list, hnr_list = [], [], []
    n_failed = 0

    for idx, row in df.iterrows():
        try:
            signal = row[audio_col]
            fs = row[fs_col]
            sound = parselmouth.Sound(signal, sampling_frequency=fs)

            pitch = call(sound, "To Pitch", inicio, f0_min, f0_max)
            meanF0 = call(pitch, "Get mean", 0, 0, unidade)
            stdF0 = call(pitch, "Get standard deviation", 0, 0, unidade)

            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)

            localJitter = call(pointProcess, "Get jitter (local)", 0, 0, *jitter_params)
            localShimmer = call([sound, pointProcess], "Get shimmer (local)",
                                 0, 0, jitter_params[0], shimmer_window, *shimmer_params)

            harmonicity = call(sound, "To Harmonicity (cc)", *hnr_params)
            hnr = call(harmonicity, "Get mean", 0, 0)

            meanF0_list.append(meanF0)
            stdF0_list.append(stdF0)
            jitter_list.append(localJitter)
            shimmer_list.append(localShimmer)
            hnr_list.append(hnr)

        except Exception as e:
            file_id = row.get('file', idx)
            logger.warning(f"Feature extraction failed for {file_id}: {e}")
            n_failed += 1
            meanF0_list.append(np.nan)
            stdF0_list.append(np.nan)
            jitter_list.append(np.nan)
            shimmer_list.append(np.nan)
            hnr_list.append(np.nan)

    df = df.copy()

    df["meanF0"] = meanF0_list
    df["stddevF0"] = stdF0_list
    df["localJitter"] = jitter_list
    df["localShimmer"] = shimmer_list
    df["HNR"] = hnr_list

    # -------------------------
    # Validation
    # -------------------------

    target_cols = [
        "meanF0",
        "stddevF0",
        "localJitter",
        "localShimmer",
        "HNR"
    ]

    for col in target_cols:

        invalid = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
            .isna()
        )

        if invalid.any():

            files = (
                df.loc[invalid, "file"]
                .tolist()
            )

            raise ValueError(
                f"Invalid values detected in '{col}'. "
                f"Affected files: {files}"
            )

    logger.info(
        f"Voice feature extraction completed: "
        f"{len(df) - n_failed}/{len(df)} succeeded"
    )
    
    # Verifying values
    if print_report:
        print(df[["localJitter", "localShimmer", "HNR"]].describe())        

    return df

