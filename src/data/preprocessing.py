# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging

from typing import Optional

logger = logging.getLogger(__name__)


def _limit_audio(
    signal: np.ndarray,
    WS: int = 15,
    k1: float = 1e-3,
    k2m: int = 10
) -> np.ndarray:
    """
    Energy-based speech trimming.

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    WS : int
        Frame size in samples.
    k1 : float
        Lower energy threshold.
    k2m : int
        Secondary threshold multiplier.

    Returns
    -------
    np.ndarray
        Trimmed signal.
    """

    n_frames = int(len(signal) / WS)
    if n_frames == 0:
        return signal

    # Peak normalize a LOCAL copy for energy calculation to avoid modifying the original signal
    max_val = np.max(np.abs(signal))

    if max_val <= 0:
        raise ValueError(
            "Cannot trim signal with zero amplitude."
        )

    norm_signal = signal / max_val

    energies = np.zeros(n_frames)
    for i in range(n_frames):
        frame = norm_signal[i * WS:(i + 1) * WS]
        energies[i] = np.sum(frame ** 2) / WS

    k2 = k2m * k1
    start_idx = 0
    end_idx = n_frames - 1

    # 1. Search beginning (Forward)
    found_start = False
    for i in range(len(energies)):
        if energies[i] > k1 and not found_start:
            start_idx = i
        if start_idx != 0 and not found_start:
            if energies[i] < k1:
                start_idx = 0
            elif energies[i] > k2:
                found_start = True
                break

    # 2. Search end (Backward) - Separated loop to avoid early break bug
    found_end = False
    for j in range(len(energies) - 1, -1, -1):
        if energies[j] > k1 and not found_end:
            end_idx = j
        if end_idx != (n_frames - 1) and not found_end:
            if energies[j] < k1:
                end_idx = n_frames - 1
            elif energies[j] > k2:
                found_end = True
                break

    start = max((start_idx - 1) * WS, 0)
    end = min((end_idx + 1) * WS, len(signal)) # +1 to include the full trailing frame

    if start >= end:
        return signal

    return signal[start:end].copy()



def preprocessing(
    df: pd.DataFrame,
    normalize: Optional[str] = None,
    dc_remove: bool = False,
    trim_signal: bool = False,
    equal_duration: bool = False,
    WS_ms: int = 15,
    k1: float = 1e-3,
    k2_ratio: int = 10
) -> pd.DataFrame:
    """
    Apply preprocessing operations to audio signals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'signal' column and optionally 'fs'.

    normalize : str or None
        Normalization method:
        - 'peak'
        - 'rms'
        - None

    dc_remove : bool
        Remove DC component.

    trim_signal : bool
        Apply energy-based silence removal.

    equal_duration : bool
        Crop all signals to the shortest signal.
        
    WS_ms : int, optional
        Frame size in milliseconds for energy-based trimming, by default 15.
        
    k1 : float, optional
        Lower energy threshold for trim_signal, by default 1e-3.
        
    k2_ratio : int, optional
        Multiplier of k1 defining the upper confirmation threshold for 
        trim_signal, by default 10.

    Returns
    -------
    pd.DataFrame
        Processed dataframe.
    """

    df = df.copy()
    
    if normalize not in [None, "peak", "rms"]:
        raise ValueError(
            "normalize must be None, 'peak' or 'rms'"
        )
    
    if df["signal"].apply(lambda x: len(x) == 0).any():
        raise ValueError(
            "DataFrame has empty audio signals"
        )


    # -----------------------------
    # 1. DC removal
    # -----------------------------
    if dc_remove:

        df["signal"] = df["signal"].apply(
            lambda x: x - np.mean(x)
        )


    # -----------------------------
    # 2. Trim silence
    # -----------------------------
    if trim_signal:
    
        if "fs" not in df.columns:
            raise ValueError(
                "Column 'fs' required when trim_signal=True"
        )

        df["signal"] = [
            _limit_audio(
                sig,
                WS=int(WS_ms * fs / 1000),
                k1=k1,
                k2m=k2_ratio
            )
            for sig, fs in zip(df["signal"], df["fs"])
        ]


    # -----------------------------
    # 3. Equal duration
    # -----------------------------
    if equal_duration:

        min_length = min(
            len(sig)
            for sig in df["signal"]
        )

        df["signal"] = df["signal"].apply(
            lambda x: x[:min_length]
        )
    
    # -----------------------------
    # 5. Normalization
    # -----------------------------
    if normalize == "peak":

        df["signal"] = df["signal"].apply(
            lambda x: x / np.max(np.abs(x))
            if np.max(np.abs(x)) > 0 else x
        )


    elif normalize == "rms":

        df["signal"] = df["signal"].apply(
            lambda x: x / np.sqrt(np.mean(x**2))
            if np.sqrt(np.mean(x**2)) > 0 else x
        )


    logger.info(
        "Preprocessing completed "
        f"(dc_remove={dc_remove}, "
        f"trim_signal={trim_signal}, "
        f"equal_duration={equal_duration}, "
        f"normalize={normalize})"
)

    return df