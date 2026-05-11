# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import soundfile as sf
import os


def _limit_audio(signal, WS=15, k1=0.001, k2m=10):
    """
    Energy-based speech trimming (endpoint detection).

    This function implements a simple voice activity detection (VAD) approach
    based on short-time energy analysis.

    The signal is divided into non-overlapping frames of fixed size (WS),
    and the energy of each frame is computed. Two thresholds are used:

        - k1: lower energy threshold used to detect potential speech regions
        - k2 = k2m * k1: higher confidence threshold used to confirm speech activity

    The algorithm scans the energy contour from both the beginning and the end
    of the signal in order to detect:

        - Speech onset (first stable high-energy region)
        - Speech offset (last stable high-energy region)

    Once these boundaries are identified, the signal is trimmed accordingly,
    removing leading and trailing low-energy segments (silence or noise).

    Notes
    -----
    - The method assumes that the input signal is already amplitude-normalized.
    - No smoothing is applied to the energy contour.
    - Designed for sustained phonation signals (e.g., vowels).

    Parameters
    ----------
    signal : np.ndarray
        Input audio signal.
    WS_ms : int, optional
        Frame size for short-time energy computation (default is 15 ms).
    k1 : float, optional
        Lower energy threshold for speech detection (default is 0.001).
    k2m : float, optional
        Multiplier for secondary threshold (k2 = k2m * k1).

    Returns
    -------
    np.ndarray
        Trimmed audio signal containing only the active speech region.
    """

    # Number of frames
    n_frames = int(len(signal) / WS)
    if n_frames == 0:
        return signal

    # Compute energies
    energies = np.zeros(n_frames)
    for i in range(n_frames):
        frame = signal[i * WS:(i + 1) * WS]
        energies[i] = np.sum(frame ** 2) / WS

    # Thresholds
    k2 = k2m * k1

    indice_inf = 0
    indice_sup = 0

    stop_inf = 0
    stop_sup = 0

    for i in range(len(energies)):

        if stop_inf == 1 and stop_sup == 1:
            break

        # Lower limit
        if energies[i] > k1 and indice_inf == 0:
            indice_inf = i

        if indice_inf != 0 and stop_inf == 0:
            if energies[i] < k1:
                indice_inf = 0
            elif energies[i] > k2:
                stop_inf = 1

        # Upper limit
        j = len(energies) - 1 - i

        if energies[j] > k1 and indice_sup == 0:
            indice_sup = j

        if indice_sup != 0 and stop_sup == 0:
            if energies[j] < k1:
                indice_sup = 0
            elif energies[j] > k2:
                stop_sup = 1

    # Convert to samples
    start = max((indice_inf - 1) * WS, 0)
    end = min(indice_sup * WS, len(signal))

    if start >= end:
        return signal  # fallback

    trimmed = signal[start:end]

    # Normalize again
    max_val = np.max(np.abs(trimmed))
    if max_val > 0:
        trimmed = trimmed / max_val

    return trimmed


def data_loader(
    dataset_name,
    data_root="../data",
    normalize=True,
    trimm_signal=True,
    WS_ms=15,
    k1=1e-3,
    k2_ratio=10
):
    """
    Load metadata and audio signals from a dataset.
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
    # 4. Encode labels
    # -----------------------------
    df['class'] = pd.Categorical(df['group']).codes

    # -----------------------------
    # 5. Load signals
    # -----------------------------
    signals = []
    samplerates = []

    for _, row in df.iterrows():
        signal, fs = sf.read(row['path'])

        # Optional normalization
        if normalize and np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal))

        signals.append(signal)
        samplerates.append(fs)

    df['signal'] = signals
    df['fs'] = samplerates

    # -----------------------------
    # 6. Optional trimming
    # -----------------------------
    if trimm_signal:
        df["signal"] = [
            _limit_audio(
                sig,
                WS=int(WS_ms * fs_i / 1000),
                k1=k1,
                k2m=k2_ratio
            )
            for sig, fs_i in zip(df["signal"], df["fs"])
        ]

    # -----------------------------
    # 7. Check sampling rate
    # -----------------------------
    if df['fs'].nunique() == 1:
        fs_global = int(df['fs'].iloc[0])
        print(f"All signals have the same sampling rate: {fs_global} Hz")
    else:
        fs_global = None
        print("Warning: inconsistent sampling rates.")

    return df, fs_global