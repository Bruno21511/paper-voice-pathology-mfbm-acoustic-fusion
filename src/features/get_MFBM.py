# -*- coding: utf-8 -*-
import logging
from typing import Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft

logger = logging.getLogger(__name__)

WindowType = Optional[Literal["hann", "hamming"]]


def _mel_filterbank(
    fs: int,
    n_filters: int,
    fmax: float,
    sobrep: float,
    n_fft: int,
    print_filters: bool = False,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a Mel-scale triangular filterbank with controllable overlap.

    Filters are uniformly spaced on the Mel scale and converted to Hz.
    Each filter is triangular with a peak of 1 at the centre frequency,
    rising linearly from the lower edge to the centre, and falling linearly
    from the centre to the upper edge. Overlap between adjacent filters is
    controlled by the sobrep parameter.

    Each filter is normalised so that its coefficients sum to 1, ensuring
    that the filterbank output represents average energy per band rather
    than total energy, which would otherwise scale with filter bandwidth.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.
    n_filters : int
        Number of triangular filters.
    fmax : float
        Maximum frequency covered by the filterbank, in Hz.
    sobrep : float
        Overlap factor between adjacent filters (e.g. 0.5 = 50% overlap).
        Applied symmetrically: each filter extends sobrep * bandwidth beyond
        its nominal lower and upper edges.
    n_fft : int
        FFT size. The filterbank covers n_fft // 2 frequency bins.
    print_filters : bool, optional
        If True, plots all filters (default: False).
    save_path : str or None, optional
        If provided, saves the filter plot to this path at 300 dpi.
        Only used when print_filters is True.

    Returns
    -------
    filt_mel : np.ndarray, shape (n_filters, n_fft // 2)
        Filterbank matrix. Each row is one normalised triangular filter.
    """

    # Frequency axis — one value per FFT bin
    freqs = np.arange(0, n_fft // 2) * (fs / n_fft)

    # ------------------------------------------------------------------
    # Mel conversion
    # ------------------------------------------------------------------
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    # ------------------------------------------------------------------
    # Uniform spacing in Mel; overlap applied before converting to Hz
    # ------------------------------------------------------------------
    mel_max = hz_to_mel(fmax)
    banda = mel_max / n_filters
    banda_inicial = 0

    low    = np.zeros(n_filters)
    center = np.zeros(n_filters)
    high   = np.zeros(n_filters)

    for i in range(n_filters):
        center[i] = banda_inicial + banda / 2
        low[i]    = banda_inicial - banda * sobrep
        high[i]   = banda_inicial + banda + banda * sobrep
        banda_inicial += banda

    # Convert Mel edges to Hz
    low    = mel_to_hz(low)
    center = mel_to_hz(center)
    high   = mel_to_hz(high)

    # ------------------------------------------------------------------
    # Build triangular filters
    # Each filter has value 1 at centre, 0 at edges.
    # The centre bin belongs exclusively to the falling slope (fall = 1),
    # avoiding double-counting and preserving the triangular shape.
    # ------------------------------------------------------------------
    freqs2D = freqs[None, :]   # (1, n_fft//2)
    low     = low[:, None]     # (n_filters, 1)
    center  = center[:, None]
    high    = high[:, None]

    # Rising slope: ]low, center[
    rise = (freqs2D - low) / (center - low + 1e-12)

    # Falling slope: [center, high]
    fall = 1 - (freqs2D - center) / (high - center + 1e-12)

    filt = np.where(
        (freqs2D > low) & (freqs2D < center),
        rise,
        np.where(
            (freqs2D >= center) & (freqs2D <= high),
            fall,
            0
        )
    )

    # ------------------------------------------------------------------
    # Normalise each filter to unit sum
    # ------------------------------------------------------------------
    filt /= (np.sum(filt, axis=1, keepdims=True) + 1e-12)

    # Preserve original behaviour: first bin of first filter set to zero
    filt[0, 0] = 0

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if print_filters or save_path:
        plt.figure(figsize=(12, 6))
        for i in range(filt.shape[0]):
            plt.plot(freqs, filt[i])
        plt.title("Mel Filterbank")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, fmax * 1.1)
        plt.grid()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if print_filters:
            plt.show()

    return filt
    
    
def _obter_FFT(
    sinal_in: np.ndarray,
    tamanho: int,
    passo: int,
    n_fft: int,
    window_type: WindowType = None
) -> np.ndarray:
    """
    Compute frame-wise FFT of a signal (STFT-like representation).

    Parameters
    ----------
    sinal_in : np.ndarray
        Input signal.
    tamanho : int
        Frame size (samples).
    passo : int
        Hop size (samples).
    n_fft : int
        FFT size.
    window_type : {"hann", "hamming"} or None
        Window applied before FFT. If None, no additional windowing is applied.

    Returns
    -------
    np.ndarray
        STFT-like matrix: (n_freq_bins, n_frames)
        using ONLY positive frequencies (excluding Nyquist).
    """
    
    # Creating window
    if window_type == 'hamming':
        window = np.hamming(tamanho)
    elif window_type == 'hann':
        window = np.hanning(tamanho)
    else:
        window = np.ones(tamanho) # Rectangular (no window)

    frames = []

    for i in range(0, len(sinal_in) - tamanho +1, passo):
        frame = sinal_in[i:i + tamanho]
        
        # Windowing
        windowed_frame = frame * window
        
        spec = fft(windowed_frame, n_fft)
        frames.append(spec)
        
    if len(frames) == 0:
        raise ValueError(
            "Signal shorter than analysis window"
    )

    sinal = np.vstack(frames)
    return sinal[:, :n_fft // 2].T




def get_MFBM(
    df: pd.DataFrame,
    tamanho_in: float,
    passo_in: float,
    n_fft: int,
    n_filters: int = 20,
    fmax: float = 4000,
    sobrep: float = 0.5,
    window_type: WindowType = None,
    edge_trim_frames: int = 2,
    print_filters: bool = False,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute Mel Filter Bank Magnitudes (MFBM).

    The function automatically adapts to the sampling rate of each signal.
    If all signals share the same sampling rate, the filterbank is computed
    only once. Otherwise, filterbanks are cached and reused per fs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing:
            - 'signal'
            - 'fs'

    tamanho_in : int
        Frame size in milliseconds.

    passo_in : int
        Hop size in milliseconds.

    n_fft : int
        FFT size.

    n_filters : int
        Number of Mel filters.

    fmax : float
        Maximum frequency of the filterbank.

    sobrep : float
        Overlap factor between filters.

    window_type : str, optional
        Window type used in STFT.
        
    edge_trim_frames : int
        Number of frames removed from the beginning and end
        of each signal representation to reduce boundary artefacts.

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with new column:
            - 'mfbm'
    """

    mfbm_list = []

    # -------------------------------------------------
    # Check sampling rates
    # -------------------------------------------------

    unique_fs = df['fs'].unique()

    same_fs = len(unique_fs) == 1

    # -------------------------------------------------
    # Precompute filterbank if fs is global
    # -------------------------------------------------

    if same_fs:

        fs_global = unique_fs[0]

        filt_global = _mel_filterbank(
            fs=fs_global,
            n_filters=n_filters,
            fmax=fmax,
            sobrep=sobrep,
            n_fft=n_fft,
            print_filters=print_filters,
            save_path=save_path            
        )

    else:

        # cache for filterbanks
        filterbank_cache = {}

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------

    for sinal, fs_i in zip(df['signal'], df['fs']):

        # --- Time parameters (in samples)
        tamanho = round(tamanho_in * fs_i / 1000)
        passo = round(passo_in * fs_i / 1000)

        # -------------------------------------------------
        # Select / create filterbank
        # -------------------------------------------------

        if same_fs:

            filt = filt_global

        else:

            # create only if not already cached
            if fs_i not in filterbank_cache:

                filterbank_cache[fs_i] = _mel_filterbank(
                    fs=fs_i,
                    n_filters=n_filters,
                    fmax=fmax,
                    sobrep=sobrep,
                    n_fft=n_fft,
                    print_filters=print_filters,
                    save_path=save_path  
                )

            filt = filterbank_cache[fs_i]

        # -------------------------------------------------
        # STFT
        # -------------------------------------------------

        X = _obter_FFT(
            sinal,
            tamanho,
            passo,
            n_fft,
            window_type=window_type
        )
        
        # Trim first and last frames
        if edge_trim_frames > 0:

            if X.shape[1] > 2 * edge_trim_frames:
                X = X[:, edge_trim_frames:-edge_trim_frames]

            else:
                logger.warning(
                    "Signal too short for edge trimming. "
                    "No frames were removed."
                )

        # magnitude
        X = np.abs(X)

        # apply filterbank
        mfbm = np.dot(filt, X)

        mfbm_list.append(mfbm)

    # -------------------------------------------------
    # Store results
    # -------------------------------------------------

    df = df.copy()
    df["mfbm"] = mfbm_list

    return df