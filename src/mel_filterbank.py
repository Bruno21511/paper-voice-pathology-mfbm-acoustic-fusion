# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def mel_filterbank(fs, n_filters, fmax, sobrep, n_fft, print_filters=False, save_path=None):
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
    if print_filters:
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

        plt.show()

    return filt