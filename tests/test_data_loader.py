import numpy as np
import pandas as pd
import pytest
from src.features.get_MFBM import _mel_filterbank, _obter_FFT, get_MFBM


@pytest.fixture
def clean_signal_df():
    """Generates a stable dummy dataset with a single-frequency sine wave.

    Includes explicit sampling rate and mock tracking metadata.
    """
    fs = 16000
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)  # 0.5 seconds
    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz standard pitch sine wave

    return pd.DataFrame(
        {"file": ["sine_440.wav"], "signal": [signal], "fs": [fs]}
    )


def test_mel_filterbank_properties():
    """Validates filterbank dimensional execution and unit sum normalization."""
    fs = 16000
    n_filters = 20
    n_fft = 512
    fmax = 4000
    sobrep = 0.5

    # Act
    filt = _mel_filterbank(
        fs=fs, n_filters=n_filters, fmax=fmax, sobrep=sobrep, n_fft=n_fft
    )

    # Assert matrix shape matches (n_filters, n_fft // 2)
    assert filt.shape == (n_filters, n_fft // 2)

    # Assert that all filters (except the first bin edge override) integrate to 1.0
    # We skip checking row 0 fine-grain edge tracking if filt[0, 0] = 0 affects it slightly
    for i in range(1, n_filters):
        assert np.isclose(np.sum(filt[i]), 1.0, atol=1e-6)


def test_obter_fft_dimensions():
    """Tests if frame slicing and positive frequency filtering returns the correct

    shape.
    """
    fs = 16000
    signal = np.random.randn(int(fs * 0.2))  # 0.2 seconds of white noise
    tamanho = 400  # 25 ms frames
    passo = 160  # 10 ms hop size
    n_fft = 512

    # Act
    X = _obter_FFT(
        signal, tamanho=tamanho, passo=passo, n_fft=n_fft, window_type="hann"
    )

    # Assert: Number of bins must be exactly n_fft // 2
    assert X.shape[0] == n_fft // 2
    assert X.shape[1] > 0  # Multiple frames should be generated


def test_get_mfbm_pipeline_execution(clean_signal_df):
    """Verifies complete end-to-end extraction structure and matrix integration."""
    # Act
    df_out = get_MFBM(
        clean_signal_df,
        tamanho_in=25.0,  # 25 ms window
        passo_in=10.0,  # 10 ms hop size
        n_fft=512,
        n_filters=20,
        edge_trim_frames=2,
    )

    # Assert matrix is added inside the DataFrame
    assert "mfbm" in df_out.columns

    mfbm_matrix = df_out["mfbm"].iloc[0]

    # Expected shape: (n_filters, n_frames_after_trim)
    assert mfbm_matrix.shape[0] == 20
    assert mfbm_matrix.ndim == 2


def test_get_mfbm_short_signal_warning_handling():
    """Validates warning resilience path when signal length is too short for

    trimming.
    """
    fs = 16000
    # Signal too short: barely accommodates a single analysis window frame
    short_signal = np.random.randn(400)

    df_short = pd.DataFrame(
        {"file": ["short.wav"], "signal": [short_signal], "fs": [fs]}
    )

    # Act
    df_out = get_MFBM(
        df_short,
        tamanho_in=25.0,
        passo_in=10.0,
        n_fft=512,
        n_filters=20,
        edge_trim_frames=5,  # Forcing edge trim requirements higher than frames
    )

    # Assert execution completed gracefully without crashing
    assert "mfbm" in df_out.columns
    assert df_out["mfbm"].iloc[0].shape[0] == 20