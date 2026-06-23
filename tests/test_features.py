# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from src.features.get_MFBM import _mel_filterbank, _obter_FFT, get_MFBM

def test_obter_FFT_shape():
    sinal = np.random.randn(22050)
    spec = _obter_FFT(sinal, tamanho=661, passo=220, n_fft=2048)
    assert spec.shape[0] == 1024

def test_get_MFBM_shape():
    df = pd.DataFrame({
        'signal': [np.random.randn(22050)],
        'fs': [22050]
    })
    df_out = get_MFBM(
        df,
        tamanho_in=30,
        passo_in=10,
        n_fft=2048,
        n_filters=20,
        fmax=4000,
        sobrep=0.5
    )
    assert df_out['mfbm'].iloc[0].shape[0] == 20
    assert not np.any(np.isnan(df_out['mfbm'].iloc[0]))
    
def test_mel_filterbank_shape():
    filt = _mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    assert filt.shape[0] == 20

def test_mel_filterbank_no_nans():
    filt = _mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    assert not np.any(np.isnan(filt))

def test_mel_filterbank_normalized():
    filt = _mel_filterbank(fs=22050, n_filters=20, fmax=4000, sobrep=0.5, n_fft=2048)
    sums = filt.sum(axis=1)
    assert np.allclose(sums[1:], 1.0, atol=1e-6)
    
    
    
def test_get_MFBM_with_multiple_sampling_rates():
    """Tests if the filterbank cache logic correctly handles and processes

    a dataset containing mixed sampling rates (e.g., 22050 Hz and 16000 Hz).
    """
    # Setup mixed sampling rates
    df_mixed = pd.DataFrame(
        {
            "signal": [
                np.random.randn(11025),  # 0.5s of audio at 22050 Hz
                np.random.randn(8000),  # 0.5s of audio at 16000 Hz
            ],
            "fs": [22050, 16000],
        }
    )

    # Act
    df_out = get_MFBM(
        df_mixed,
        tamanho_in=25.0,
        passo_in=10.0,
        n_fft=1024,
        n_filters=26,
        fmax=4000,
        sobrep=0.5,
        edge_trim_frames=0,  # disable trimming to isolate cache logic
    )

    # Assert: verify both rows generated the requested 26 Mel bands
    assert df_out["mfbm"].iloc[0].shape[0] == 26
    assert df_out["mfbm"].iloc[1].shape[0] == 26
    assert len(df_out) == 2


def test_get_MFBM_short_signal_trimming_resilience():
    """Tests that a signal too short for edge trimming doesn't crash the code

    and is handled gracefully via the logger warning block.
    """
    # A very short signal (e.g., only 600 samples)
    short_signal = np.random.randn(600)
    df_short = pd.DataFrame({"signal": [short_signal], "fs": [16000]})

    # Act
    # Using 25ms windows with 10ms hops means 300 samples will yield very few frames,
    # making it lower than 2 * edge_trim_frames (2 * 5 = 10)
    df_out = get_MFBM(
        df_short,
        tamanho_in=25.0,
        passo_in=10.0,
        n_fft=512,
        n_filters=20,
        edge_trim_frames=5,
    )

    # Assert: It should bypass trimming without an IndexError and output the feature matrix
    assert "mfbm" in df_out.columns
    assert df_out["mfbm"].iloc[0].shape[0] == 20