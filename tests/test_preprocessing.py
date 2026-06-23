# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from src.data.preprocessing import preprocessing


def test_preprocessing_peak_normalization():

    df = pd.DataFrame({
        "signal": [
            np.array([0.5, -1.0, 0.2])
        ]
    })

    df_out = preprocessing(
        df,
        normalize="peak",
        dc_remove=False,
        trim_signal=False,
        equal_duration=False
    )

    signal = df_out["signal"].iloc[0]

    assert np.isclose(np.max(np.abs(signal)), 1.0)


def test_preprocessing_dc_removal():

    df = pd.DataFrame({
        "signal": [
            np.array([2.0, 3.0, 4.0])
        ]
    })

    df_out = preprocessing(
        df,
        normalize=None,
        dc_remove=True,
        trim_signal=False,
        equal_duration=False
    )

    signal = df_out["signal"].iloc[0]

    assert np.isclose(np.mean(signal), 0.0)

   
    
def test_preprocessing_empty_signal_handling():
    
    df = pd.DataFrame({"signal": [np.array([])], "fs": [16000]})
    
    
    with pytest.raises(ValueError) as exc_info:
        preprocessing(df, dc_remove=True, trim_signal=True, normalize="peak")
        
    assert "DataFrame has empty audio signals" in str(exc_info.value)
    
    
    
import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import preprocessing


def test_preprocessing_rms_normalization():
    """Tests if RMS normalization correctly forces the signal's root-mean-square

    to 1.0.
    """
    df = pd.DataFrame({"signal": [np.array([1.0, 2.0, 3.0, 4.0])]})

    df_out = preprocessing(
        df,
        normalize="rms",
        dc_remove=False,
        trim_signal=False,
        equal_duration=False,
    )

    signal = df_out["signal"].iloc[0]
    rms_value = np.sqrt(np.mean(signal**2))

    assert np.isclose(rms_value, 1.0)


def test_preprocessing_trim_silence():
    """Tests if energy-based trimming correctly removes padding silence from edges

    while keeping the high-energy center.
    """
    fs = 16000
    # Create a signal: 0.2s silence + 0.2s high energy sine wave + 0.2s silence
    silence_len = int(fs * 0.2)
    sine_len = int(fs * 0.2)

    t = np.linspace(0, 0.2, sine_len, endpoint=False)
    sine_wave = np.sin(2 * np.pi * 100 * t)  # High energy

    full_signal = np.concatenate(
        [np.zeros(silence_len), sine_wave, np.zeros(silence_len)]
    )

    df = pd.DataFrame({"signal": [full_signal], "fs": [fs]})

    df_out = preprocessing(
        df,
        normalize=None,
        dc_remove=False,
        trim_signal=True,
        equal_duration=False,
        WS_ms=15,
        k1=1e-3,
        k2_ratio=10,
    )

    trimmed_signal = df_out["signal"].iloc[0]

    # The resulting signal should be shorter than the original 0.6 seconds
    assert len(trimmed_signal) < len(full_signal)
    # It should still contain our target high energy components
    assert np.max(np.abs(trimmed_signal)) > 0.9
    
    


def test_preprocessing_zero_amplitude_raises_value_error():
    """Tests that an audio signal filled with pure zeros triggers a ValueError

    when trim_signal=True, because energy scaling is impossible.
    """
    df = pd.DataFrame({"signal": [np.zeros(1000)], "fs": [16000]})

    with pytest.raises(ValueError) as exc_info:
        preprocessing(df, trim_signal=True)

    assert "Cannot trim signal with zero amplitude." in str(exc_info.value)
    
    
    


def test_preprocessing_invalid_normalize_param():
    """Tests that providing an unsupported normalization string raises a

    ValueError.
    """
    df = pd.DataFrame({"signal": [np.array([1, 2, 3])]})

    with pytest.raises(ValueError) as exc_info:
        preprocessing(df, normalize="invalid_method")

    assert "normalize must be None, 'peak' or 'rms'" in str(exc_info.value)