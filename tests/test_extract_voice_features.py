import numpy as np
import pandas as pd
import pytest
from src.features.extract_voice_features import extract_voice_features


@pytest.fixture
def valid_audio_df():
    """Generates a mock DataFrame with a valid synthetic sine wave signal (1 kHz).

    Praat requires a clean periodic signal to extract pitch, jitter, and shimmer
    successfully.
    """
    fs = 16000  # Sampling rate
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # A clean 100 Hz sine wave (perfect for voice pitch detection)
    signal = np.sin(2 * np.pi * 100 * t)

    df = pd.DataFrame(
        {"file": ["clean_voice.wav"], "signal": [signal], "fs": [fs]}
    )
    return df


@pytest.fixture
def corrupt_audio_df():
    """Generates a mock DataFrame with a signal that will cause Praat to fail.

    Pure silence or random uniform noise does not have periodic pitch, causing
    pitch/jitter extraction to return NaN.
    """
    fs = 16000
    signal = np.zeros(fs)  # Pure silence, impossible to extract pitch

    df = pd.DataFrame(
        {"file": ["silent_voice.wav"], "signal": [signal], "fs": [fs]}
    )
    return df


def test_extract_voice_features_success(valid_audio_df):
    """Test that features are successfully extracted from a clean periodic

    signal.
    """
    # Act
    result_df = extract_voice_features(
        valid_audio_df, audio_col="signal", fs_col="fs", print_report=False
    )

    # Assert: Check if all columns were added
    expected_cols = ["meanF0", "stddevF0", "localJitter", "localShimmer", "HNR"]
    for col in expected_cols:
        assert col in result_df.columns
        # Ensure values are valid floats and not NaN
        assert not np.isnan(result_df[col].iloc[0])

    # Assert: Basic range validation for a 100Hz sine wave
    assert 95.0 <= result_df["meanF0"].iloc[0] <= 105.0


def test_extract_voice_features_raises_value_error_on_failure(
    corrupt_audio_df,
):
    """Test that a ValueError is raised when Praat encounters an invalid or

    silent signal.
    """
    # Act & Assert
    # The function should capture the failure in the try/except block,
    # append NaN, and then target the validation block which triggers ValueError.
    with pytest.raises(ValueError) as exc_info:
        extract_voice_features(
            corrupt_audio_df, audio_col="signal", fs_col="fs"
        )

    # Check if the error message mentions the validation failure and the file name
    assert "Invalid values detected" in str(exc_info.value)
    assert "silent_voice.wav" in str(exc_info.value)