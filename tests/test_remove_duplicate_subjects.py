import pandas as pd
import pytest
from src.data.remove_duplicate_subjects import remove_duplicate_subjects


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a clean, representative DataFrame for testing."""
    data = {
        "file": [
            "audio1.wav",
            "audio1.wav",
            "audio2.wav",
            "audio3.wav",
            "audio3.wav",
        ],
        "group": ["control", "phlp", "uvfp", "control", "phlp"],
        "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    return pd.DataFrame(data)


def test_remove_duplicate_subjects_success(sample_dataframe):
    """Test that duplicates are successfully removed based on file and groups."""
    # Setup duplicate logic: remove 'audio1.wav' only when it is in 'phlp'
    duplicate_subjects = [{"file": "audio1.wav", "groups": ["phlp"]}]

    result_df = remove_duplicate_subjects(sample_dataframe, duplicate_subjects)

    # Assertions
    assert len(result_df) == 4  # Should remove exactly 1 row
    # The 'audio1.wav' with 'control' must still exist
    assert (
        result_df[
            (result_df["file"] == "audio1.wav")
            & (result_df["group"] == "control")
        ].shape[0]
        == 1
    )
    # The 'audio1.wav' with 'phlp' must be gone
    assert (
        result_df[
            (result_df["file"] == "audio1.wav") & (result_df["group"] == "phlp")
        ].shape[0]
        == 0
    )


def test_remove_duplicate_subjects_no_matching_group(sample_dataframe):
    """Test that rows are NOT removed if the file matches but the group does

    not.
    """
    # Setup: 'audio2.wav' exists, but its group is 'uvfp', not 'control'
    duplicate_subjects = [{"file": "audio2.wav", "groups": ["control"]}]

    result_df = remove_duplicate_subjects(sample_dataframe, duplicate_subjects)

    # Assertions
    assert (
        len(result_df) == 5
    )  # No rows should be removed because groups don't match
    assert "audio2.wav" in result_df["file"].values


def test_remove_duplicate_subjects_empty_list(sample_dataframe):
    """Test that the DataFrame remains untouched if the duplicate list is

    empty.
    """
    duplicate_subjects = []

    result_df = remove_duplicate_subjects(sample_dataframe, duplicate_subjects)

    # Assertions
    assert len(result_df) == 5
    pd.testing.assert_frame_equal(result_df, sample_dataframe)


def test_remove_multiple_groups_for_same_file(sample_dataframe):
    """Test that multiple groups specified for the same file are all removed."""
    # Setup: remove 'audio3.wav' from both 'control' and 'phlp'
    duplicate_subjects = [
        {"file": "audio3.wav", "groups": ["control", "phlp"]}
    ]

    result_df = remove_duplicate_subjects(sample_dataframe, duplicate_subjects)

    # Assertions
    assert len(result_df) == 3  # Should remove both rows for audio3.wav
    assert "audio3.wav" not in result_df["file"].values