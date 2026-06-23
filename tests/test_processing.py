# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import pytest  

from src.analysis.compute_band_statistics import compute_band_statistics
from src.data.merge_pathology_classes import merge_pathology_classes
from src.analysis.aggregate_band_statistics_per_class import aggregate_band_statistics_per_class

def test_compute_band_statistics_shape():
    df = pd.DataFrame({'mfbm': [np.random.rand(20, 50)]})
    df_out = compute_band_statistics(df)
    assert df_out['mean_MFBM'].iloc[0].shape == (20,)
    assert df_out['std_MFBM'].iloc[0].shape == (20,)

def test_merge_pathology_classes():
    df = pd.DataFrame({'group': ['control', 'edema', 'nodulo']})
    df_out = merge_pathology_classes(df, ['edema', 'nodulo'], 'physio')
    assert set(df_out['group']) == {'control', 'physio'}
    assert df_out['class'].nunique() == 2
    
def test_compute_band_statistics_exact_math():
    """Tests if the mean and standard deviation are computed across the correct axis

    (axis=1, frames) using predictable static values.
    """
    # Create a mock MFBM matrix with shape (2 filters, 3 frames)
    # Filter 0: [1.0, 2.0, 3.0] -> Mean = 2.0, Std = 0.81649658
    # Filter 1: [4.0, 5.0, 6.0] -> Mean = 5.0, Std = 0.81649658
    mock_mfbm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    df = pd.DataFrame({"mfbm": [mock_mfbm]})

    # Act
    df_out = compute_band_statistics(df)

    # Assert columns were created
    assert "mean_MFBM" in df_out.columns
    assert "std_MFBM" in df_out.columns

    extracted_mean = df_out["mean_MFBM"].iloc[0]
    extracted_std = df_out["std_MFBM"].iloc[0]

    # Verify exact mathematical alignment per band
    assert np.allclose(extracted_mean, [2.0, 5.0])
    assert np.allclose(extracted_std, [np.std([1, 2, 3]), np.std([4, 5, 6])])


def test_compute_band_statistics_no_nans_on_stable_input():
    """Ensures that the aggregation pipeline outputs clean arrays without

    any corrupted NaN values.
    """
    n_filters = 20
    n_frames = 50
    random_mfbm = np.abs(np.random.randn(n_filters, n_frames))

    df = pd.DataFrame({"mfbm": [random_mfbm]})

    # Act
    df_out = compute_band_statistics(df)

    # Assert no NaNs are present in the aggregated features
    assert not np.any(np.isnan(df_out["mean_MFBM"].iloc[0]))
    assert not np.any(np.isnan(df_out["std_MFBM"].iloc[0]))
    
    
@pytest.fixture
def mock_processing_df():
    """Fixture que cria um DataFrame simulando a saída de múltiplos sujeitos

    divididos por grupos, contendo 3 bandas de frequência cada.
    """
    # Sujeito 1 (control) - 3 bandas
    mean_s1 = np.array([1.0, 2.0, 3.0])
    std_s1 = np.array([0.1, 0.2, 0.3])

    # Sujeito 2 (control) - 3 bandas
    mean_s2 = np.array([3.0, 4.0, 5.0])
    std_s2 = np.array([0.3, 0.4, 0.5])

    # Sujeito 3 (pathology) - 3 bandas
    mean_s3 = np.array([10.0, 20.0, 30.0])
    std_s3 = np.array([1.0, 1.0, 1.0])

    df = pd.DataFrame(
        {
            "group": ["control", "control", "pathology"],
            "mean_MFBM": [mean_s1, mean_s2, mean_s3],
            "std_MFBM": [std_s1, std_s2, std_s3],
        }
    )
    return df


def test_aggregate_band_statistics_per_class_success(mock_processing_df):
    """Testa se a agregação matemática global por classe é calculada

    corretamente no eixo certo (axis=0).
    """
    # Act
    mean_dict, std_dict = aggregate_band_statistics_per_class(
        mock_processing_df
    )

    # Assert: Devem existir as duas chaves detetadas automaticamente
    assert set(mean_dict.keys()) == {"control", "pathology"}
    assert set(std_dict.keys()) == {"control", "pathology"}

    # Verificação Matemática para o grupo 'control' (Média entre Sujeito 1 e Sujeito 2)
    # mean global = [(1+3)/2, (2+4)/2, (3+5)/2] = [2.0, 3.0, 4.0]
    assert np.allclose(mean_dict["control"], [2.0, 3.0, 4.0])

    # std global = [(0.1+0.3)/2, (0.2+0.4)/2, (0.3+0.5)/2] = [0.2, 0.3, 0.4]
    assert np.allclose(std_dict["control"], [0.2, 0.3, 0.4])

    # Verificação de formato: O output deve ter shape (n_bands,) -> shape (3,)
    assert mean_dict["control"].shape == (3,)


def test_aggregate_band_statistics_with_explicit_classes(mock_processing_df):
    """Testa se a função respeita a lista explícita de classes passada pelo utilizador

    e ignora as restantes.
    """
    # Forçar a agregação apenas de uma classe específica
    mean_dict, std_dict = aggregate_band_statistics_per_class(
        mock_processing_df, classes=["pathology"]
    )

    assert list(mean_dict.keys()) == ["pathology"]
    assert np.allclose(mean_dict["pathology"], [10.0, 20.0, 30.0])


def test_aggregate_band_statistics_empty_class_error(mock_processing_df):
    """Testa se a função lança um ValueError esperado (comportamento atual do numpy)

    caso seja solicitada uma classe que não existe no DataFrame.
    """
    # 'phantom_class' não existe no DataFrame, o subset será vazio
    with pytest.raises(ValueError) as exc_info:
        aggregate_band_statistics_per_class(
            mock_processing_df, classes=["phantom_class"]
        )

    # O numpy queixa-se que precisa de pelo menos uma matriz para empilhar (vstack)
    assert "need at least one array to concatenate" in str(exc_info.value)