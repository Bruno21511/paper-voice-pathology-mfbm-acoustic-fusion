# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.data.export_dataframe import export_dataframe
from src.data.import_dataframe import import_dataframe

def test_export_import_roundtrip(tmp_path):
    matriz_original = np.random.rand(20, 50)
    df = pd.DataFrame({
        'file': ['a.wav'],
        'mfbm': [matriz_original]
    })

    export_dataframe(
        df,
        dataset_name="test",
        output_root=str(tmp_path),
        expand_mfbm=True,
        drop_columns=['mfbm']
    )

    # Deixa o import reconstruir a matriz (rebuild_mfbm=True por defeito)
    df_back = import_dataframe(
        dataset_name="test",
        input_root=str(tmp_path)
    )

    # Compara a matriz inteira devolvida com a original
    np.testing.assert_allclose(
        df_back['mfbm'].iloc[0],
        matriz_original
    )