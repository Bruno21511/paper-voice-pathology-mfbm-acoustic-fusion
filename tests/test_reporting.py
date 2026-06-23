# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import pytest

from src.evaluation.build_accuracy_table import build_accuracy_table
from src.evaluation.build_auc_table import build_auc_table
from src.analysis.flatten_results import flatten_results


# -------------------------------------------------------------------------
# 1. FIXTURES E MOCKS
# -------------------------------------------------------------------------
@pytest.fixture
def mock_tidy_df():
    """Cria um DataFrame em formato tidy longo que simula os resultados

    finais do pipeline para alimentar as tabelas de Accuracy e AUC.
    """
    rows = []
    features_list = ["acoustic", "spectral", "all"]
    tasks_list = ["Control_vs_PhLP", "Control_vs_UVFP", "All_classes"]

    for feat in features_list:
        for task in tasks_list:
            # Adicionar métrica base de accuracy
            rows.append(
                {
                    "features": feat,
                    "task": task,
                    "metric": "accuracy",
                    "value": 0.8543,
                    "std": 0.0312,
                }
            )
            # Adicionar métrica base de AUC
            rows.append(
                {
                    "features": feat,
                    "task": task,
                    "metric": "auc",
                    "value": 0.9125,
                    "std": 0.0144,
                }
            )

        # Adicionar métricas One-vs-All (apenas para a tarefa multiclasse)
        for idx in range(3):
            rows.append(
                {
                    "features": feat,
                    "task": "All_classes",
                    "metric": f"ova_class_{idx}",
                    "value": 0.8800,
                    "std": 0.0200,
                }
            )

    return pd.DataFrame(rows)


@dataclass
class MockConfig:
    task_name: str
    feature_set: str


@dataclass
class MockExperimentResult:
    """Mock leve para simular a classe estruturada ExperimentResult."""

    config: MockConfig
    confusion_matrix: List[List[int]]
    confusion_matrix_norm: List[List[float]]
    accuracy_mean: float = 0.85
    accuracy_std: float = 0.03
    precision_mean: float = 0.84
    precision_std: float = 0.04
    recall_mean: float = 0.83
    recall_std: float = 0.05
    f1_mean: float = 0.83
    f1_std: float = 0.05
    auc_mean: float = 0.90
    auc_std: float = 0.02
    ova_mean: List[float] = None
    ova_std: List[float] = None

    def __post_init__(self):
        if self.ova_mean is None:
            self.ova_mean = [0.88, 0.87, 0.89]
        if self.ova_std is None:
            self.ova_std = [0.01, 0.02, 0.01]


# -------------------------------------------------------------------------
# 2. TESTES PARA AS TABELAS DE APRESENTAÇÃO (Accuracy e AUC)
# -------------------------------------------------------------------------
def test_build_accuracy_table_structure_and_mapping(mock_tidy_df):
    """Garante que a build_accuracy_table faz o pivot correto, aplica os nomes

    bonitos e junta as colunas clássicas com as colunas One-vs-All (*).
    """
    table = build_accuracy_table(mock_tidy_df)

    # Verifica se as linhas foram reordenadas para os nomes bonitos
    assert list(table.index) == ["Acoustic", "Spectral", "Combined"]

    # Verifica se as colunas principais e as colunas OvA foram mapeadas e fundidas
    expected_cols = [
        "HE vs. PhLP",
        "HE vs. UVFP",
        "3-Class",
        "HE vs. All (*)",
        "PhLP vs. All (*)",
        "UVFP vs. All (*)",
    ]
    for col in expected_cols:
        assert col in table.columns

    # Verifica a formatação string do valor "mean ± std" convertido para percentagem
    # 0.8543 -> 85.43% | 0.0312 -> 3.12%
    assert table.loc["Acoustic", "HE vs. PhLP"] == "85.43 ± 3.12"


def test_build_auc_table_structure_and_filtering(mock_tidy_df):
    """Garante que a build_auc_table filtra as tarefas multiclasse e formata

    o AUC com as 4 casas decimais exigidas.
    """
    table = build_auc_table(mock_tidy_df)

    # Verifica o índice ordenado
    assert list(table.index) == ["Acoustic", "Spectral", "Combined"]

    # Garante que a tarefa "All_classes" foi excluída (AUC é apenas binário aqui)
    assert "3-Class" not in table.columns
    assert "All_classes" not in table.columns

    # Verifica se as colunas binárias esperadas estão presentes
    assert "Control vs. PhLP" in table.columns
    assert "Control vs. UVFP" in table.columns

    # Verifica a formatação com 4 casas decimais (0.9125 e 0.0144)
    assert table.loc["Acoustic", "Control vs. PhLP"] == "0.9125 ± 0.0144"


# -------------------------------------------------------------------------
# 3. TESTES PARA O ACHATAMENTO DE RESULTADOS (flatten_results)
# -------------------------------------------------------------------------
def test_flatten_results_matrices_and_scalars():
    """Garante que o flatten_results consegue extrair corretamente as métricas

    escalares e lida dinamicamente com as matrizes de confusão (ex: 2x2).
    """
    # Configurar um resultado fictício de matriz 2x2
    config = MockConfig(task_name="Binary_Task", feature_set="acoustic")
    cm_mock = [[10, 2], [1, 15]]
    cm_norm_mock = [[0.833, 0.167], [0.062, 0.938]]

    result_obj = MockExperimentResult(
        config=config,
        confusion_matrix=cm_mock,
        confusion_matrix_norm=cm_norm_mock,
    )

    # Executar achatamento
    df_metrics, df_cm = flatten_results([result_obj])

    # Assertions sobre o DataFrame de Métricas Escalares
    assert isinstance(df_metrics, pd.DataFrame)
    assert set(df_metrics["metric"].unique()).issuperset(
        {"accuracy", "precision", "recall", "f1", "auc", "ova_class_0"}
    )
    # Verificar valor específico
    acc_row = df_metrics[df_metrics["metric"] == "accuracy"].iloc[0]
    assert acc_row["value"] == 0.85
    assert acc_row["std"] == 0.03

    # Assertions sobre o DataFrame das Matrizes de Confusão
    assert isinstance(df_cm, pd.DataFrame)
    assert len(df_cm) == 4  # Matriz 2x2 gera exatamente 4 combinações/linhas

    # Verificar se as coordenadas da célula foram preservadas
    target_cell = df_cm[
        (df_cm["true_index"] == 0) & (df_cm["pred_index"] == 1)
    ].iloc[0]
    assert target_cell["count"] == 2
    assert target_cell["norm_value"] == 0.167