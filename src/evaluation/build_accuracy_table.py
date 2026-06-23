# -*- coding: utf-8 -*-
import pandas as pd

def build_accuracy_table(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the accuracy summary table from a tidy long-format dataframe.
    """

    # ---------------------------------------------------------
    # Pretty names
    # ---------------------------------------------------------
    feature_map = {
        "acoustic": "Acoustic",
        "spectral": "Spectral",
        "all": "Combined"
    }

    task_map = {
        "Control_vs_PhLP": "HE vs. PhLP",
        "Control_vs_UVFP": "HE vs. UVFP",
        "PhLP_vs_UVFP": "PhLP vs. UVFP",
        "All_classes": "3-Class"
    }

    ova_map = {
        "ova_class_0": "HE vs. All (*)",
        "ova_class_1": "PhLP vs. All (*)",
        "ova_class_2": "UVFP vs. All (*)"
    }

    # ---------------------------------------------------------
    # 1) MAIN ACCURACY TABLE
    # ---------------------------------------------------------
    df_acc = tidy_df[tidy_df["metric"] == "accuracy"].copy()

    df_acc["formatted"] = (
        (df_acc["value"] * 100).round(2).astype(str)
        + " ± " +
        (df_acc["std"] * 100).round(2).astype(str)
    )

    table_main = df_acc.pivot(
        index="features",
        columns="task",
        values="formatted"
    )

    table_main = table_main.rename(index=feature_map, columns=task_map)

    # ---------------------------------------------------------
    # 2) OVA TABLE — filter only All_classes
    # ---------------------------------------------------------
    df_ova = tidy_df[
        (tidy_df["metric"].str.startswith("ova_class_")) &
        (tidy_df["task"] == "All_classes")
    ].copy()

    df_ova["formatted"] = (
        (df_ova["value"] * 100).round(2).astype(str)
        + " ± " +
        (df_ova["std"] * 100).round(2).astype(str)
    )

    table_ova = df_ova.pivot(
        index="features",
        columns="metric",
        values="formatted"
    )

    table_ova = table_ova.rename(index=feature_map, columns=ova_map)

    # ---------------------------------------------------------
    # 3) MERGE BOTH TABLES
    # ---------------------------------------------------------
    final_table = table_main.join(table_ova)

    # ---------------------------------------------------------
    # 4) ORDER ROWS
    # ---------------------------------------------------------
    final_table = final_table.loc[["Acoustic", "Spectral", "Combined"]]

    return final_table
