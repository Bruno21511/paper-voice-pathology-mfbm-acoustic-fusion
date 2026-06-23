# -*- coding: utf-8 -*-
import pandas as pd

def build_auc_table(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the AUC summary table from a tidy long-format dataframe,
    focusing strictly on binary classification tasks.
    """

    # ---------------------------------------------------------
    # Pretty names mapping
    # ---------------------------------------------------------
    feature_map = {
        "acoustic": "Acoustic",
        "spectral": "Spectral",
        "all": "Combined"
    }

    task_map = {
        "Control_vs_PhLP": "Control vs. PhLP",
        "Control_vs_UVFP": "Control vs. UVFP",
        "PhLP_vs_UVFP": "PhLP vs. UVFP"
    }

    # ---------------------------------------------------------
    # 1) Filter AUC and exclude multi-class tasks
    # ---------------------------------------------------------
    df_auc = tidy_df[
        (tidy_df["metric"] == "auc") & 
        (~tidy_df["task"].isin(["All_classes", "3-Class"]))
    ].copy()

    # ---------------------------------------------------------
    # 2) Format values as "mean ± std" (4 decimal places for AUC)
    # ---------------------------------------------------------
    df_auc["formatted"] = (
        df_auc["value"].round(4).get_level_values if isinstance(df_auc["value"], pd.MultiIndex) 
        else df_auc["value"].apply(lambda x: f"{x:.4f}")
    ) + " ± " + df_auc["std"].apply(lambda x: f"{x:.4f}")

    # ---------------------------------------------------------
    # 3) Pivot to wide format
    # ---------------------------------------------------------
    table_auc = df_auc.pivot(
        index="features",
        columns="task",
        values="formatted"
    )

    # Rename indexes and columns using the maps
    table_auc = table_auc.rename(index=feature_map, columns=task_map)

    # ---------------------------------------------------------
    # 4) Order rows and columns to preserve strict presentation
    # ---------------------------------------------------------
    row_order = ["Acoustic", "Spectral", "Combined"]
    col_order = ["Control vs. PhLP", "Control vs. UVFP", "PhLP vs. UVFP"]
    
    # Reindex safely (ignores tasks that might not be in the current slice)
    final_table = table_auc.reindex(index=row_order, columns=col_order)

    return final_table
