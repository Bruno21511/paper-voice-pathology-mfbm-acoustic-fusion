# -*- coding: utf-8 -*-
"""Main execution pipeline for voice pathology acoustic fusion experiments."""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, does NOT open windows

import logging
from pathlib import Path
import sys
import yaml

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Fix execution context: resolve project root independently of invocation directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load project modules
from src.analysis.aggregate_band_statistics_per_class import (
    aggregate_band_statistics_per_class,
)
from src.analysis.compute_band_statistics import compute_band_statistics
from src.analysis.flatten_results import flatten_results
from src.analysis.run_experiment_grid import run_experiment_grid
from src.data.import_dataframe import import_dataframe
from src.data.merge_pathology_classes import merge_pathology_classes
from src.evaluation.build_accuracy_table import build_accuracy_table
from src.evaluation.build_auc_table import build_auc_table
from src.visualization.plot_3_confusion_matrices import plot_3_confusion_matrices
from src.visualization.plot_acoustic_features import plot_acoustic_features
from src.visualization.plot_accuracy_bars import plot_accuracy_bars
from src.visualization.plot_mfbm_statistics_per_class import (
    plot_mfbm_statistics_per_class,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Pipeline Initialization & Configuration
    # ------------------------------------------------------------------
    logging.info("Loading project configuration...")
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve output directories and ensure they exist
    metrics_dir = PROJECT_ROOT / config["results"]["metrics_dir"]
    figures_dir = PROJECT_ROOT / config["results"]["figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # ------------------------------------------------------------------
    dataset_name = config["data"]["corpus_name"]
    logging.info(f"Loading precomputed MFBM features for corpus: {dataset_name}")

    df = import_dataframe(
        dataset_name, input_root=str(PROJECT_ROOT / "data" / "processed")
    )

    logging.info("Computing per-band statistics...")
    df = compute_band_statistics(df)

    logging.info("Merging pathology classes according to configuration...")
    merge_cfg = config["class_merging"]
    df = merge_pathology_classes(
        df,
        classes_to_merge=merge_cfg["groups_to_merge"],
        merged_label=merge_cfg["merged_label"],
    )

    # ------------------------------------------------------------------
    # 3. Exploratory Data Visualizations
    # ------------------------------------------------------------------
    logging.info("Generating profile plots (Mean/Std per class)...")
    mean_dict, std_dict = aggregate_band_statistics_per_class(df)
    plot_mfbm_statistics_per_class(
        mean_dict,
        std_dict,
        save_path=str(figures_dir / "02_mean_std_per_class.png"),
    )

    logging.info("Generating acoustic features boxplots...")
    acoustic_features = [
        ("localJitter", "Jitter (dB)"),
        ("localShimmer", "Shimmer (dB)"),
        ("HNR", "HNR (dB)"),
    ]
    plot_acoustic_features(
        df,
        acoustic_features,
        log_transform=("localJitter", "localShimmer"),
        save_path=str(figures_dir / "03_acoustic_features_per_class.png"),
    )

    # ------------------------------------------------------------------
    # 4. Experiment Grid Execution
    # ------------------------------------------------------------------
    feature_sets = config["experiments"]["feature_sets"]
    tasks = config["experiments"]["tasks"]
    num_iters = config["experiments"]["num_iters"]

    logging.info(
        f"Starting experiment grid (Tasks: {len(tasks)} × Feature Sets: {len(feature_sets)}) with {num_iters} iterations..."
    )
    all_results = run_experiment_grid(
        df,
        tasks=tasks,
        feature_sets=feature_sets,
        num_iters=num_iters,
        print_report=False,
    )

    # ------------------------------------------------------------------
    # 5. Result Flattening & Exporting
    # ------------------------------------------------------------------
    logging.info("Processing and exporting metrics...")
    tidy_metrics_df, tidy_cm_df = flatten_results(all_results)

    tidy_metrics_df.to_csv(
        str(metrics_dir / "tidy_metrics.csv"), index=False, sep=";"
    )
    tidy_cm_df.to_csv(str(metrics_dir / "tidy_cm.csv"), index=False, sep=";")

    # Generate summary evaluation tables
    accuracy_table = build_accuracy_table(tidy_metrics_df)
    accuracy_table.to_csv(
        str(metrics_dir / "accuracy_table.csv"), index=False, sep=";"
    )
    print("\n=== ACCURACY TABLE ===")
    print(accuracy_table)

    auc_table = build_auc_table(tidy_metrics_df)
    auc_table.to_csv(str(metrics_dir / "auc_table.csv"), index=False, sep=";")
    print("\n=== AUC TABLE ===")
    print(auc_table)

    # ------------------------------------------------------------------
    # 6. Evaluation Plots & Confusion Matrices
    # ------------------------------------------------------------------
    logging.info("Generating final evaluation plots...")
    plot_accuracy_bars(
        tidy_metrics_df, save_path=str(figures_dir / "04_accuracies_bar.png")
    )

    # Build confusion matrix task list mapping
    label_mapping = {"control": "Control", "phlp": "PhLP", "uvfp": "UvfP"}
    tasks_to_plot = []

    for task_name, classes in config["experiments"]["tasks"].items():
        if classes is None:
            labels = list(label_mapping.values())
            filename = "05_ConfMatr_3_classes.png"
        else:
            labels = [label_mapping[c] for c in classes]
            filename = f"05_ConfMatr_{labels[0]}_vs_{labels[1]}.png"

        tasks_to_plot.append(
            {"task": task_name, "labels": labels, "filename": filename}
        )

    # Loop to save each matrix contextually
    for item in tasks_to_plot:
        logging.info(f"Plotting confusion matrices for task: {item['task']}")
        plot_3_confusion_matrices(
            tidy_cm_df,
            task_name=item["task"],
            class_names=item["labels"],
            save_path=str(figures_dir / item["filename"]),
        )

    logging.info("Pipeline executed successfully. All outputs saved.")


if __name__ == "__main__":
    main()