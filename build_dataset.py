# build_dataset.py
# ----------------------------------------------------------------------
# Build processed dataset from raw audio corpus (MFBM + Acoustic Features)
# ----------------------------------------------------------------------
import argparse
import logging
import sys
from pathlib import Path
import yaml

# Project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).resolve().parent

# Force project root into path to ensure local src modules are found
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Data Modules
from src.data.data_loader import data_loader
from src.data.preprocessing import preprocessing
from src.data.remove_duplicate_subjects import remove_duplicate_subjects
from src.data.export_dataframe import export_dataframe

# --- Feature Modules
from src.features.get_MFBM import get_MFBM
from src.features.extract_voice_features import extract_voice_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Loads the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset from raw audio corpus extracting MFBM and acoustic features."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--corpora-root", type=str, default=None, help="Path to shared corpora root")
    args = parser.parse_args()

    # Load project configuration
    config_path = PROJECT_ROOT / args.config
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Define Corpora Root (Default: one level above PROJECT_ROOT/corpora)
    corpora_root = args.corpora_root
    if corpora_root is None:
        corpora_root = PROJECT_ROOT.parent / "corpora"
    else:
        corpora_root = Path(corpora_root)

    dataset_name = config["data"]["corpus_name"]

    # --- 1. Load Data
    logger.info(f"Loading dataset: {dataset_name} from {corpora_root}")
    df = data_loader(
        dataset_name=dataset_name,
        data_root=str(corpora_root)
    )

    # --- 2. Data Cleaning (Remove duplicate subjects)
    duplicate_cfg = config["data_cleaning"].get("duplicate_subjects", [])
    if duplicate_cfg:
        logger.info("Checking and removing duplicate subject entries...")
        df = remove_duplicate_subjects(df, duplicate_cfg)
        
        # Quick verification log (optional check mimicking your notebook display)
        check_file = duplicate_cfg[0]["file"]
        remaining = df[df["file"] == check_file]
        logger.info(f"Verification for duplicate file '{check_file}': {len(remaining)} entry remaining.")

    # --- 3. Preprocessing (Normalization & Trimming)
    normalize = config["audio"]["normalize"]
    trim_signal = config["audio"]["trim_signal"]
    
    logger.info(f"Preprocessing signals (normalize={normalize}, trim_signal={trim_signal})...")
    df = preprocessing(
        df,
        normalize=normalize,
        trim_signal=trim_signal
    )

    # --- 4. MFBM Feature Extraction
    features_cfg = config["features"]
    filterbank_plot_path = PROJECT_ROOT / config["results"]["figures_dir"] / "01_mel_filterbank.png"
    
    logger.info("Extracting Mel-Frequency Band Magnitudes (MFBM)...")
    df = get_MFBM(
        df,
        tamanho_in=features_cfg["frame_size_ms"],
        passo_in=features_cfg["hop_size_ms"],
        n_fft=features_cfg["n_fft"],
        n_filters=features_cfg["n_filters"],
        fmax=features_cfg["fmax"],
        sobrep=features_cfg["overlap"],
        edge_trim_frames=features_cfg["edge_trim_frames"],
        save_path=str(filterbank_plot_path),
        print_filters=False
    )

    # --- 5. Acoustic Feature Extraction (Jitter, Shimmer, HNR)
    acoustic_cfg = config["acoustic_features"]
    logger.info("Extracting traditional acoustic features (Jitter, Shimmer, HNR)...")
    df = extract_voice_features(
        df,
        f0_min=acoustic_cfg["f0_min"],
        f0_max=acoustic_cfg["f0_max"],
        jitter_params=tuple(acoustic_cfg["jitter_params"]),
        shimmer_window=acoustic_cfg["shimmer_window"],
        shimmer_params=tuple(acoustic_cfg["shimmer_params"]),
        hnr_params=tuple(acoustic_cfg["hnr_params"]),
        print_report=True
    )

    # --- 6. Export Processed Dataframe
    output_root = PROJECT_ROOT / config["data"].get("processed_dir", "data/processed")
    logger.info(f"Exporting processed dataset to: {output_root}")
    
    export_dataframe(
        df,
        dataset_name=dataset_name,
        output_root=str(output_root),
        expand_mfbm=True,  # Garante compatibilidade dinâmica com o main.py antigo
        drop_columns=['signal', 'mfbm', 'path', 'fs']
    )

    logger.info("Dataset construction complete. Done.")


if __name__ == "__main__":
    main()