# Discriminating Voice Pathologies Through a Combination of Spectral and Acoustic Features
### Feature fusion analysis in the MEEI corpus

![CI](https://github.com/Bruno21511/paper-voice-pathology-mfbm/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Paper](https://img.shields.io/badge/Paper-CENTERIS_2025-red)

This repository contains the code and data associated with the paper:

> **"Discriminating Voice Pathologies Through a Combination of Spectral and Acoustic Features"**  
> B. Rodrigues, H. Cordeiro, G. Marques  
> CENTERIS – International Conference on ENTERprise Information Systems / ProjMAN / HCist 2024  
> *Procedia Computer Science*, Volume 256, Pages 835–842, 2025  
> DOI: [10.1016/j.procs.2025.02.185](https://doi.org/10.1016/j.procs.2025.02.185)

The full paper can be found [here:](paper/Rodrigues_et_al_HCist2024.pdf)

The work investigates whether combining spectral (MFBM) and acoustic (jitter, shimmer and HNR) parameters improves the discrimination of healthy and pathological voices, using sustained vowel /a/ recordings and a Support Vector Machine (SVM) classifier.

### Methodological Note
All results shown below were obtained using the current implementation in this repository.
<b>Minor differences from the original publication may occur</b> due to refactoring and random seed handling, but mainly due to a correction in the dataset (see Data section for details).<br>
Values are reported as <b>mean ± standard deviation</b> across cross‑validation folds, rather than mean ± 95% confidence interval as presented in the article.

---

## Overview

This work addresses the following question: <br>
> Does combining spectral and acoustic parameters improve the discrimination of healthy speakers (HE), speakers with physiological laryngeal pathologies (PhLP), and speakers with unilateral vocal fold paralysis (UvfP)?

Four classification tasks are addressed: three One vs. One binary and one 3-class, each performed with acoustic features only, spectral features only, and their combination.

### 1. Binary and 3-class classification accuracies (in %)

| Task | Acoustic | Spectral | Combined |
| :--- | :---: | :---: | :---: |
| Control vs. PhLP | 84.66 ± 1.00 | 87.45 ± 1.89 | **94.84 ± 1.09** |
| Control vs. UVFP | 90.24 ± 0.72 | 85.03 ± 1.71 | **93.37 ± 0.77** |
| PhLP vs. UVFP | 69.85 ± 1.29 | 66.10 ± 2.77 | **73.55 ± 2.12** |
| 3-Class | 67.67 ± 1.11 | 66.70 ± 2.05 | **76.54 ± 1.67** |

The figure below shows a bar plot summarizing the accuracies of all 12 experimental conditions (four tasks × three feature sets).

<p align="center">
  <img src="results/figures/04_accuracies_bar.png" width="750">
</p>

### 2. One-vs-All classification results (derived from 3-class model) (in %)

These results are derived directly from the 3‑class confusion matrices, enabling fair comparison with previous studies that report One‑vs‑All performance on the same corpus subset.

| Task | Acoustic | Spectral | Combined | Study <sup>[[1]](#ref1)</sup> result | Study <sup>[[2]](#ref2)</sup> result | 
| :--- | :---: | :---: | :---: | :---: | :---: |
| Control vs. All (*) | 88.08 ± 0.61 | 88.05 ± 1.24 | **92.53 ± 0.72** | 87.0 | 90.1 | 
| PhLP vs. All (*) | 70.34 ± 1.12 | 73.36 ± 2.04 | **80.10 ± 1.74** | 73.4 | 74.7 | 
| UVFP vs. All (*) | 76.93 ± 0.95 | 71.99 ± 2.14 | **80.45 ± 1.59** | 76.0 | 79.2 | 


### 3. Binary and 3-class classification AUC (area Under the Curve)

AUC scores provide a threshold-independent measure of  of discriminative performance

| Task | Acoustic | Spectral | Combined |
| :--- | :---: | :---: | :---: |
| Control vs. PhLP | 0.903 ± 0.005 | 0.947 ± 0.008 | **0.968 ± 0.004** |
| Control vs. UVFP | 0.960 ± 0.003 | 0.938 ± 0.010 | **0.974 ± 0.005** |
| PhLP vs. UVFP | 0.755 ± 0.010 | 0.705 ± 0.024 | **0.767 ± 0.014** |

Across all tasks, <b>combining acoustic and spectral parameters consistently yields the highest performance</b>.

---

## Repository Structure

```
├── data/
│   ├── mysMEEI
│   │   └── mysMEEI.csv                	# Audio corpus metadata (filename, age, gender, group)
│   └── processed/
│       └── mysMEEI.parquet             # Pre-extracted features (no audio signals)
│
├── paper/
│   └── Rodrigues_et_al_HCist2024.pdf   # Original article re-implemented by this repository
│
├── notebooks/
│   ├── 01_feature_extraction.ipynb     # Spectral and acoustic feature extraction
│   └── 02_classification.ipynb         # Classification, cross-validation, results
│
├── results/
│   ├── figures/                        # Plots and visualisations
│   └── metrics/                        # Accuracy tables, confusion matrices and per-class metrics
│
├── src/
│   ├── data_loader.py                      # Load audio files and metadata
│   ├── mel_filterbank.py                   # Define Mel filterbank (custom bands)
│   ├── get_MFBM.py                         # Extract Mel Frequency Band Magnitudes (MFBM)
│   ├── extract_voice_features.py           # Compute acoustic features (Jitter, Shimmer, HNR)
│   ├── build_X.py                          # Assemble feature matrices for classification
│   ├── run_cv_experiment.py                # Cross-validation pipeline (SVM, metrics)
│   ├── export_dataframe.py                 # Save extracted features to disk
│   ├── import_dataframe.py                 # Load precomputed feature data
│   ├── plot_meldefined_magnitudes_per_class.py  # Visualise spectral features per class
│   └── plot_acoustic_features.py           # Visualise acoustic feature distributions
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Data

### Audio Corpus (MEEI)

The audio signals used in this work are a subset of the **Massachusetts Eye and Ear Infirmary (MEEI)** Voice Disorders Database, commercialised by Kay Elemetrics Corp. The subset used in this implementation closely follows the one used in [[1]](#ref1)[[2]](#ref2) (see Note on Dataset below). 

| Class | Description | N |
|-------|-------------|---|
| HE | Healthy speakers | 36 |
| PhLP | Physiological laryngeal pathologies (vocal nodules, edema) | 55 |
| UvfP | Unilateral vocal fold paralysis | 58 |
| **Total** | | **149** |

Signals contain the sustained vowel /a/, sampled at 25 kHz, 16-bit PCM encoding. One signal per speaker was used, with no separation by age or gender.

The audio files are **not included** in this repository as the MEEI database is a commercial product.

### Note on Dataset

The original MEEI subset contains 154 recordings. Three subjects were excluded 
due to failed acoustic feature extraction, reducing the working dataset to 151, consistent with the original study.

In this implementation, an additional duplicate subject was identified, present in 
both the Edema and UVFP classes. Both instances were removed, resulting in a final 
dataset of 149 subjects.

This correction introduces minor differences in the reported results relative to 
the published article, but does not affect the conclusions: the advantage of 
combining acoustic and spectral features holds consistently across all tasks.

### Known Limitations of the MEEI Database

The MEEI database is widely used in the literature, but presents known methodological limitations that should be considered when interpreting results. These limitations are documented in the literature (Saenz-Lechon et al., 2006) and are acknowledged here as a limitation of the present work.

---

## Method

### 1. Pre-processing
- Amplitude normalisation (absolute maximum set to 1)
- Silence removal at signal boundaries using an energy threshold method based on [[3]](#ref3)

### 2. Spectral Feature Extraction
- Framing: 30 ms windows, 10 ms step; first and last 3 frames discarded
- FFT: 4096-point, per frame
- Mel filterbank: 20 triangular filters, unit area, uniformly spaced on the Mel scale, covering 0–4 kHz
- Per-band mean and standard deviation computed across frames → two 20-dimensional vectors
- Last 8 dimensions discarded (high-frequency bands, not discriminative)
- Remaining 12 dimensions split into lower (bands 1–6) and intermediate (bands 7–12) frequency groups
- PCA: trained on training set; applied to both sets; lower bands reduced to 2 PCs, intermediate to 1 PC
- Final spectral representation: 6 dimensions (four arrays: 2+2+1+1)

### 3. Acoustic Feature Extraction
- Extracted via Parselmouth (Python interface to Praat) [[4]](#ref4)
- Features: Jitter (%), Shimmer (%), Harmonic-to-Noise Ratio (HNR)
- Final acoustic representation: 3 dimensions

### 4. Feature Combination
- Spectral (6D) and acoustic (3D) features concatenated → 9-dimensional feature vector

### 5. Classification
- Classifier: SVM with RBF kernel (Scikit-learn defaults)
- Scaling: StandardScaler fitted on training data, applied separately to train and test
- Validation: Stratified 5-fold cross-validation, 1000 iterations with unique random states
- Metric: Accuracy reported as mean ± standard deviation across iterations

### Note on Terminology
The spectral features extracted are **filterbank magnitudes**, not energies, despite the terminology used in the published article. This reflects an evolution of the codebase during the research; the results were obtained using magnitudes, as implemented here.

---

## Key Findings

The experimental results highlight several consistent patterns across all classification tasks:

- **Feature combination is consistently beneficial**: Combining acoustic and spectral features yields the highest performance across all classification tasks, outperforming either feature set used in isolation.
- **Feature relevance is class-dependent**: Spectral features show greater discriminative power for PhLP, while acoustic features are more effective for identifying UvfP, suggesting that different pathologies manifest more strongly in distinct feature domains.
- **Complementarity between feature types**: The observed performance gains indicate that acoustic and spectral features capture complementary information, reinforcing the advantage of integrating both representations.

These findings suggest potential benefits in adopting ensemble or hierarchical classification strategies, where acoustic and spectral features are processed separately and combined at a later stage, leveraging their complementary discriminative contributions.

---

## Usage

### Requirements

See requirements.txt for dependencies.

### Quick start (recommended)

#### Option A — Starting from pre-extracted features

If you already have the processed parquet file (included in this repo under `data/processed/`), simply run:
python main.py

This reproduces all paper results: classification metrics, confusion matrices, and figures, saved to `results/metrics/` and `results/figures/`.

#### Option B — Starting from raw audio files

Place the audio corpus outside the repository, following the structure below:

```
parent_directory/
├── paper-voice-pathology-mfbm-acoustic-fusion/
└── corpora/
    └── mysMEEI/
        ├── control/
        │   └── *.wav
        ├── edema/
        │   └── *.wav
        ├── uvfp/
        │   └── *.wav
        ├── nodulo/
        │   └── *.wav
        └── mysMEEI.csv              # Audio corpus metadata (filename, age, gender, group)
```

**Note**: The corpus directory is configured separately from the repository to avoid storing large audio files in version control.

Run the two steps in order:<br><br>
python build_dataset.py<br>
python main.py

### Interactive exploration (notebooks)

For step-by-step exploration, visualization, and the figures used in the README, open and run the notebooks in order:

- `notebooks/01_MFBM_extraction.ipynb` — feature extraction (requires raw audio corpus)
- `notebooks/02_analysis.ipynb` — classification and analysis (requires only the processed parquet file)
<br><br>

**Configuration**: Before running the pipeline, check or adjust the dataset names, paths, and extraction parameters in the `config.yaml` file located at the root of the project.

---

## Citation

If you use this code in your research, please cite:

> **"Discriminating Voice Pathologies Through a Combination of Spectral and Acoustic Features"**  
> B. Rodrigues, H. Cordeiro, G. Marques  
> CENTERIS – International Conference on ENTERprise Information Systems / ProjMAN / HCist 2024  
> *Procedia Computer Science*, Volume 256, Pages 835–842, 2025  
> DOI: [10.1016/j.procs.2025.02.185](https://doi.org/10.1016/j.procs.2025.02.185)

---

## References

<a id="ref1"></a>
[1] H. Cordeiro, J. Fonseca, I. Guimarães, C. Meneses, "Hierarchical Classification and System Combination for Automatically Identifying Physiological and Neuromuscular Laryngeal Pathologies," Journal of Voice, 2017.

<a id="ref2"></a>
[2] H. Cordeiro, J. Fonseca, I. Guimarães, C. Meneses, "Voice Pathologies Identification: Speech Signals, Features and Classifiers Evaluation," SPA, 2015.

<a id="ref3"></a>
[3] L. Lamel, L. Rabiner, A. Rosenberg, J. Wilpon, "An Improved Endpoint Detector for Isolated Word Recognition," IEEE Transactions on Acoustics, Speech and Signal Processing, 1981.

<a id="ref4"></a>
[4] Y. Jadoul, B. Thompson, B. de Boer, "Introducing Parselmouth: A Python Interface to Praat," Journal of Phonetics, 2018.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
