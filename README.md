# Noise Detection in Dynamical Systems and ECG via Complexity Metrics — Experimental Code Repository

This repository implements the experimental pipeline described in the paper:

**“Detection and Comparative Evaluation of Noise Perturbations in Dynamical Systems and ECG Signals Using Complexity-Based Features”** :contentReference[oaicite:0]{index=0}

---

## Overview

The project studies how controlled noise perturbations affect complexity metrics in different signal generators and evaluates whether supervised learning can predict:

1. **Noise type** (Gaussian vs. pink vs. low-frequency)  
2. **Noise intensity** (multiple levels within each noise type, plus clean)  
3. **Noise presence** (clean vs. noisy)

Signals are generated for:
- **Rössler** (continuous chaotic system)
- **Lorenz** (continuous chaotic system)
- **Hénon map** (discrete chaotic system)
- **Synthetic ECG** (NeuroKit2-based)
- **AR(1)** (autoregressive baseline, used as additional experiment)

Noise types and intensity grids are consistent across systems:
- Gaussian: **[1, 0.3, 0.1]**
- Pink: **[2, 1, 0.2]**
- Low-frequency: **[0.2, 1, 2]** :contentReference[oaicite:1]{index=1}

---

## Conceptual Pipeline

1. **Step 1 — Signal generation + controlled noise injection**
   - Generate long clean realizations per system
   - Z-score normalization
   - Inject Gaussian / pink / low-frequency noise at configured intensities

2. **Step 2 — Feature dataset construction**
   - Rolling windows (default: length **300**, stride configurable)
   - Compute **18 complexity metrics**
   - Export ML-ready CSV datasets for three classification tasks 

3. **Step 3 — Machine learning experiments**
   - Repeated **group-aware** train/test splits using `signal_id` (run-level grouping; no leakage)
   - CatBoostClassifier (out-of-the-box; no hyperparameter tuning)

4. **Step 4/5 — Post-hoc analysis and visualization**
   - Confusion matrices, feature importances, summary tables
   - Global cross-system comparison plots (boxplots + intensity line plots)

---

## Repository Structure

### Step 1 — Data generation (per system)

- `01a_create_ECG_data_restructured_seeded.py` :contentReference[oaicite:3]{index=3}  
- `01b_create_Lorenz_data_restructured_seeded.py`
- `01c_create_Roessler_data_restrcutured_seeded.py`
- `01d_create_Henon_data_restrcutured_seeded.py`
- `01e_create_AR1_data_restructured_seeded.py` :contentReference[oaicite:4]{index=4}  

Each Step-1 script:
- generates clean signals (multiple runs),
- produces noisy variants for each noise configuration,
- writes `.npz` files and an index CSV,
- writes example plots (PNG + EPS).

### Step 2 — ML dataset construction

- `02_create_ML_dataset_restrctured_id.py` :contentReference[oaicite:5]{index=5}  

Builds rolling-window feature tables and task-specific CSVs under `ML_tasks/`.

### Step 3 — ML experiments runner

- `03_Run_ML.py` :contentReference[oaicite:6]{index=6}  

Runs repeated, group-aware splits and stores per-split and aggregated results under `ML_experiments/`.

### Step 4 — Post-hoc plots and aggregated results

- `04_plots_and_analysis.py` :contentReference[oaicite:7]{index=7}  

Creates plots (PNG + EPS) and aggregated tables under `Results/<DATASET>/...`.

### Step 5 — Global comparison across systems

- `05_boxplots_lineplots.py`   

Loads Step-2 task CSVs across systems and produces comparable plots under `Results/Global_Comparison/...`.

---

## Complexity Metrics (18 features)

The feature set consists of entropy-based, fractal/scaling, statistical, and SVD/embedding-spectrum measures. The paper lists the full set of 18 metrics and their representations (raw signal vs. embedding matrix). 

The ML runner and global comparison scripts explicitly use the following feature columns (18 metrics): 
- `var1der`, `var2der`
- `std_dev`, `mad`, `cv`
- `approximate_entropy`, `sample_entropy`, `permutation_entropy`, `lempel_ziv_complexity`
- `dfa`, `hurst`
- `fisher_info`, `fisher_info_nk`
- `svd_entropy`, `rel_decay`, `svd_energy`, `condition_number`, `spectral_skewness`

---

## Requirements

python=3.11

antropy==0.1.9
bottleneck==1.4.2
brotlicffi==1.1.0.0
catboost==1.2.8
certifi==2025.11.12
cffi==2.0.0
charset-normalizer==3.4.4
contourpy==1.3.1
cycler==0.11.0
fonttools==4.60.1
graphviz==0.20.1
idna==3.11
joblib==1.5.2
kiwisolver==1.4.8
llvmlite==0.45.1
matplotlib==3.10.6
narwhals==2.7.0
neurokit2==0.2.10
numba==0.62.1
numexpr==2.14.1
numpy==2.2.5
packaging==25.0
pandas==2.3.3
pillow==11.1.0
plotly==6.3.0
pycparser==2.23
pyparsing==3.2.0
PyQt6==6.9.1
PyQt6-sip==13.10.2
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.5
scikit-learn==1.7.1
scipy==1.15.3
setuptools==80.9.0
sip==6.12.0
six==1.17.0
threadpoolctl==3.5.0
tomli==2.2.1
tornado==6.5.1
urllib3==2.5.0
wheel==0.45.1
win_inet_pton==1.1.0
future==1.0.0
nolds==0.6.2
openpyxl==3.1.5
