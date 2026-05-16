<div align="center">

# Appliance Power Consumption — Time Series Classification

![OOF Accuracy](https://img.shields.io/badge/OOF%20Accuracy-81%25-2ea44f?style=for-the-badge)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-88%25-0075ca?style=for-the-badge)
![Best Model](https://img.shields.io/badge/Best%20Model-Random%20Forest-6e40c9?style=for-the-badge)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.81-e05d44?style=for-the-badge)
![Features](https://img.shields.io/badge/Feature%20Space-256%20features-f0883e?style=for-the-badge)

**Course:** Time Series Analysis & Classification (TSAC) 2025/2026  
**Dataset:** 100 training samples × 10 appliance classes × 1460 timesteps

</div>

---

## Problem

Given raw power consumption recordings of home appliances, classify each signal into one of **10 categories**:

| Label | Appliance |
|:---:|---|
| `0` | Mobile phone charger |
| `1` | Coffee machine |
| `2` | Computer station + monitor |
| `3` | Fridge / Freezer |
| `4` | Hi-Fi system |
| `5` | Lamp (CFL) |
| `6` | Laptop charger |
| `7` | Microwave oven |
| `8` | Printer |
| `9` | Television (LCD/LED) |

---

## Approach

The pipeline combines four complementary feature families:

**Handcrafted Features**
Time-domain statistics (mean, std, kurtosis, percentiles), peak/transition counts, FFT spectral features, and Daubechies wavelet coefficients.

**ROCKET / MiniRocket**
5,000–10,000 random convolutional kernels applied to the raw signal — captures local shape patterns and variable-length subsequences that statistical features miss. Standalone OOF accuracy: 75%.

**DTW-KNN Meta-Features**
Out-of-fold shape-distance probabilities using Dynamic Time Warping — handles time-shifted signals elastically (e.g. Fridge startup lag varying across recordings). Standalone OOF: 43% — used as meta-features, not standalone.

**Catch22**
22 canonical nonlinear time-series statistics including autocorrelation at multiple lags and entropy measures. Critical for Fridge whose periodic compressor cycling produces a distinctive autocorrelation signature.

### Key Design Decisions

- **10-fold stratified CV** throughout — with only 10 samples per class, any single split is unreliable
- **OOF stacking** for ROCKET and DTW-KNN meta-features — fully leak-free
- **Heavy regularisation** (L1/L2 penalties, depth limits, subsampling) on all models — N=100 with 256 features is a high-dimensional small-sample problem
- **Optuna tuning** — 50 trials per model with TPE sampler

---

## Results

<div align="center">

| Rank | Model | OOF Accuracy |
|:---:|---|:---:|
| 1 | **Random Forest** | **81%** |
| 2 | XGBoost | 80% |
| 3 | LightGBM | 77% |
| 4 | Extra Trees | 76% |
| 5 | Logistic Regression | 74% |
| 6 | SVM (RBF) | 71% |
| — | Voting Ensemble | 79% |

</div>

> **Final test set accuracy (Kaggle): 88%** — 7 points above the OOF estimate, confirming the model generalises well to unseen data.

### Per-Class Highlights

| Best Classes | Metric | Hardest Classes | Metric |
|---|---|---|---|
| TV | 100% Recall | Microwave | F1 = 0.70 |
| Mobile | 100% Precision | Lamp | F1 = 0.75 |

---

## Repository Structure

```
appliance-classification/
├── final_submission_notebook.ipynb   # Full notebook (report + code combined)
├── KR.csv                            # Final prediction vector (test set)
├── train.csv                         # Training data (100 samples × 1460 timesteps)
├── test.csv                          # Test data
└── README.md
```

---

## How to Run

### Locally

```bash
pip install lightgbm xgboost catboost optuna scikit-learn sktime pycatch22 dtaidistance
jupyter notebook final_submission_notebook.ipynb
```

`train.csv` and `test.csv` must be in the same directory as the notebook.

### On Kaggle

Change the two data loading lines in Section 2 to:

```python
train_raw = pd.read_csv('/kaggle/input/competitions/appliance-consumption-signatures/train.csv', header=None)
test_raw  = pd.read_csv('/kaggle/input/competitions/appliance-consumption-signatures/test.csv')
```

Then run all cells. Output is saved to `KR.csv`.

---

## Dependencies

```
scikit-learn    lightgbm    xgboost    catboost    optuna
sktime          pycatch22   dtaidistance
numpy           pandas      matplotlib  seaborn     scipy
```

---

<div align="center">

TSAC 2025/2026 — Time Series Analysis & Classification

</div>
