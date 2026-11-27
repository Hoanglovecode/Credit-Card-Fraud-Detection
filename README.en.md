# Credit Card Fraud Detection

> 🌐 **Language:** [Tiếng Việt (README.md)](README.md) | **English**

A Machine Learning project for detecting fraudulent credit card transactions (**Credit Card Fraud Detection**) on a real-world, highly imbalanced dataset.

The workflow in [creditcard.ipynb](creditcard.ipynb) covers the full lifecycle: Exploratory Data Analysis (**EDA**), feature scaling (**StandardScaler**), baseline model benchmarking (**Random Forest** vs **Logistic Regression**), hyperparameter tuning (**GridSearchCV**), **Precision–Recall** decision threshold optimization, and final evaluation on an unseen test set using **Accuracy, Precision, Recall, F1-Score, PR-AUC, and ROC-AUC**.

---

## Table of Contents

- [1. Problem Overview](#1-problem-overview)
- [2. Dataset](#2-dataset)
- [3. Project Structure](#3-project-structure)
- [4. Methodology & Pipeline](#4-methodology--pipeline)
  - [4.1. Exploratory Data Analysis (EDA)](#41-exploratory-data-analysis-eda)
  - [4.2. Data Splitting & Feature Scaling](#42-data-splitting--feature-scaling)
  - [4.3. Baseline Model Comparison (Validation Set)](#43-baseline-model-comparison-validation-set)
  - [4.4. Hyperparameter Tuning with GridSearchCV](#44-hyperparameter-tuning-with-gridsearchcv)
  - [4.5. Decision Threshold Optimization](#45-decision-threshold-optimization)
  - [4.6. Final Model Retraining & Test Evaluation](#46-final-model-retraining--test-evaluation)
- [5. Installation & Usage](#5-installation--usage)
- [6. Future Improvements](#6-future-improvements)

---

## 1. Problem Overview

In real-world financial systems, fraudulent transactions represent only a tiny fraction of daily transactions. Because of this extreme class imbalance:
- **Accuracy** alone is misleading (a naive classifier predicting every transaction as valid still achieves >99.8% accuracy).
- The core evaluation metrics are **Precision** (minimizing false alarms for legitimate cardholders), **Recall** (catching as many fraudulent transactions as possible to prevent financial loss), **F1-Score**, and threshold-independent ranking metrics **PR-AUC** (*Precision-Recall AUC*) and **ROC-AUC**.

---

## 2. Dataset

> **Note:** The raw dataset file `creditcard.csv` (~144 MB) exceeds GitHub's 100 MB limit and is excluded from this repository via `.gitignore`. Download it separately into the project root before running the notebook.

- **Source:** [Credit Card Fraud Detection – Kaggle (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dimensions:** `284,807` rows × `31` columns (`30` input features + `1` target label `Class`).
- **Feature Description:**
  - `Time`: Seconds elapsed between each transaction and the first transaction (`0` to `172,792` seconds, ~48 hours).
  - `V1` – `V28`: 28 anonymized numerical features produced via PCA (*Principal Component Analysis*) to protect customer privacy.
  - `Amount`: Transaction amount.
  - `Class`: Binary target label (`0` = Valid transaction, `1` = Fraudulent transaction).

### Class Distribution

| Transaction Type | Label (`Class`) | Count | Percentage (%) |
| :--- | :---: | :---: | :---: |
| Valid Transaction | `0` | 284,315 | 99.8273% |
| Fraudulent Transaction | `1` | 492 | 0.1727% |
| **Total** | — | **284,807** | **100.0000%** |

---

## 3. Project Structure

```text
Credit-Card-Fraud-Detection/
├── creditcard.ipynb    # Jupyter Notebook with EDA, tuning, threshold optimization & evaluation
├── README.md           # Vietnamese documentation
├── README.en.md        # English documentation
├── .gitignore          # Excludes large dataset files (*.csv) and local-only folders
└── creditcard.csv      # (Download separately from Kaggle and place in root directory)
```

---

## 4. Methodology & Pipeline

### 4.1. Exploratory Data Analysis (EDA)

1. **Transaction `Amount` Statistics:**

   | Statistic | Fraudulent (`Class = 1`) | Valid (`Class = 0`) |
   | :--- | :---: | :---: |
   | `count` | 492 | 284,315 |
   | `mean` | **122.21** | 88.29 |
   | `std` | 256.68 | 250.11 |
   | `min` | 0.00 | 0.00 |
   | `25%` | 1.00 | 5.65 |
   | `50%` (median) | 9.25 | 22.00 |
   | `75%` | 105.89 | 77.05 |
   | `max` | 2,125.87 | 25,691.16 |

   > **Key Observation:** Fraudulent transactions have a noticeably higher mean amount (`122.21`) than valid transactions (`88.29`), despite having a lower median (`9.25` vs `22.00`).

2. **Correlation Analysis (`Correlation Heatmap`):**
   - Plotted via `seaborn.heatmap(data.corr(), vmax=.8, square=True)`.
   - Most PCA components (`V1`–`V28`) have low inter-correlation, while features such as `V2` and `V5` exhibit noticeable negative correlation with `Amount`.

---

### 4.2. Data Splitting & Feature Scaling

The dataset is partitioned into **60% Train – 20% Validation – 20% Test** using `train_test_split` (`random_state=42`):

| Split | Ratio | Sample Count |
| :--- | :---: | :---: |
| **Training Set** (`X_train`, `Y_train`) | 60% | 170,883 |
| **Validation Set** (`x_val`, `y_val`) | 20% | 56,962 *(56,884 valid; 78 fraud)* |
| **Test Set** (`x_test`, `y_test`) | 20% | 56,962 *(56,864 valid; 98 fraud)* |

- For baseline comparison, input features are standardized using `StandardScaler` (`fit_transform` on `X_train`, `transform` on `x_val`).

---

### 4.3. Baseline Model Comparison (Validation Set)

Two baseline models were trained on `X_train_scale` and evaluated on `X_val_scale`:
- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Logistic Regression:** `LogisticRegression(max_iter=1000, random_state=42)`

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | TP | FN | FP | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9996** | **0.9661** | **0.7308** | **0.8321** | **57** | **21** | **2** | **56,882** |
| Logistic Regression | 0.9993 | 0.8364 | 0.5897 | 0.6917 | 46 | 32 | 9 | 56,875 |

**Random Forest** outperformed Logistic Regression across all four metrics (detecting 57/78 frauds vs 46/78, while cutting false positives from 9 down to 2) and was selected for hyperparameter tuning and threshold optimization.

---

### 4.4. Hyperparameter Tuning with GridSearchCV

`GridSearchCV` (3-fold CV, optimizing `scoring='roc_auc'`) was run on the training split (`X_train`, `Y_train`) over:
- `n_estimators`: `[50, 100]`
- `max_depth`: `[None, 10, 20]`

The best estimator selected **`max_depth=10`, `n_estimators=100`** (`random_state=42`). Evaluated on `x_val` at the default threshold `0.5`:
- **Accuracy:** `0.9996` | **Precision:** `0.9667` | **Recall:** `0.7436` | **F1-Score:** `0.8406`
- **Confusion Matrix:** `[[56882, 2], [20, 58]]` (58/78 fraud cases detected).

---

### 4.5. Decision Threshold Optimization

Instead of relying on the default `0.5` probability cutoff, predicted probabilities (`predict_proba(x_val)[:, 1]`) were used with `precision_recall_curve` to plot **Precision and Recall vs Threshold** and select `best_threshold` maximizing the **Validation F1-Score**.

#### Validation Set Progression

| Model Stage (Validation Set) | Accuracy | Precision | Recall | F1-Score | TP / 78 | FN (Missed) | FP (False Alarms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Random Forest | 0.9996 | 0.9661 | 0.7308 | 0.8321 | 57 | 21 | 2 |
| GridSearchCV Best RF (`threshold = 0.5`) | 0.9996 | **0.9667** | 0.7436 | 0.8406 | 58 | 20 | **2** |
| **GridSearchCV Best RF + Optimal Threshold** | **0.9997** | 0.9538 | **0.7949** | **0.8671** | **62** | **16** | 3 |

Threshold tuning boosted Validation Recall from `74.36%` to **`79.49%`** and F1-Score from `84.06%` to **`86.71%`**.

---

### 4.6. Final Model Retraining & Test Evaluation

The tuned model (`RandomForestClassifier(**search.best_params_, random_state=42)` with `max_depth=10, n_estimators=100`) was retrained on the combined **Train + Validation** set (`X_train_final`, `Y_train_final` – 80% of the dataset) and evaluated on the held-out **Test** set (`56,962` samples: `56,864` valid and `98` fraud) using `best_threshold`:

| Evaluation Metric | Test Set Score |
| :--- | :---: |
| **Accuracy** | **0.9996** (99.96%) |
| **Precision (Class 1)** | **0.9398** (93.98%) |
| **Recall (Class 1)** | **0.7959** (79.59%) |
| **F1-Score (Class 1)** | **0.8619** (86.19%) |
| **PR-AUC** (`average_precision_score`) | **0.8572** |
| **ROC-AUC** (`roc_auc_score`) | **0.9773** |

**Test Set Confusion Matrix:**

```text
                Predicted: Valid (0)    Predicted: Fraud (1)
Actual: Valid (0)      56,859                   5
Actual: Fraud (1)          20                  78
```

- The final tuned model detected **78 out of 98** fraudulent transactions (`Recall = 79.59%`) while maintaining **`93.98%` Precision** (only `5` false alarms across `56,864` valid transactions).

---

## 5. Installation & Usage

### Prerequisites
- Python `>= 3.8`
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

### Steps to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hoanglovecode/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Download the dataset:**
   - Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
   - Extract and place `creditcard.csv` in the project root directory alongside `creditcard.ipynb`.

4. **Launch and run the Notebook:**
   ```bash
   jupyter notebook creditcard.ipynb
   ```
   Then choose **Run All** to execute all cells sequentially.

---

## 6. Future Improvements

- **Stratified Splitting & Pipelines:** Pass `stratify=ydata` in `train_test_split` to preserve the fraud ratio across all splits, and wrap preprocessing inside a `sklearn.pipeline.Pipeline` during cross-validation.
- **Imbalanced Learning Techniques:** Experiment with **SMOTE**, **Random Undersampling**, or `class_weight="balanced"` paired with `StratifiedKFold`.
- **Advanced Models & Deployment:** Benchmark **XGBoost**, **LightGBM**, and **CatBoost**, add **SHAP** explainability, and package the trained model into a real-time inference API (FastAPI / Streamlit).
