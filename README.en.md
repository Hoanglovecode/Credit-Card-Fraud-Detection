# Credit Card Fraud Detection

> 🌐 **Language:** [Tiếng Việt (README.md)](README.md) | **English**

A Machine Learning project for detecting fraudulent credit card transactions (**Credit Card Fraud Detection**) on a real-world, highly imbalanced dataset.

This project covers the complete end-to-end workflow: Exploratory Data Analysis (**EDA**), feature scaling (**StandardScaler**), training and benchmarking **Random Forest** against **Logistic Regression** on a validation set, and retraining the selected model on the combined (`Train + Validation`) data for final evaluation on a held-out test set.

---

## Table of Contents

- [1. Problem Overview](#1-problem-overview)
- [2. Dataset](#2-dataset)
- [3. Project Structure](#3-project-structure)
- [4. Methodology & Pipeline](#4-methodology--pipeline)
  - [4.1. Exploratory Data Analysis (EDA)](#41-exploratory-data-analysis-eda)
  - [4.2. Data Splitting & Feature Scaling](#42-data-splitting--feature-scaling)
  - [4.3. Model Training & Validation Comparison](#43-model-training--validation-comparison)
  - [4.4. Final Model Retraining & Test Evaluation](#44-final-model-retraining--test-evaluation)
- [5. Installation & Usage](#5-installation--usage)
- [6. Future Improvements](#6-future-improvements)

---

## 1. Problem Overview

In real-world banking and financial systems, fraudulent transactions make up only a tiny fraction of daily transaction volume. Because of this extreme class imbalance:
- **Accuracy** alone is misleading (a naive classifier predicting all transactions as valid would still achieve >99.8% accuracy).
- The key evaluation metrics are **Precision** (minimizing false alarms for legitimate customers), **Recall** (catching as many fraudulent transactions as possible to prevent financial loss), and **F1-Score** (balancing Precision and Recall).

---

## 2. Dataset

> **Note:** The raw dataset file `creditcard.csv` (~144 MB) exceeds GitHub's 100 MB file size limit and is excluded from this repository via `.gitignore`. Please download it separately before running the notebook.

- **Source:** [Credit Card Fraud Detection – Kaggle (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dimensions:** `284,807` rows × `31` columns (`30` input features + `1` target label `Class`).
- **Feature Description:**
  - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset (`0` to `172,792` seconds, ~48 hours).
  - `V1` – `V28`: 28 anonymized numerical features obtained via PCA (*Principal Component Analysis*) to protect user confidentiality.
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
├── creditcard.ipynb    # Jupyter Notebook containing EDA, training, and evaluation
├── README.md           # Vietnamese documentation
├── README.en.md        # English documentation
├── .gitignore          # Excludes large dataset files (*.csv) and Python/Jupyter artifacts
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

   > **Key Observation:** Fraudulent transactions have a noticeably higher mean amount (`122.21`) than valid transactions (`88.29`), even though their median amount is lower (`9.25` vs `22.00`).

2. **Correlation Analysis (`Correlation Heatmap`):**
   - Visualized with `seaborn.heatmap(data.corr(), vmax=.8, square=True)`.
   - Most PCA components (`V1`–`V28`) exhibit minimal correlation with one another, while features such as `V2` and `V5` show noticeable negative correlation with `Amount`.

---

### 4.2. Data Splitting & Feature Scaling

The dataset is split into **60% Train – 20% Validation – 20% Test** using `train_test_split` (`random_state=42`):

| Split | Ratio | Sample Count |
| :--- | :---: | :---: |
| **Training Set** (`X_train`, `Y_train`) | 60% | 170,883 |
| **Validation Set** (`x_val`, `y_val`) | 20% | 56,962 *(56,884 valid; 78 fraud)* |
| **Test Set** (`x_test`, `y_test`) | 20% | 56,962 *(56,864 valid; 98 fraud)* |

- Input features are standardized using `StandardScaler` (`fit_transform` on the training set, `transform` on the validation set).

---

### 4.3. Model Training & Validation Comparison

Two classification models were trained on the training set and evaluated on the validation set (`56,962` transactions, including `78` fraud cases):

- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Logistic Regression:** `LogisticRegression(max_iter=1000, random_state=42)`

#### Validation Set Comparison

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | TP (Fraud Detected) | FN (Fraud Missed) | FP (False Alarms) | TN (Valid Correct) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9996** | **0.9661** | **0.7308** | **0.8321** | **57** | **21** | **2** | **56,882** |
| Logistic Regression | 0.9993 | 0.8364 | 0.5897 | 0.6917 | 46 | 32 | 9 | 56,875 |

#### Model Selection
- **Random Forest** outperformed **Logistic Regression** across all four metrics.
- Specifically, Random Forest detected **57 out of 78** fraudulent transactions (**11 more** than Logistic Regression), lowered missed fraud cases from `32` to `21`, and reduced false alarms from `9` to `2`.
- Therefore, **Random Forest** was selected as the final model architecture.

---

### 4.4. Final Model Retraining & Test Evaluation

After selecting Random Forest, `final_model = RandomForestClassifier(n_estimators=100, random_state=42)` was retrained on the combined **Train + Validation** sets (80% of the data) and evaluated on the held-out **Test** set (`56,962` samples: `56,864` valid and `98` fraud):

| Evaluation Metric | Test Set Score |
| :--- | :---: |
| **Accuracy** | **0.9994** (99.94%) |
| **Precision (Class 1)** | **0.8916** (89.16%) |
| **Recall (Class 1)** | **0.7551** (75.51%) |
| **F1-Score (Class 1)** | **0.8177** (81.77%) |

**Test Set Confusion Matrix:**

```text
                Predicted: Valid (0)    Predicted: Fraud (1)
Actual: Valid (0)      56,855                   9
Actual: Fraud (1)          24                  74
```

- The final model correctly identified **74 out of 98** fraudulent transactions (`Recall = 75.51%`, up from `73.08%` on validation) while raising only `9` false alarms out of `56,864` valid transactions.

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

Potential next steps to further improve detection performance and engineering rigor:
- **Consistent Scaling & Stratified Splitting:** Pass `stratify=ydata` into `train_test_split` to preserve the fraud ratio across splits, and wrap `StandardScaler` inside a `sklearn.pipeline.Pipeline` (combining raw `X_train` and `x_val` before fitting `scaler_final`).
- **Imbalanced Learning Techniques:** Experiment with **SMOTE** (*Synthetic Minority Over-sampling Technique*), **Random Undersampling**, or `class_weight="balanced"`.
- **Additional Models & Metrics:** Benchmark gradient-boosted trees (**XGBoost**, **LightGBM**), evaluate using **PR-AUC** (*Precision-Recall AUC*) / **ROC-AUC**, and tune the classification threshold based on business cost trade-offs.
