# Credit Card Fraud Detection

> 🌐 **Ngôn ngữ / Language:** **Tiếng Việt** | [English (README.en.md)](README.en.md)

Dự án xây dựng hệ thống Học máy (**Machine Learning**) phát hiện giao dịch thẻ tín dụng gian lận (**Credit Card Fraud Detection**) trên bộ dữ liệu thực tế có độ mất cân bằng lớp rất cao (*highly imbalanced dataset*).

Toàn bộ quy trình trong [creditcard.ipynb](creditcard.ipynb) bao gồm: Khám phá dữ liệu (**EDA**), chuẩn hóa đặc trưng (**StandardScaler**), so sánh mô hình cơ sở (**Random Forest** vs **Logistic Regression**), tinh chỉnh siêu tham số (**GridSearchCV**), tối ưu ngưỡng phân loại theo đường cong **Precision–Recall** và đánh giá toàn diện trên tập Test độc lập với các chỉ số **Accuracy, Precision, Recall, F1-Score, PR-AUC, ROC-AUC**.

---

## Mục lục

- [1. Tổng quan bài toán](#1-tổng-quan-bài-toán)
- [2. Bộ dữ liệu (Dataset)](#2-bộ-dữ-liệu-dataset)
- [3. Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
- [4. Quy trình thực hiện](#4-quy-trình-thực-hiện)
  - [4.1. Khám phá dữ liệu (EDA)](#41-khám-phá-dữ-liệu-eda)
  - [4.2. Chia tập dữ liệu & Chuẩn hóa đặc trưng](#42-chia-tập-dữ-liệu--chuẩn-hóa-đặc-trưng)
  - [4.3. So sánh mô hình cơ sở (Validation Set)](#43-so-sánh-mô-hình-cơ-sở-validation-set)
  - [4.4. Tinh chỉnh siêu tham số với GridSearchCV](#44-tinh-chỉnh-siêu-tham-số-với-gridsearchcv)
  - [4.5. Tối ưu ngưỡng quyết định (Threshold Optimization)](#45-tối-ưu-ngưỡng-quyết-định-threshold-optimization)
  - [4.6. Huấn luyện mô hình cuối cùng & Đánh giá (Test Set)](#46-huấn-luyện-mô-hình-cuối-cùng--đánh-giá-test-set)
- [5. Cài đặt và Hướng dẫn chạy](#5-cài-đặt-và-hướng-dẫn-chạy)
- [6. Hướng cải tiến trong tương lai](#6-hướng-cải-tiến-trong-tương-lai)

---

## 1. Tổng quan bài toán

Trong các hệ thống tài chính – ngân hàng, giao dịch gian lận chỉ chiếm một tỷ lệ rất nhỏ so với lượng giao dịch hợp lệ mỗi ngày. Vì vậy:
- **Accuracy** đơn thuần không phản ánh đúng hiệu năng (dự đoán toàn bộ là hợp lệ vẫn đạt >99.8% Accuracy).
- Các chỉ số cốt lõi cần tập trung gồm **Precision** (hạn chế cảnh báo nhầm làm phiền khách hàng), **Recall** (phát hiện tối đa giao dịch gian lận để giảm thiểu tổn thất tài chính), **F1-Score**, cùng hai thước đo không phụ thuộc ngưỡng là **PR-AUC** (*Precision-Recall AUC*) và **ROC-AUC**.

---

## 2. Bộ dữ liệu (Dataset)

> **Lưu ý:** File dữ liệu `creditcard.csv` (~144 MB) vượt giới hạn 100 MB của GitHub nên được loại trừ khỏi repository qua `.gitignore`. Bạn cần tải riêng file này về thư mục gốc trước khi chạy notebook.

- **Nguồn dữ liệu:** [Credit Card Fraud Detection – Kaggle (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Kích thước:** `284,807` dòng × `31` cột (`30` đặc trưng đầu vào + `1` cột nhãn `Class`).
- **Mô tả các cột:**
  - `Time`: Số giây trôi qua kể từ giao dịch đầu tiên (`0` đến `172,792` giây, tương đương ~48 giờ).
  - `V1` – `V28`: 28 đặc trưng số đã được biến đổi bằng PCA (*Principal Component Analysis*) nhằm ẩn danh thông tin khách hàng.
  - `Amount`: Số tiền của giao dịch.
  - `Class`: Biến mục tiêu nhị phân (`0` = Giao dịch hợp lệ, `1` = Giao dịch gian lận).

### Phân bố nhãn (Class Imbalance)

| Loại giao dịch | Nhãn (`Class`) | Số lượng | Tỷ lệ (%) |
| :--- | :---: | :---: | :---: |
| Giao dịch hợp lệ (*Valid*) | `0` | 284,315 | 99.8273% |
| Giao dịch gian lận (*Fraud*) | `1` | 492 | 0.1727% |
| **Tổng cộng** | — | **284,807** | **100.0000%** |

---

## 3. Cấu trúc thư mục

```text
Credit-Card-Fraud-Detection/
├── creditcard.ipynb    # Jupyter Notebook chứa toàn bộ code EDA, tuning, tối ưu threshold & đánh giá
├── README.md           # Tài liệu mô tả chi tiết dự án (Tiếng Việt)
├── README.en.md        # Tài liệu mô tả chi tiết dự án (Tiếng Anh / English)
├── .gitignore          # Loại trừ file dữ liệu lớn (*.csv) và các file/thư mục cục bộ
└── creditcard.csv      # (Tải riêng từ Kaggle, đặt cùng thư mục gốc khi chạy)
```

---

## 4. Quy trình thực hiện

### 4.1. Khám phá dữ liệu (EDA)

1. **Thống kê mô tả số tiền giao dịch (`Amount`):**

   | Thống kê | Giao dịch gian lận (`Class = 1`) | Giao dịch hợp lệ (`Class = 0`) |
   | :--- | :---: | :---: |
   | Số lượng (`count`) | 492 | 284,315 |
   | Trung bình (`mean`) | **122.21** | 88.29 |
   | Độ lệch chuẩn (`std`) | 256.68 | 250.11 |
   | Nhỏ nhất (`min`) | 0.00 | 0.00 |
   | Phân vị 25% | 1.00 | 5.65 |
   | Trung vị (`50%`) | 9.25 | 22.00 |
   | Phân vị 75% | 105.89 | 77.05 |
   | Lớn nhất (`max`) | 2,125.87 | 25,691.16 |

   > **Nhận xét:** Giao dịch gian lận có giá trị trung bình (`122.21`) cao hơn giao dịch hợp lệ (`88.29`), dù trung vị thấp hơn (`9.25` so với `22.00`).

2. **Phân tích tương quan (`Correlation Heatmap`):**
   - Trực quan hóa ma trận tương quan bằng `seaborn.heatmap(data.corr(), vmax=.8, square=True)`.
   - Phần lớn các thành phần PCA (`V1`–`V28`) ít tương quan với nhau; một số đặc trưng như `V2` và `V5` có tương quan âm đáng kể với `Amount`.

---

### 4.2. Chia tập dữ liệu & Chuẩn hóa đặc trưng

Dữ liệu được chia thành 3 tập theo tỷ lệ **60% Train – 20% Validation – 20% Test** bằng `train_test_split` (`random_state=42`):

| Tập dữ liệu | Tỷ lệ | Số lượng mẫu |
| :--- | :---: | :---: |
| **Training Set** (`X_train`, `Y_train`) | 60% | 170,883 |
| **Validation Set** (`x_val`, `y_val`) | 20% | 56,962 *(56,884 hợp lệ; 78 gian lận)* |
| **Test Set** (`x_test`, `y_test`) | 20% | 56,962 *(56,864 hợp lệ; 98 gian lận)* |

- Ở bước so sánh mô hình cơ sở, dữ liệu đầu vào được chuẩn hóa bằng `StandardScaler` (`fit_transform` trên `X_train`, `transform` trên `x_val`).

---

### 4.3. So sánh mô hình cơ sở (Validation Set)

Hai mô hình cơ sở được huấn luyện trên `X_train_scale` và đánh giá trên `X_val_scale`:
- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Logistic Regression:** `LogisticRegression(max_iter=1000, random_state=42)`

| Mô hình | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | TP | FN | FP | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9996** | **0.9661** | **0.7308** | **0.8321** | **57** | **21** | **2** | **56,882** |
| Logistic Regression | 0.9993 | 0.8364 | 0.5897 | 0.6917 | 46 | 32 | 9 | 56,875 |

**Random Forest** vượt trội trên cả 4 chỉ số (phát hiện 57/78 ca gian lận so với 46/78 của Logistic Regression, đồng thời giảm cảnh báo nhầm từ 9 xuống 2), nên được chọn để tối ưu hóa ở các bước tiếp theo.

---

### 4.4. Tinh chỉnh siêu tham số với GridSearchCV

Sử dụng `GridSearchCV` (3-fold cross-validation, tối ưu theo `scoring='roc_auc'`) trên tập Training gốc (`X_train`, `Y_train`) với không gian tìm kiếm:
- `n_estimators`: `[50, 100]`
- `max_depth`: `[None, 10, 20]`

Cấu hình tốt nhất được chọn là **`max_depth=10`, `n_estimators=100`** (`random_state=42`). Khi đánh giá `best_model` trên Validation Set tại ngưỡng mặc định `0.5`:
- **Accuracy:** `0.9996` | **Precision:** `0.9667` | **Recall:** `0.7436` | **F1-Score:** `0.8406`
- **Confusion Matrix:** `[[56882, 2], [20, 58]]` (tăng lên 58/78 giao dịch gian lận phát hiện đúng).

---

### 4.5. Tối ưu ngưỡng quyết định (Threshold Optimization)

Thay vì dùng ngưỡng phân loại mặc định `0.5`, notebook trích xuất xác suất dự đoán `predict_proba(x_val)[:, 1]`, vẽ biểu đồ **Precision and Recall vs Threshold** bằng `precision_recall_curve`, và chọn `best_threshold` sao cho **F1-Score trên Validation Set đạt giá trị cao nhất**.

#### Bảng tiến triển hiệu năng trên Validation Set

| Phiên bản mô hình (Validation Set) | Accuracy | Precision | Recall | F1-Score | TP / 78 | FN (Bỏ sót) | FP (Báo nhầm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Random Forest | 0.9996 | 0.9661 | 0.7308 | 0.8321 | 57 | 21 | 2 |
| GridSearchCV Best RF (`threshold = 0.5`) | 0.9996 | **0.9667** | 0.7436 | 0.8406 | 58 | 20 | **2** |
| **GridSearchCV Best RF + Optimal Threshold** | **0.9997** | 0.9538 | **0.7949** | **0.8671** | **62** | **16** | 3 |

Việc tối ưu threshold giúp Recall trên tập Validation tăng từ `74.36%` lên **`79.49%`** và F1-Score tăng từ `84.06%` lên **`86.71%`**.

---

### 4.6. Huấn luyện mô hình cuối cùng & Đánh giá (Test Set)

Mô hình tốt nhất (`RandomForestClassifier(**search.best_params_, random_state=42)`, tức `max_depth=10, n_estimators=100`) được huấn luyện lại trên tập dữ liệu gộp **Train + Validation** (`X_train_final`, `Y_train_final` – 80% dữ liệu) và áp dụng `best_threshold` lên tập **Test** (`56,962` mẫu gồm `56,864` hợp lệ và `98` gian lận):

| Chỉ số đánh giá | Kết quả trên Test Set |
| :--- | :---: |
| **Accuracy** | **0.9996** (99.96%) |
| **Precision (Class 1)** | **0.9398** (93.98%) |
| **Recall (Class 1)** | **0.7959** (79.59%) |
| **F1-Score (Class 1)** | **0.8619** (86.19%) |
| **PR-AUC** (`average_precision_score`) | **0.8572** |
| **ROC-AUC** (`roc_auc_score`) | **0.9773** |

**Confusion Matrix trên Test Set:**

```text
                Dự đoán: Valid (0)    Dự đoán: Fraud (1)
Thực tế: Valid (0)     56,859                 5
Thực tế: Fraud (1)         20                78
```

- Mô hình cuối cùng phát hiện chính xác **78/98** giao dịch gian lận (`Recall = 79.59%`) với độ chính xác dự báo gian lận rất cao (**`93.98%` Precision**, chỉ `5` cảnh báo nhầm trên `56,864` giao dịch hợp lệ).

---

## 5. Cài đặt và Hướng dẫn chạy

### Yêu cầu môi trường
- Python `>= 3.8`
- Các thư viện: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

### Các bước thực hiện

1. **Clone repository:**
   ```bash
   git clone https://github.com/Hoanglovecode/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Cài đặt thư viện cần thiết:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Tải bộ dữ liệu:**
   - Tải file `creditcard.csv` từ [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
   - Giải nén và đặt `creditcard.csv` vào thư mục gốc của dự án (cùng cấp với `creditcard.ipynb`).

4. **Mở và chạy Notebook:**
   ```bash
   jupyter notebook creditcard.ipynb
   ```
   Sau đó chọn **Run All** để thực thi toàn bộ các cell.

---

## 6. Hướng cải tiến trong tương lai

- **Chia dữ liệu phân tầng & Pipeline chuẩn hóa:** Sử dụng `stratify=ydata` trong `train_test_split` để bảo toàn tỷ lệ gian lận ở từng tập, đồng thời đóng gói tiền xử lý vào `sklearn.pipeline.Pipeline` khi chạy `GridSearchCV`.
- **Xử lý mất cân bằng dữ liệu:** Thử nghiệm **SMOTE**, **Random Undersampling**, hoặc `class_weight="balanced"` kết hợp với đánh giá `StratifiedKFold`.
- **Mở rộng mô hình & triển khai:** So sánh thêm **XGBoost**, **LightGBM**, **CatBoost**, giải thích mô hình bằng **SHAP**, và đóng gói mô hình thành API dự đoán thời gian thực (FastAPI / Streamlit).
