# Credit Card Fraud Detection

Dự án xây dựng mô hình Học máy (**Machine Learning**) nhằm phát hiện giao dịch thẻ tín dụng gian lận (**Credit Card Fraud Detection**) trên bộ dữ liệu giao dịch thực tế có độ mất cân bằng lớp rất cao (*highly imbalanced dataset*).

Dự án thực hiện toàn bộ quy trình từ khám phá dữ liệu (**EDA**), tiền xử lý (**StandardScaler**), huấn luyện và so sánh hai mô hình (**Random Forest** và **Logistic Regression**) trên tập Validation, đến huấn luyện lại mô hình tốt nhất trên tập dữ liệu gộp (`Train + Validation`) và đánh giá khách quan trên tập Test độc lập.

---

## Mục lục

- [1. Tổng quan bài toán](#1-tổng-quan-bài-toán)
- [2. Bộ dữ liệu (Dataset)](#2-bộ-dữ-liệu-dataset)
- [3. Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
- [4. Quy trình thực hiện](#4-quy-trình-thực-hiện)
  - [4.1. Khám phá dữ liệu (EDA)](#41-khám-phá-dữ-liệu-eda)
  - [4.2. Chia tập dữ liệu & Chuẩn hóa đặc trưng](#42-chia-tập-dữ-liệu--chuẩn-hóa-đặc-trưng)
  - [4.3. Huấn luyện & So sánh mô hình (Validation Set)](#43-huấn-luyện--so-sánh-mô-hình-validation-set)
  - [4.4. Huấn luyện mô hình cuối cùng & Đánh giá (Test Set)](#44-huấn-luyện-mô-hình-cuối-cùng--đánh-giá-test-set)
- [5. Cài đặt và Hướng dẫn chạy](#5-cài-đặt-và-hướng-dẫn-chạy)
- [6. Hướng cải tiến trong tương lai](#6-hướng-cải-tiến-trong-tương-lai)

---

## 1. Tổng quan bài toán

Trong thực tế tài chính – ngân hàng, các giao dịch gian lận chỉ chiếm một tỷ lệ cực nhỏ so với hàng trăm nghìn giao dịch hợp lệ mỗi ngày. Vì vậy:
- Chỉ số **Accuracy** đơn thuần không phản ánh đúng chất lượng mô hình (một mô hình dự đoán toàn bộ là hợp lệ vẫn đạt >99.8% Accuracy).
- Các chỉ số trọng tâm cần tối ưu là **Precision** (giảm cảnh báo nhầm cho khách hàng), **Recall** (giảm tối đa số giao dịch gian lận bị bỏ sót – tránh thiệt hại tài chính) và **F1-Score** (cân bằng giữa Precision và Recall).

---

## 2. Bộ dữ liệu (Dataset)

> **Lưu ý:** File dữ liệu `creditcard.csv` có dung lượng ~144 MB (vượt giới hạn 100 MB của GitHub) nên được loại trừ khỏi repository thông qua `.gitignore`. Bạn cần tải dữ liệu về máy trước khi chạy notebook.

- **Nguồn dữ liệu:** [Credit Card Fraud Detection – Kaggle (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Kích thước:** `284,807` dòng × `31` cột (`30` đặc trưng đầu vào + `1` cột nhãn `Class`).
- **Mô tả các cột:**
  - `Time`: Số giây trôi qua kể từ giao dịch đầu tiên trong tập dữ liệu (từ `0` đến `172,792` giây, tương đương ~48 giờ).
  - `V1` – `V28`: 28 đặc trưng số đã được biến đổi bằng PCA (*Principal Component Analysis*) để bảo mật thông tin người dùng.
  - `Amount`: Giá trị số tiền của giao dịch.
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
├── creditcard.ipynb    # Jupyter Notebook chứa toàn bộ code EDA, huấn luyện & đánh giá
├── README.md           # Tài liệu mô tả chi tiết dự án
├── .gitignore          # Loại trừ file dữ liệu lớn (*.csv) và các file tạm của Python/Jupyter
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

   > **Nhận xét:** Giao dịch gian lận có giá trị trung bình (`122.21`) cao hơn đáng kể so với giao dịch hợp lệ (`88.29`), dù trung vị lại thấp hơn (`9.25` so với `22.00`).

2. **Phân tích tương quan (`Correlation Heatmap`):**
   - Trực quan hóa ma trận tương quan bằng `seaborn.heatmap(data.corr(), vmax=.8, square=True)`.
   - Phần lớn các đặc trưng PCA (`V1`–`V28`) độc lập tuyến tính với nhau; một số đặc trưng như `V2` và `V5` có tương quan âm rõ rệt với `Amount`.

---

### 4.2. Chia tập dữ liệu & Chuẩn hóa đặc trưng

Dữ liệu được chia thành 3 tập theo tỷ lệ **60% Train – 20% Validation – 20% Test** bằng `train_test_split` với `random_state=42`:

| Tập dữ liệu | Tỷ lệ | Số lượng mẫu |
| :--- | :---: | :---: |
| **Training Set** (`X_train`, `Y_train`) | 60% | 170,883 |
| **Validation Set** (`x_val`, `y_val`) | 20% | 56,962 *(56,884 hợp lệ; 78 gian lận)* |
| **Test Set** (`x_test`, `y_test`) | 20% | 56,962 *(56,864 hợp lệ; 98 gian lận)* |

- Các đặc trưng đầu vào được chuẩn hóa bằng `StandardScaler` (`fit_transform` trên tập Train, `transform` trên tập Validation).

---

### 4.3. Huấn luyện & So sánh mô hình (Validation Set)

Hai mô hình được huấn luyện trên tập Training và đánh giá trên tập Validation (`56,962` mẫu, trong đó có `78` giao dịch gian lận):

- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Logistic Regression:** `LogisticRegression(max_iter=1000, random_state=42)`

#### Bảng so sánh kết quả trên Validation Set

| Mô hình | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | TP (Fraud đúng) | FN (Fraud sót) | FP (Báo nhầm) | TN (Valid đúng) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9996** | **0.9661** | **0.7308** | **0.8321** | **57** | **21** | **2** | **56,882** |
| Logistic Regression | 0.9993 | 0.8364 | 0.5897 | 0.6917 | 46 | 32 | 9 | 56,875 |

#### Kết luận lựa chọn mô hình
- **Random Forest** vượt trội hơn **Logistic Regression** trên toàn bộ 4 chỉ số đánh giá.
- Cụ thể, Random Forest phát hiện đúng **57/78** giao dịch gian lận (nhiều hơn **11** ca so với Logistic Regression), giảm số ca bỏ sót từ `32` xuống `21` và giảm cảnh báo nhầm từ `9` xuống `2`.
- Vì vậy, **Random Forest** được chọn làm mô hình chính cho bước đánh giá cuối cùng.

---

### 4.4. Huấn luyện mô hình cuối cùng & Đánh giá (Test Set)

Sau khi chọn Random Forest, mô hình `final_model = RandomForestClassifier(n_estimators=100, random_state=42)` được huấn luyện lại trên tập dữ liệu gộp từ **Train + Validation** (80% dữ liệu) và kiểm thử trên tập **Test** (`56,962` mẫu, gồm `56,864` hợp lệ và `98` gian lận):

| Chỉ số đánh giá | Giá trị trên Test Set |
| :--- | :---: |
| **Accuracy** | **0.9994** (99.94%) |
| **Precision (Class 1)** | **0.8916** (89.16%) |
| **Recall (Class 1)** | **0.7551** (75.51%) |
| **F1-Score (Class 1)** | **0.8177** (81.77%) |

**Confusion Matrix trên Test Set:**

```text
                Dự đoán: Valid (0)    Dự đoán: Fraud (1)
Thực tế: Valid (0)     56,855                 9
Thực tế: Fraud (1)         24                74
```

- Mô hình phát hiện chính xác **74/98** giao dịch gian lận (`Recall = 75.51%`, tăng so với `73.08%` trên Validation), chỉ báo nhầm `9` trong tổng số `56,864` giao dịch hợp lệ.

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
   Sau đó chọn **Run All** để chạy toàn bộ các cell từ đầu đến cuối.

---

## 6. Hướng cải tiến trong tương lai

Để nâng cao hơn nữa hiệu suất phát hiện gian lận và chuẩn hóa quy trình huấn luyện, có thể áp dụng các cải tiến sau:
- **Chuẩn hóa Pipeline & Chia dữ liệu phân tầng:** Dùng `stratify=ydata` trong `train_test_split` để giữ nguyên tỷ lệ gian lận ở các tập, và đóng gói `StandardScaler` vào `sklearn.pipeline.Pipeline` (tránh scale hai lần khi gộp `X_train` và `x_val`).
- **Xử lý mất cân bằng dữ liệu:** Thử nghiệm kỹ thuật **SMOTE** (*Synthetic Minority Over-sampling Technique*), **Random Undersampling**, hoặc tham số `class_weight="balanced"`.
- **Thử nghiệm thêm thuật toán & tối ưu ngưỡng:** So sánh thêm với **XGBoost**, **LightGBM**, đánh giá bằng **PR-AUC** (*Precision-Recall AUC*) / **ROC-AUC**, và tinh chỉnh ngưỡng xác suất (*decision threshold*) theo bài toán chi phí thực tế.
