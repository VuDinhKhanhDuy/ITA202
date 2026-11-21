# Assignment Giai đoạn 1: Chuẩn bị dữ liệu và Xây dựng mô hình Cơ sở
**Môn học: Học máy cơ bản (ITA202) - FPT Polytechnic**

Thư mục này chứa đầy đủ mã nguồn, dữ liệu, biểu đồ và báo cáo cho **Giai đoạn 1** của bài Assignment theo đúng yêu cầu đề bài:
* **Mục tiêu:** Hoàn thành quy trình ML cơ bản từ dữ liệu thô đến mô hình dự đoán đầu tiên (Linear Regression).

---

## 📁 Cấu trúc thư mục Assignment 1

```text
Assignment_1/
├── data/
│   └── housing.csv                               # Dữ liệu thực tế California Housing
├── images/                                       # Đồ thị EDA và đánh giá mô hình cơ sở
│   ├── eda_target_distribution.png
│   ├── eda_geographical_scatter.png
│   ├── eda_correlation_heatmap.png
│   ├── linear_regression_actual_vs_predicted.png
│   └── linear_regression_residuals.png
├── assignment_1_machine_learning.ipynb           # Jupyter Notebook thực thi 5 bước Giai đoạn 1
├── main_giai_doan_1.py                           # Python script chạy tự động toàn bộ Giai đoạn 1
├── BAO_CAO_GIAI_DOAN_1.md                        # Báo cáo ngắn về dữ liệu & kết quả mô hình cơ sở
└── README.md                                     # Tài liệu hướng dẫn
```

---

## 🚀 Hướng dẫn chạy

1. **Cài đặt thư viện (nếu chưa có):**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

2. **Chạy script Python tự động:**
   ```bash
   python main_giai_doan_1.py
   ```

3. **Chạy Jupyter Notebook:**
   Mở file `assignment_1_machine_learning.ipynb` trong VS Code hoặc Jupyter Notebook và nhấn **Run All**.

---

## 📊 Tóm tắt kết quả Mô hình Cơ sở (Giai đoạn 1)

* **Mô hình:** Hồi quy tuyến tính (Linear Regression)
* **Tập Train (80%):** $RMSE = \$67,593.21$ | $R^2 = 0.6582$
* **Tập Test (20%):** $RMSE = \$69,127.04$ | $R^2 = 0.6353$ | $MAE = \$49,645.49$
* **Nhận xét:** Mô hình đang gặp hiện tượng **Underfitting (Thiếu khớp)** do tính phi tuyến của dữ liệu; đây là tiền đề cơ sở để triển khai các mô hình nâng cao (Decision Tree, Random Forest) và GridSearchCV trong **Giai đoạn 2**.
