# BÁO CÁO ASSIGNMENT GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU VÀ XÂY DỰNG MÔ HÌNH CƠ SỞ
**TRƯỜNG CAO ĐẲNG FPT POLYTECHNIC**  
**MÔN HỌC: HỌC MÁY CƠ BẢN (ITA202)**

---

* **Mã môn học:** ITA202 (Học máy cơ bản)
* **Số hiệu Assignment:** 1/1 (Giai đoạn 1: Chuẩn bị dữ liệu và Mô hình cơ sở)
* **Người điều phối:** Lã Ngọc Quang
* **Tên đề tài:** **Dự đoán giá nhà California (California Housing Price Prediction)**
* **Sinh viên thực hiện:** Vũ Đình Khánh Duy
* **Email:** vukhanh25519@gmail.com
* **Mục tiêu Giai đoạn 1:** Hoàn thành quy trình ML cơ bản từ dữ liệu thô đến mô hình dự đoán đầu tiên (Linear Regression), đánh giá ban đầu và chuẩn bị dữ liệu chuẩn hóa sẵn sàng cho Giai đoạn 2.

---

## 1. CHỌN VÀ KHÁM PHÁ DỮ LIỆU (EDA)

### 1.1 Chọn tập dữ liệu
Tập dữ liệu được lựa chọn cho bài toán là **California Housing**, tập dữ liệu thực tế thu thập từ điều tra dân số tại các khu vực thuộc bang California (Hoa Kỳ).
* **Bài toán:** Hồi quy (Regression)
* **Biến mục tiêu ($y$):** `median_house_value` (Giá trị nhà trung vị tại khu vực, đơn vị USD).

### 1.2 Khám phá cấu trúc dữ liệu
* **Kích thước dữ liệu (Shape):** Gồm **20,640 dòng** (mẫu quan sát) và **10 cột** (thuộc tính).
* **Kiểu dữ liệu:**
  - 9 cột thuộc tính số (`float64`): `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`.
  - 1 cột phân loại dạng văn bản (`object`): `ocean_proximity` (vị trí tương đối so với biển).

### 1.3 Thống kê mô tả (Describe)
* Giá nhà trung vị (`median_house_value`):
  - Giá nhỏ nhất: **\$14,999**
  - Giá trung bình (Mean): **\$206,855.82**
  - Giá trung vị (Median / 50%): **\$179,700.00**
  - Giá lớn nhất: **\$500,001.00**
  - Độ lệch chuẩn (Std): **\$115,395.62**
* Thu nhập trung vị (`median_income`): Dao động từ $0.5$ đến $15.0$ (tương đương \$5,000 đến \$150,000/năm), trung bình là $3.87$ (~ \$38,700/năm).

### 1.4 Kiểm tra giá trị khuyết thiếu (Missing Values)
* Cột `total_bedrooms` có **207 giá trị khuyết thiếu (NaN)**, chiếm tỷ lệ khoảng **1.00%**.
* Các cột còn lại đều có đầy đủ 20,640 giá trị hợp lệ.

### 1.5 Trực quan hóa dữ liệu ban đầu
1. **Phân phối biến mục tiêu (`images/eda_target_distribution.png`):**
   - Phân phối giá nhà bị lệch phải (Right-skewed) với đỉnh nhọn tại mốc \$500,001 do dữ liệu điều tra đã được chặn trần (Capped).
2. **Bản đồ địa lý theo tọa độ (`images/eda_geographical_scatter.png`):**
   - Khu vực ven biển phía Tây và các cụm đô thị vịnh San Francisco, Los Angeles có giá nhà cao vượt trội và dân cư tập trung đông đúc hơn hẳn các vùng sâu trong nội địa.
3. **Ma trận tương quan (`images/eda_correlation_heatmap.png`):**
   - Thuộc tính `median_income` có tương quan thuận mạnh nhất với giá nhà ($r = +0.69$).

---

## 2. LÀM SẠCH DỮ LIỆU & KỸ NGHỆ ĐẶC TRƯNG

### 2.1 Xử lý giá trị khuyết thiếu (NaN)
* Để không làm mất 207 mẫu quan sát thực tế, ta không dùng phương án xóa bỏ hàng (Drop).
* Áp dụng phương pháp điền khuyết thiếu bằng **Trung vị (Median Imputation)**:
  $$\text{Median}(\text{total\_bedrooms}) = 435.0$$
* *Lý do:* Phân phối số phòng ngủ bị lệch phải, việc dùng Trung vị (Median) giúp tránh bị ảnh hưởng bởi các giá trị ngoại lệ cực lớn so với dùng Trung bình (Mean).

### 2.2 Xử lý giá trị ngoại lệ & Kỹ nghệ đặc trưng (Feature Engineering)
Tổng số phòng và tổng số phòng ngủ của một khu vực rộng lớn không phản ánh trực quan quy mô của từng ngôi nhà. Vì vậy, ta tạo ra 3 đặc trưng tỷ lệ mang giá trị phân loại cao:
1. `rooms_per_household = total_rooms / households`: Số phòng trung bình trong một căn nhà của mỗi hộ.
2. `bedrooms_per_room = total_bedrooms / total_rooms`: Tỷ lệ phòng ngủ trên tổng số phòng (tương quan âm $r = -0.26$ với giá nhà).
3. `population_per_household = population / households`: Số lượng người trung bình trong một hộ gia đình.

---

## 3. TIỀN XỬ LÝ DỮ LIỆU & CHIA TẬP TRAIN / TEST

### 3.1 Phân chia tập dữ liệu
* Chia tập dữ liệu theo đúng tỷ lệ chuẩn của đề bài: **80% Train** và **20% Test** với `random_state=42` để đảm bảo kết quả tái lập:
  - **Tập Train (80%):** $16,512$ mẫu
  - **Tập Test (20%):** $4,128$ mẫu

### 3.2 Chuẩn hóa đặc trưng & Mã hóa dữ liệu
* Sử dụng `ColumnTransformer` tích hợp:
  - **Nhóm đặc trưng số:** `SimpleImputer(strategy='median')` kết hợp `StandardScaler` đưa các biến về phân phối chuẩn có $\mu = 0$ và $\sigma = 1$.
  - **Nhóm đặc trưng phân loại (`ocean_proximity`):** `OneHotEncoder(handle_unknown='ignore')` để biến đổi 5 nhóm thành các cột nhị phân (0/1).
* **Chống rò rỉ dữ liệu (Data Leakage):** Tiền xử lý chỉ được `fit` trên tập Train, sau đó áp dụng `transform` trên cả tập Train và Test.

---

## 4. HUẤN LUYỆN MÔ HÌNH CƠ SỞ (LINEAR REGRESSION)

Theo yêu cầu của Giai đoạn 1 đối với bài toán Hồi quy, ta áp dụng mô hình **Hồi quy tuyến tính (Linear Regression)** để làm cơ sở đối chuẩn (Baseline Model).

Mô hình được huấn luyện trực tiếp trên tập Train đã tiền xử lý gồm 16 đặc trưng đầu vào.

---

## 5. ĐÁNH GIÁ BAN ĐẦU MÔ HÌNH CƠ SỞ & TRỰC QUAN HÓA

### 5.1 Bảng chỉ số đánh giá thực nghiệm:

| Chỉ số đánh giá | Tập Huấn luyện (Train) | Tập Kiểm thử (Test) | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **MAE (Mean Absolute Error)** | \$49,438.25 | **\$49,645.49** | Sai số tuyệt đối trung bình (\$ USD) |
| **MSE (Mean Squared Error)** | 4,568,841,754.21 | **4,778,547,424.03** | Sai số bình phương trung bình |
| **RMSE (Root Mean Squared Error)**| \$67,593.21 | **\$69,127.04** | Độ lệch chuẩn sai số dự đoán (\$ USD) |
| **$R^2$ Score (Hệ số xác định)** | 0.6582 | **0.6353** | Tỷ lệ phương sai giải thích được (~63.5%) |

### 5.2 Trực quan hóa kết quả:
1. **Biểu đồ Thực tế vs Dự đoán (`images/linear_regression_actual_vs_predicted.png`):**
   - Các điểm dự đoán nằm phân tán dọc theo đường lý tưởng $y = x$, tuy nhiên độ phân tán còn khá rộng, cho thấy sai số còn đáng kể.
2. **Biểu đồ Phân phối sai số phần dư (`images/linear_regression_residuals.png`):**
   - Phần dư $\text{Residual} = y_{true} - y_{pred}$ tập trung quanh mức 0 nhưng đuôi lệch khá dài ở hai phía.

---

## 6. KẾT LUẬN GIAI ĐOẠN 1 & ĐỊNH HƯỚNG GIAI ĐOẠN 2

### 6.1 Nhận xét chuyên môn về hiện tượng Underfitting:
* Mô hình Hồi quy tuyến tính đạt $R^2 \approx 0.6353$ và $RMSE \approx \$69,127$ trên tập kiểm thử. Sai số tuyệt đối MAE khoảng \$49,645 (chiếm ~24% giá trị trung vị của căn nhà).
* **Kết luận:** Mô hình cơ sở đang gặp hiện tượng **Thiếu khớp (Underfitting)**. Do cấu trúc giá nhà trong không gian địa lý có tính phi tuyến tính cao, mô hình tuyến tính đơn giản không đủ năng lực để mô tả trọn vẹn toàn bộ quy luật dữ liệu.

### 6.2 Chuẩn bị cho Giai đoạn 2 (Tối ưu hóa và Hoàn thiện):
1. Thử nghiệm các thuật toán phi tuyến tính nâng cao: **K-Nearest Neighbors (KNN)**, **Decision Tree**, và **Random Forest** (phương pháp Ensemble Bagging).
2. Đóng gói quy trình tiền xử lý và mô hình vào **Scikit-Learn Pipeline** hoàn chỉnh.
3. Ứng dụng **GridSearchCV** kết hợp Cross-Validation để tìm ra bộ siêu tham số tối ưu nhất.
4. Đánh giá và so sánh mô hình nâng cao với mô hình cơ sở ở Giai đoạn 1.
5. Đóng gói mô hình cuối cùng ra file `.pkl` bằng thư viện `joblib`.

---
*Báo cáo hoàn thành Giai đoạn 1 - Assignment Môn Học máy cơ bản (ITA202) - FPT Polytechnic.*
