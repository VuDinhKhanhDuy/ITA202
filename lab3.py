"""
FPT POLYTECHNIC - HỌC MÁY CƠ BẢN
LAB 3: HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)
Bao gồm đầy đủ các bài tập được giao (Bài 1 -> Bài 4)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# %% ===========================================================================
# BÀI 1: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH
# ==============================================================================
print("=" * 60)
print("BÀI 1: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH (DỮ LIỆU GIẢ LẬP)")
print("=" * 60)

# 1. Tạo dữ liệu giả lập có mối quan hệ tuyến tính: y = 2X + 1 + nhiễu
np.random.seed(42)  # Cố định seed để tái lập kết quả
X = 2 * np.random.rand(100, 1)        # 100 điểm dữ liệu trong khoảng [0, 2)
noise = np.random.randn(100, 1) * 0.5 # Nhiễu ngẫu nhiên
y = 2 * X + 1 + noise                 # y = 2X + 1 + nhiễu

# 2. Sử dụng thư viện sklearn.linear_model.LinearRegression để huấn luyện mô hình
model_b1 = LinearRegression()
model_b1.fit(X, y)

# 3. In ra các hệ số của mô hình (coefficient và intercept)
print(f"Hệ số góc (Coefficient / Slope): {model_b1.coef_[0][0]:.4f}")
print(f"Hệ số chặn (Intercept / Bias):    {model_b1.intercept_[0]:.4f}")
print(f"=> Phương trình: y_pred = {model_b1.coef_[0][0]:.4f} * X + {model_b1.intercept_[0]:.4f}\n")


# %% ===========================================================================
# BÀI 2: ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
print("=" * 60)
print("BÀI 2: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP DỮ LIỆU DIABETES")
print("=" * 60)

# 1. Sử dụng tập dữ liệu diabetes (chỉ lấy 1 đặc trưng duy nhất: BMI ở cột index 2)
diabetes = load_diabetes()
X_diab = diabetes.data[:, np.newaxis, 2]
y_diab = diabetes.target

# 2. Chia dữ liệu Train/Test (80% Train, 20% Test)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)

# 3. Huấn luyện mô hình và tính toán chỉ số đánh giá trên tập Test
model_b2 = LinearRegression()
model_b2.fit(X_train_d, y_train_d)
y_pred_d = model_b2.predict(X_test_d)

mae_d = mean_absolute_error(y_test_d, y_pred_d)
mse_d = mean_squared_error(y_test_d, y_pred_d)
r2_d = r2_score(y_test_d, y_pred_d)

print(f"- MAE (Mean Absolute Error):  {mae_d:.4f}")
print(f"- MSE (Mean Squared Error):   {mse_d:.4f}")
print(f"- R-squared (R2 Score):       {r2_d:.4f}")

# 4. GIẢI THÍCH Ý NGHĨA CÁC CHỈ SỐ:
# - MAE (Mean Absolute Error): Độ lệch tuyệt đối trung bình giữa giá trị thực tế và dự đoán.
#   Cùng đơn vị với biến mục tiêu y, MAE càng nhỏ thì mô hình càng chính xác và ít bị ảnh hưởng bởi outlier.
# - MSE (Mean Squared Error): Trung bình của bình phương các sai số. Do có bình phương nên MSE phạt
#   rất nặng các sai số lớn (outlier), MSE càng nhỏ chứng tỏ chất lượng dự đoán càng tốt.
# - R-squared (R2): Đo lường tỷ lệ phần trăm phương sai của biến mục tiêu được giải thích bởi mô hình.
#   R2 càng gần 1 thì mô hình càng khớp tốt với dữ liệu thực tế.


# %% ===========================================================================
# BÀI 3: TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================
print("\n" + "=" * 60)
print("BÀI 3: TRỰC QUAN HÓA KẾT QUẢ (SCATTER PLOT & ĐƯỜNG DỰ ĐOÁN)")
print("=" * 60)

# Vẽ biểu đồ tán xạ dữ liệu thực tế và đường thẳng hồi quy của Bài 2
plt.figure(figsize=(8, 5))
plt.scatter(X_test_d, y_test_d, color='black', alpha=0.7, label='Dữ liệu thực tế (Test set)')
plt.plot(X_test_d, y_pred_d, color='blue', linewidth=2, label='Đường thẳng hồi quy dự đoán')

plt.title('Bài 3: Trực quan hóa Hồi quy tuyến tính trên tập Diabetes')
plt.xlabel('Chỉ số BMI (đã chuẩn hóa)')
plt.ylabel('Tiến triển bệnh (Target)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# %% ===========================================================================
# BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - THỬ NGHIỆM MÔ HÌNH REGRESSION
# ==============================================================================
print("\n" + "=" * 60)
print("BÀI 4: QUY TRÌNH ML HOÀN CHỈNH VỚI CALIFORNIA HOUSING")
print("=" * 60)

# 1. Load dữ liệu
housing = fetch_california_housing(as_frame=True)
X_house = housing.data
y_house = housing.target
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# 2. Preprocess (Scaling): Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()
X_train_h_scaled = scaler.fit_transform(X_train_h)
X_test_h_scaled = scaler.transform(X_test_h)

# 3. Train: Huấn luyện LinearRegression
model_b4 = LinearRegression()
model_b4.fit(X_train_h_scaled, y_train_h)

# 4. Evaluate (MSE): Đánh giá mô hình trên tập Test
y_pred_h = model_b4.predict(X_test_h_scaled)
mse_h = mean_squared_error(y_test_h, y_pred_h)
mae_h = mean_absolute_error(y_test_h, y_pred_h)
r2_h = r2_score(y_test_h, y_pred_h)

print(f"- MSE (Mean Squared Error):  {mse_h:.4f}")
print(f"- MAE (Mean Absolute Error): {mae_h:.4f}")
print(f"- R-squared (R2 Score):       {r2_h:.4f}")

# 5. Visualize một phần kết quả (50 mẫu đầu tiên của tập Test)
sample_n = 50
plt.figure(figsize=(10, 5))
plt.plot(range(sample_n), y_test_h.iloc[:sample_n].values, 'o-', color='black', label='Thực tế (Actual)')
plt.plot(range(sample_n), y_pred_h[:sample_n], 's--', color='red', label='Dự đoán (Predicted)')
plt.title(f'Bài 4: So sánh Giá trị Thực tế vs Dự đoán trên {sample_n} mẫu đầu (California Housing)')
plt.xlabel('Mẫu kiểm tra thứ i')
plt.ylabel('Giá nhà ($100,000)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nĐã hoàn thành các bài tập từ 1 đến 4!")
