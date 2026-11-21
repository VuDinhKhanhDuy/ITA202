"""
=============================================================================
ASSIGNMENT GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU VÀ XÂY DỰNG MÔ HÌNH CƠ SỞ
Môn học: Học máy cơ bản (ITA202) - FPT Polytechnic
Đề tài: Dự đoán giá nhà California (California Housing)
=============================================================================
Nhiệm vụ Giai đoạn 1:
1. Chọn và Khám phá dữ liệu: Kiểm tra Shape, Info, Describe, Missing values.
2. Làm sạch dữ liệu: Xử lý giá trị khuyết thiếu (NaN) và ngoại lệ (Outliers).
3. Tiền xử lý: Chuẩn hóa dữ liệu (Scaling) và chia tập Train (80%) - Test (20%).
4. Huấn luyện mô hình cơ sở: Áp dụng Hồi quy tuyến tính (Linear Regression).
5. Đánh giá ban đầu: Tính toán MAE, MSE, RMSE, R² và trực quan hóa kết quả.
=============================================================================
"""

import os
import matplotlib
matplotlib.use('Agg') # Sử dụng backend Agg tránh lỗi GUI trên Windows
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Thiết lập giao diện đồ thị
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def run_giai_doan_1():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'housing.csv')
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    print("=" * 75)
    print("ASSIGNMENT GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH CƠ SỞ")
    print("MÔN: HỌC MÁY CƠ BẢN (ITA202) - FPT POLYTECHNIC")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # BƯỚC 1: CHỌN VÀ KHÁM PHÁ DỮ LIỆU (EDA)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 1] Khám phá dữ liệu (EDA):")
    df = pd.read_csv(data_path)
    print(f"- Kích thước tập dữ liệu (Shape): {df.shape[0]} dòng, {df.shape[1]} cột")
    
    print("\n- Cấu trúc các cột và kiểu dữ liệu (Info):")
    df.info()
    
    print("\n- Thống kê mô tả các đặc trưng số (Describe):")
    print(df.describe().round(2))
    
    print("\n- Kiểm tra số lượng giá trị khuyết thiếu (Missing values):")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0])

    # Biểu đồ 1: Phân phối giá nhà (Target distribution)
    plt.figure(figsize=(9, 5))
    sns.histplot(df['median_house_value'], bins=50, kde=True, color='royalblue')
    plt.title("Phân phối giá nhà trung vị (Median House Value)", fontsize=13, fontweight='bold')
    plt.xlabel("Giá trị nhà ($ USD)", fontsize=11)
    plt.ylabel("Số lượng mẫu (Count)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'eda_target_distribution.png'), dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ: eda_target_distribution.png")

    # Biểu đồ 2: Bản đồ địa lý California
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df['longitude'], df['latitude'],
        c=df['median_house_value'], cmap='jet', alpha=0.4,
        s=df['population'] / 100, label='Dân số (/100)'
    )
    plt.colorbar(scatter, label='Giá nhà ($ USD)')
    plt.title("Phân bố giá nhà California theo Kinh độ - Vĩ độ", fontsize=13, fontweight='bold')
    plt.xlabel("Kinh độ (Longitude)", fontsize=11)
    plt.ylabel("Vĩ độ (Latitude)", fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'eda_geographical_scatter.png'), dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ: eda_geographical_scatter.png")

    # Biểu đồ 3: Ma trận tương quan
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Ma trận tương quan Pearson", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'eda_correlation_heatmap.png'), dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ: eda_correlation_heatmap.png")

    # -------------------------------------------------------------------------
    # BƯỚC 2: LÀM SẠCH DỮ LIỆU & KỸ NGHỆ ĐẶC TRƯNG
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 2] Làm sạch dữ liệu & Kỹ nghệ đặc trưng:")
    df_clean = df.copy()
    
    # Kỹ nghệ đặc trưng
    df_clean['rooms_per_household'] = df_clean['total_rooms'] / df_clean['households']
    df_clean['bedrooms_per_room'] = df_clean['total_bedrooms'] / df_clean['total_rooms']
    df_clean['population_per_household'] = df_clean['population'] / df_clean['households']
    print("- Đã tạo 3 đặc trưng tỷ lệ: rooms_per_household, bedrooms_per_room, population_per_household.")

    # -------------------------------------------------------------------------
    # BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU & CHIA TẬP TRAIN / TEST
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 3] Tiền xử lý dữ liệu:")
    X = df_clean.drop(columns=['median_house_value'])
    y = df_clean['median_house_value']

    # Chia tập dữ liệu 80% Train - 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"- Kích thước tập Train (80%): {X_train.shape[0]} mẫu")
    print(f"- Kích thước tập Test  (20%): {X_test.shape[0]} mẫu")

    num_cols = [c for c in X.columns if c != 'ocean_proximity']
    cat_cols = ['ocean_proximity']

    # Xử lý khuyết thiếu (Median) và Chuẩn hóa (StandardScaler) cho biến số
    # Mã hóa One-Hot cho biến phân loại
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    print(f"- Đã tiền xử lý thành công. Số chiều đặc trưng sau mã hóa: {X_train_prepared.shape[1]}")

    # -------------------------------------------------------------------------
    # BƯỚC 4: HUẤN LUYỆN MÔ HÌNH CƠ SỞ (LINEAR REGRESSION)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 4] Huấn luyện mô hình cơ sở (Linear Regression):")
    lr = LinearRegression()
    lr.fit(X_train_prepared, y_train)
    print("-> Đã huấn luyện xong mô hình Linear Regression!")

    # -------------------------------------------------------------------------
    # BƯỚC 5: ĐÁNH GIÁ BAN ĐẦU & TRỰC QUAN HÓA
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 5] Đánh giá ban đầu mô hình cơ sở:")
    y_train_pred = lr.predict(X_train_prepared)
    y_test_pred = lr.predict(X_test_prepared)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print("-" * 50)
    print(f"Chỉ số Train: RMSE = ${train_rmse:,.2f} | R² = {train_r2:.4f}")
    print(f"Chỉ số Test : RMSE = ${test_rmse:,.2f} | R² = {test_r2:.4f}")
    print(f"Chỉ số Test : MAE  = ${test_mae:,.2f}")
    print(f"Chỉ số Test : MSE  = {test_mse:,.2f}")
    print("-" * 50)

    # Đồ thị Giá thực tế vs Giá dự đoán
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.3, color='steelblue', s=20)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Đường lý tưởng (y = x)')
    plt.title("Giá thực tế vs Giá dự đoán (Linear Regression)", fontsize=13, fontweight='bold')
    plt.xlabel("Giá trị thực tế ($ USD)", fontsize=11)
    plt.ylabel("Giá trị dự đoán ($ USD)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'linear_regression_actual_vs_predicted.png'), dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ: linear_regression_actual_vs_predicted.png")

    # Đồ thị Sai số phần dư (Residuals)
    residuals = y_test - y_test_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=50, kde=True, color='crimson')
    plt.axvline(0, color='black', linestyle='--', lw=1.5)
    plt.title("Phân phối sai số phần dư (Residuals)", fontsize=13, fontweight='bold')
    plt.xlabel("Phần dư = Thực tế - Dự đoán ($ USD)", fontsize=11)
    plt.ylabel("Tần suất", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'linear_regression_residuals.png'), dpi=300)
    plt.close()
    print("-> Đã lưu biểu đồ: linear_regression_residuals.png")

    print("\n" + "=" * 75)
    print("KẾT THÚC GIAI ĐOẠN 1: Dữ liệu và mô hình cơ sở đã sẵn sàng cho Giai đoạn 2!")
    print("=" * 75)

if __name__ == '__main__':
    run_giai_doan_1()
