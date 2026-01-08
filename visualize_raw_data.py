import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu
df = pd.read_csv('cleaned_nhatot_data.csv', encoding='utf-8')

# Hiển thị thông tin cơ bản
print("Kích thước dữ liệu:", df.shape)
print("\nCác cột:", df.columns.tolist())

# Tiền xử lý giá
def clean_price(value):
    if isinstance(value, str):
        value = value.replace('"', '').replace(',', '.')
        if 'tỷ' in value.lower():
            return float(value.lower().replace(' tỷ', '').replace('tỷ', '').strip()) * 1e9
        elif 'triệu' in value.lower():
            return float(value.lower().replace(' triệu', '').replace('triệu', '').strip()) * 1e6
        else:
            try:
                return float(value.strip())
            except:
                return np.nan
    else:
        try:
            return float(value)
        except:
            return np.nan

df['Giá bán'] = df['Giá bán'].apply(clean_price)
df = df.dropna(subset=['Giá bán'])
df = df[df['Giá bán'] > 0]  # Loại bỏ giá âm hoặc bằng 0

print(f"\nSố lượng bản ghi sau khi làm sạch: {len(df)}")
print(f"Giá trung bình: {df['Giá bán'].mean():,.0f} VND")
print(f"Giá cao nhất: {df['Giá bán'].max():,.0f} VND")
print(f"Giá thấp nhất: {df['Giá bán'].min():,.0f} VND")

# 1. PHÂN TÍCH THÀNH PHỐ
print("\n" + "="*50)
print("1. PHÂN TÍCH THEO THÀNH PHỐ")
city_price = df.groupby('Thành phố')['Giá bán'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
print(city_price)

plt.figure(figsize=(12, 6))
city_price_top = city_price.head(15)  # Top 15 thành phố
plt.bar(city_price_top.index, city_price_top['mean']/1e9, color='skyblue')
plt.title('Giá Trung Bình Theo Thành Phố (Top 15)', fontsize=14)
plt.xlabel('Thành phố')
plt.ylabel('Giá (tỷ VND)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. PHÂN TÍCH THEO LOẠI HÌNH NHÀ
print("\n" + "="*50)
print("2. PHÂN TÍCH THEO LOẠI HÌNH NHÀ")
type_price = df.groupby('Loại hình')['Giá bán'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
print(type_price)

plt.figure(figsize=(10, 6))
plt.bar(type_price.index, type_price['mean']/1e9, color='lightcoral')
plt.title('Giá Trung Bình Theo Loại Hình Nhà', fontsize=14)
plt.xlabel('Loại hình nhà')
plt.ylabel('Giá (tỷ VND)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. PHÂN TÍCH THEO HƯỚNG NHÀ
print("\n" + "="*50)
print("3. PHÂN TÍCH THEO HƯỚNG NHÀ")
# Làm sạch dữ liệu hướng
df['Hướng'] = df['Hướng'].fillna('Không xác định')
direction_price = df.groupby('Hướng')['Giá bán'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
print(direction_price)

plt.figure(figsize=(10, 6))
direction_top = direction_price.head(10)  # Top 10 hướng
plt.bar(direction_top.index, direction_top['mean']/1e9, color='lightgreen')
plt.title('Giá Trung Bình Theo Hướng Nhà (Top 10)', fontsize=14)
plt.xlabel('Hướng')
plt.ylabel('Giá (tỷ VND)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. PHÂN TÍCH THEO GIẤY TỜ PHÁP LÝ
print("\n" + "="*50)
print("4. PHÂN TÍCH THEO GIẤY TỜ PHÁP LÝ")
df['Giấy tờ pháp lý'] = df['Giấy tờ pháp lý'].fillna('Không xác định')
legal_price = df.groupby('Giấy tờ pháp lý')['Giá bán'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
print(legal_price)

plt.figure(figsize=(8, 6))
plt.bar(legal_price.index, legal_price['mean']/1e9, color='gold')
plt.title('Giá Trung Bình Theo Giấy Tờ Pháp Lý', fontsize=14)
plt.xlabel('Loại giấy tờ')
plt.ylabel('Giá (tỷ VND)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. PHÂN TÍCH THEO TÌNH TRẠNG NỘI THẤT
print("\n" + "="*50)
print("5. PHÂN TÍCH THEO TÌNH TRẠNG NỘI THẤT")
df['Tình trạng nội thất'] = df['Tình trạng nội thất'].fillna('Không xác định')
furniture_price = df.groupby('Tình trạng nội thất')['Giá bán'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
print(furniture_price)

plt.figure(figsize=(8, 6))
plt.bar(furniture_price.index, furniture_price['mean']/1e9, color='violet')
plt.title('Giá Trung Bình Theo Tình Trạng Nội Thất', fontsize=14)
plt.xlabel('Tình trạng nội thất')
plt.ylabel('Giá (tỷ VND)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. PHÂN TÍCH THEO SỐ PHÒNG NGỦ
print("\n" + "="*50)
print("7. PHÂN TÍCH THEO SỐ PHÒNG NGỦ")
df['Số phòng ngủ'] = df['Số phòng ngủ'].apply(lambda x: str(x).replace('nhiều hơn 10', '11')).replace('', np.nan)
df['Số phòng ngủ'] = pd.to_numeric(df['Số phòng ngủ'], errors='coerce')
bedroom_price = df.groupby('Số phòng ngủ')['Giá bán'].agg(['count', 'mean', 'std']).sort_values('Số phòng ngủ')
print(bedroom_price)

plt.figure(figsize=(10, 6))
valid_bedroom = bedroom_price[bedroom_price['count'] >= 5]  # Chỉ lấy giá trị có đủ dữ liệu
plt.bar(valid_bedroom.index.astype(str), valid_bedroom['mean']/1e9, color='orange')
plt.title('Giá Trung Bình Theo Số Phòng Ngủ', fontsize=14)
plt.xlabel('Số phòng ngủ')
plt.ylabel('Giá (tỷ VND)')
plt.tight_layout()
plt.show()

# 8. PHÂN TÍCH THEO SỐ TẦNG
print("\n" + "="*50)
print("8. PHÂN TÍCH THEO SỐ TẦNG")
df['Số tầng'] = pd.to_numeric(df['Số tầng'], errors='coerce')
floor_price = df.groupby('Số tầng')['Giá bán'].agg(['count', 'mean', 'std']).sort_values('Số tầng')
print(floor_price)

plt.figure(figsize=(10, 6))
valid_floor = floor_price[floor_price['count'] >= 5]  # Chỉ lấy giá trị có đủ dữ liệu
plt.bar(valid_floor.index.astype(str), valid_floor['mean']/1e9, color='brown')
plt.title('Giá Trung Bình Theo Số Tầng', fontsize=14)
plt.xlabel('Số tầng')
plt.ylabel('Giá (tỷ VND)')
plt.tight_layout()
plt.show()

# 9. ĐÁNH GIÁ MỨC ĐỘ ẢNH HƯỞNG CỦA TẤT CẢ FEATURE (CV CHUẨN)
print("\n" + "="*50)
print("9. ĐÁNH GIÁ MỨC ĐỘ ẢNH HƯỞNG CỦA TẤT CẢ FEATURE")
print("CV = std(mean_price_per_group) / mean(mean_price_per_group)")

def calculate_feature_cv(group_col, min_count=5):
    """
    CV dựa trên GIÁ TRUNG BÌNH giữa các nhóm của feature
    """
    stats = (
        df.groupby(group_col)['Giá bán']
        .agg(['count', 'mean'])
    )

    stats = stats[stats['count'] >= min_count]

    if len(stats) < 2:
        return np.nan

    mean_prices = stats['mean']
    return mean_prices.std() / mean_prices.mean()


# ====== XỬ LÝ FEATURE SỐ LIÊN TỤC → BIN ======
df['Nhóm diện tích'] = pd.cut(
    df['Diện tích (m2)'],
    bins=[0, 30, 50, 80, 120, 200, 500, 1000],
    right=False
)

df['Nhóm chiều ngang'] = pd.cut(
    df['Chiều ngang (m)'],
    bins=[0, 3, 4, 5, 6, 8, 12, 20],
    right=False
)

df['Nhóm chiều dài'] = pd.cut(
    df['Chiều dài (m)'],
    bins=[0, 8, 12, 16, 20, 30, 50],
    right=False
)


# ====== TÍNH CV CHO TẤT CẢ FEATURE ======
cv_scores = {
    # Vị trí
    'Thành phố': calculate_feature_cv('Thành phố'),
    'Phường/Xã': calculate_feature_cv('Phường/Xã'),

    # Hình thức
    'Loại hình': calculate_feature_cv('Loại hình'),
    'Giấy tờ pháp lý': calculate_feature_cv('Giấy tờ pháp lý'),
    'Hướng': calculate_feature_cv('Hướng'),
    'Tình trạng nội thất': calculate_feature_cv('Tình trạng nội thất'),

    # Kích thước
    'Diện tích': calculate_feature_cv('Nhóm diện tích'),
    'Chiều ngang': calculate_feature_cv('Nhóm chiều ngang'),
    'Chiều dài': calculate_feature_cv('Nhóm chiều dài'),

    # Công năng
    'Số phòng ngủ': calculate_feature_cv('Số phòng ngủ'),
    'Số phòng vệ sinh': calculate_feature_cv('Số phòng vệ sinh'),
    'Số tầng': calculate_feature_cv('Số tầng'),
}

cv_df = (
    pd.DataFrame.from_dict(cv_scores, orient='index', columns=['CV'])
    .dropna()
    .sort_values('CV', ascending=False)
)

print("\nHệ số biến thiên (CV) của toàn bộ feature:")
print(cv_df)


# ====== VISUALIZE CV ======
plt.figure(figsize=(12, 6))
plt.bar(cv_df.index, cv_df['CV'])
plt.title('Độ Biến Động Giá Giữa Các Feature (CV)', fontsize=14)
plt.xlabel('Feature')
plt.ylabel('Hệ số biến thiên (CV)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


