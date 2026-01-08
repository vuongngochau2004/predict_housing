# Hướng Dẫn Tiền Xử Lý Dữ Liệu / Data Preprocessing Guide

## 📋 Tổng Quan

Script `data_preprocessing.py` được thiết kế để xử lý dữ liệu bất động sản từ Nhatot.com với các chức năng:

- ✅ **Giữ nguyên dữ liệu gốc** - File gốc không bị thay đổi
- 🧹 Làm sạch dữ liệu (empty rows, duplicates)
- 🔧 Xử lý missing values
- 🏷️ Encoding categorical features
- 📏 Scaling numerical features
- ⚙️ Feature engineering
- 💾 Lưu dữ liệu đã xử lý

## 🚀 Cách Sử Dụng

### 1. Chạy Script Cơ Bản

```bash
python data_preprocessing.py
```

Script sẽ tự động:
- Đọc file `nhatot_crawl4ai.csv`
- Xử lý dữ liệu
- Tạo 2 file mới:
  - `nhatot_crawl4ai_processed.csv` - Dữ liệu đã xử lý
  - `nhatot_crawl4ai_original_backup.csv` - Backup dữ liệu gốc

**File gốc `nhatot_crawl4ai.csv` vẫn được giữ nguyên!**

### 2. Sử Dụng Trong Code Python

```python
from data_preprocessing import HousingDataPreprocessor

# Khởi tạo preprocessor
preprocessor = HousingDataPreprocessor('nhatot_crawl4ai.csv')

# Xử lý dữ liệu
preprocessor.load_data()
preprocessor.clean_empty_rows()
preprocessor.remove_duplicates()
preprocessor.analyze_missing_values()
preprocessor.clean_price_column()
preprocessor.clean_numeric_columns()
preprocessor.handle_missing_values(strategy='auto')
preprocessor.feature_engineering()
preprocessor.encode_categorical_features(method='label')

# Lưu file
output_file = preprocessor.save_processed_data()

# Lấy dữ liệu đã xử lý
df_processed = preprocessor.get_processed_data()
```

### 3. Tùy Chỉnh Xử Lý

#### A. Xử lý Missing Values

```python
# Chiến lược 1: Tự động (auto) - Điền median cho số, mode cho categorical
preprocessor.handle_missing_values(strategy='auto')

# Chiến lược 2: Xóa hàng có missing values
preprocessor.handle_missing_values(strategy='drop')

# Chiến lược 3: Impute - Điền giá trị
preprocessor.handle_missing_values(strategy='impute')
```

#### B. Encoding Categorical Features

```python
# Label Encoding (mặc định)
preprocessor.encode_categorical_features(method='label')

# One-Hot Encoding
preprocessor.encode_categorical_features(method='onehot')
```

#### C. Scaling Features

```python
# Standard Scaling (Z-score normalization)
preprocessor.scale_features(method='standard')

# MinMax Scaling (0-1 normalization)
preprocessor.scale_features(method='minmax')

# Scale chỉ một số cột cụ thể
preprocessor.scale_features(
    method='standard',
    columns=['Diện tích (m2)', 'Chiều ngang (m)', 'Chiều dài (m)']
)
```

## 📊 Các Features Được Tạo

Script tự động tạo các features mới:

1. **Giá/m2** - Giá bán trên mỗi m²
2. **Tổng số phòng** - Tổng phòng ngủ + phòng vệ sinh
3. **Diện tích ước tính** - Chiều ngang × Chiều dài
4. **Kích thước** (category):
   - Rất nhỏ: < 30m²
   - Nhỏ: 30-50m²
   - Trung bình: 50-80m²
   - Lớn: 80-150m²
   - Rất lớn: > 150m²

## 📁 Cấu Trúc Files

```
PredictHousing/
├── nhatot_crawl4ai.csv                    # ✅ File GỐC (không đổi)
├── nhatot_crawl4ai_processed.csv          # 🆕 File đã xử lý
├── nhatot_crawl4ai_original_backup.csv    # 🆕 Backup an toàn
├── data_preprocessing.py                  # Script xử lý
└── README_PREPROCESSING.md                # File này
```

## 🔍 Các Columns Sau Khi Xử Lý

### Columns Gốc (được giữ lại):
- Giá bán, Thành phố, Phường/Xã, Diện tích (m2), etc.

### Columns Mới (được tạo):
- `Giá bán (VND)` - Giá đã chuyển về số
- `Thành phố_encoded` - Encoded city
- `Loại hình_encoded` - Encoded property type
- `Giá/m2` - Price per square meter
- `Tổng số phòng` - Total rooms
- `Kích thước` - Size category
- *(và các columns scaled nếu bạn chọn scaling)*

## 🎯 Chuẩn Bị Cho Machine Learning

```python
# Chia train/test set
X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
    target_col='Giá bán (VND)',
    test_size=0.2,
    random_state=42
)

# Sử dụng cho training
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

## 📈 Phân Tích Dữ Liệu

```python
# Xem thống kê tổng quan
preprocessor.get_summary_statistics()

# Phân tích missing values
preprocessor.analyze_missing_values()

# Lấy dataframe để phân tích
df = preprocessor.get_processed_data()
print(df.info())
print(df.describe())
```

## ⚠️ Lưu Ý Quan Trọng

1. **File gốc luôn được bảo toàn** - Script không ghi đè lên file gốc
2. **Encoding** - Các categorical features được encode thành số
3. **Missing values** - Được xử lý tự động (median/mode)
4. **Price format** - Giá được chuyển từ "1,5 tỷ" → 1500000000 VND

## 🔧 Troubleshooting

### Lỗi: File không tồn tại
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'nhatot_crawl4ai.csv'
```
**Giải pháp**: Đảm bảo file CSV nằm cùng thư mục với script

### Lỗi: Encoding
```bash
UnicodeDecodeError
```
**Giải pháp**: File được lưu với `encoding='utf-8-sig'`

### Muốn không lưu backup
```python
# Tắt lưu backup file gốc
output_file = preprocessor.save_processed_data(save_original=False)
```

## 📞 Hỗ Trợ

Nếu cần thêm features hoặc tùy chỉnh, hãy chỉnh sửa class `HousingDataPreprocessor` trong file `data_preprocessing.py`.

---

**Created**: 2026-01-08  
**Author**: DUT-AI PredictHousing Project
