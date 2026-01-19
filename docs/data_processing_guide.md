# 📊 Hướng Dẫn Xử Lý Dữ Liệu Bất Động Sản

## 📋 Tổng Quan Pipeline

```
Raw Data (nhatot_crawl4ai.csv - 23,527 rows)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1-2: CLEAN & DEDUPE                                    │
│ → 7,150 rows → 6,396 rows                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: PARSE PRICE                                         │
│ → "3,5 tỷ" → 3.5 (tỷ VNĐ)                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: OUTLIER REMOVAL                                     │
│ → Domain bounds + IQR (k=3.0)                               │
│ → 5,967 rows                                                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: HANDLE MISSING VALUES                               │
│ → Tính toán từ features liên quan                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: FEATURE ENGINEERING                                 │
│ → 4 features mới                                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: ENCODING                                            │
│ → OOF Target Encoding + One-Hot + Ordinal                   │
└─────────────────────────────────────────────────────────────┘
    ↓
Final Data (5,967 rows × 20 features)
```

---

## 📝 Chi Tiết Từng Bước

### Step 1-2: Clean & Dedupe

**WHY?**
- Raw data từ crawl có nhiều rows trống, trùng lặp
- Rows có quá nhiều NaN không mang nhiều thông tin

**HOW?**
```python
# Bỏ rows hoàn toàn trống
df_cleaned = df.dropna(how='all')

# Bỏ rows có > 6 giá trị NaN
missing_count = df_cleaned.isnull().sum(axis=1)
df_cleaned = df_cleaned[missing_count <= 6]

# Xoá duplicates
df = df.drop_duplicates(keep="first")
```

**KẾT QUẢ:** 23,527 → 7,150 → 6,396 rows

---

### Step 3: Parse Price

**WHY?**
- Giá dạng text ("3,5 tỷ", "750 triệu") không thể tính toán
- Cần chuyển về đơn vị thống nhất (tỷ VNĐ)

**HOW?**
```python
def convert_price_to_billion(price_str):
    if 'tỷ' in price_str:
        return float(value)       # "3,5 tỷ" → 3.5
    elif 'triệu' in price_str:
        return float(value) / 1000  # "750 triệu" → 0.75
```

**KẾT QUẢ:** 
- Min: 0.00 tỷ
- Max: 1,250 tỷ (outlier!)
- Median: 5.90 tỷ

---

### Step 4: Outlier Removal

**WHY?**
- Outliers cực đoan ảnh hưởng training
- Cần giữ lại outliers "thật" (biệt thự, nhà mặt phố đắc địa)

**HOW?**

**1. Domain Knowledge Bounds:**
```python
PRICE_MIN = 0.2      # tỷ - dưới 200 triệu không hợp lý
PRICE_MAX = 200      # tỷ - trên 200 tỷ rất hiếm

AREA_MIN = 10        # m2
AREA_MAX = 1500      # m2

WIDTH_MIN = 2, WIDTH_MAX = 50   # m
LENGTH_MIN = 3, LENGTH_MAX = 100  # m
ROOM_MAX = 20
FLOOR_MAX = 15
```

**2. IQR Filter (k=3.0 - nhẹ tay):**
```python
def iqr_filter(data, col, k=3.0):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    return data[data[col].between(q1 - k*iqr, q3 + k*iqr)]
```

> **Tại sao k=3.0?** 
> - k=1.5 quá chặt, loại nhiều outliers thật (nhà cao cấp)
> - k=3.0 giữ lại 99.7% data trong phân phối chuẩn

**KẾT QUẢ:** 6,396 → 5,967 rows

---

### Step 5: Handle Missing Values

**WHY?**
- Models không xử lý NaN trực tiếp
- Cần cách điền hợp lý dựa trên domain knowledge

**HOW?**

| Cột | % NaN | Phương pháp | Lý do |
|-----|-------|-------------|-------|
| Chiều ngang/dài | ~20% | Tính từ Diện tích | DT = Ngang × Dài |
| Số phòng ngủ | 1.2% | Tính từ Diện tích | Median m²/phòng |
| Số phòng vệ sinh | 26% | Tính từ Số phòng ngủ | Group median |
| Số tầng | 34% | Median | Global median |
| Hướng | 75% | "Không xác định" | Category mới |
| Nội thất | 48% | "Không xác định" | Category mới |

```python
# Chiều ngang từ DT và Chiều dài
df.loc[m, 'Chiều ngang (m)'] = df['Diện tích (m2)'] / df['Chiều dài (m)']

# WC từ Phòng ngủ (group median)
wc_med = df.groupby('Số phòng ngủ')['Số phòng vệ sinh'].median()
df['Số phòng vệ sinh'] = df['Số phòng ngủ'].map(wc_med)
```

> **Tại sao Hướng/Nội thất → "Không xác định"?**
> - NaN rất cao (~50-75%)
> - Đây là thông tin seller KHÔNG CUNG CẤP, không phải "không biết"
> - Model có thể học pattern "không cung cấp" riêng

---

### Step 6: Feature Engineering

**WHY?**
- Features thô chỉ capture 1 khía cạnh
- Features mới capture relationships giữa các features

**HOW?**

| Feature Mới | Công Thức | Insight |
|-------------|-----------|---------|
| **Giá_per_m2** | Giá / Diện tích | Indicator của khu vực đắt/rẻ |
| **Tổng_phòng** | Số phòng ngủ + WC | Tổng tiện nghi |
| **Aspect_ratio** | Ngang / Dài | Hình dạng lô (vuông vs dài) |
| **Diện_tích_per_phòng** | DT / Tổng phòng | Rộng rãi hay chật chội |

> **Tại sao Giá_per_m2 quan trọng?**
> - Là proxy cho "giá trị vị trí"
> - Correlation cao với target trước khi encode location

---

### Step 7: Encoding

**WHY?**
- Models cần input số
- High-cardinality features (Phường/Xã) gây curse of dimensionality nếu One-Hot

**HOW?**

#### 1. OOF Target Encoding: Thành phố

```python
# 5-Fold: encode mỗi row bằng data từ 4 folds còn lại
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(df):
    fold_train = df.iloc[train_idx]
    fold_val = df.iloc[val_idx]
    
    # Mean từ TRAIN fold
    means = fold_train.groupby('Thành phố')[target].mean()
    counts = fold_train.groupby('Thành phố')[target].count()
    
    # Smoothing: tránh overfitting category ít samples
    smooth = (means*counts + global_mean*10) / (counts + 10)
    
    # Apply cho VAL fold
    df.loc[val_idx, 'Thành phố_encoded'] = fold_val['Thành phố'].map(smooth)
```

> **Tại sao dùng K-Fold?**
> - Tránh data leakage
> - Mỗi row được encode bằng data KHÔNG CHỨA giá trị của chính nó

> **Tại sao Smoothing?**
> - Phường chỉ có 3 nhà → mean không đáng tin
> - Smoothing kéo về global mean để tránh overfitting

#### 2. Target Encoding: Phường/Xã

```python
# Simple smoothed (không K-Fold vì đã có K-Fold cho Thành phố)
smooth = (counts*means + global_mean*10) / (counts + 10)
df['Phường/Xã_encoded'] = df['Phường/Xã'].map(smooth)
```

#### 3. One-Hot Encoding: Loại hình

```python
# Low cardinality (4 categories) → One-Hot OK
df = pd.concat([df, pd.get_dummies(df['Loại hình'], prefix='Loại hình')], axis=1)
```

#### 4. Ordinal Encoding: Giấy tờ pháp lý

```python
# Có thứ tự tự nhiên
phap_ly_order = {
    'Đã có sổ': 4,          # Tốt nhất
    'Sổ chung': 3,
    'Đang chờ sổ': 2,
    'Giấy tờ viết tay': 1,
    'Không có sổ': 0         # Rủi ro nhất
}
```

---

## 📊 Output Files

| File | Mô tả |
|------|-------|
| `cleaned_nhatot_data.csv` | Sau bước 1-4 |
| `data_nan_handled_final.csv` | Sau bước 5 |
| `data_with_new_features.csv` | Sau bước 6 |
| `data_encoded.csv` | Sau bước 7 (final) |

---

## 🎓 Key Takeaways

1. **Domain Knowledge quan trọng hơn kỹ thuật**
   - Bounds từ thực tế BĐS hiệu quả hơn IQR blind

2. **K-Fold Target Encoding**
   - Luôn dùng khi encode based on target
   - Smoothing giúp tránh overfitting

3. **Missing Values khác nhau có strategy khác nhau**
   - Numeric → tính từ features liên quan
   - Categorical với NaN cao → category riêng

4. **Feature Engineering từ domain**
   - Giá/m² là indicator quan trọng nhất cho location value
