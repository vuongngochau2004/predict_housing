# 📊 Giải Thích Chi Tiết: HOW & WHY Xử Lý Dữ Liệu

> **Tài liệu này giải thích QUY TRÌNH và LÝ DO đằng sau mỗi bước xử lý dữ liệu**

---

## 📋 Mục Lục

1. [Tổng Quan Pipeline](#1-tổng-quan-pipeline)
2. [Log Transform](#2-log-transform---chi-tiết)
3. [Outlier Detection](#3-outlier-detection---chi-tiết)
4. [Missing Value Imputation](#4-missing-value-imputation---chi-tiết)
5. [Feature Engineering](#5-feature-engineering---chi-tiết)
6. [Encoding](#6-encoding---chi-tiết)
7. [Train/Test Split](#7-traintestsplit---chi-tiết)

---

## 1. Tổng Quan Pipeline

### **Quy trình xử lý (7 bước)**

```
Raw Data (19,733 rows)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: CLEANING                                            │
│ - Remove empty rows         (13,888 rows removed)           │
│ - Parse Vietnamese price    ("3,5 tỷ" → 3,500,000,000)      │
│ - Clean string in numeric   ("nhiều hơn 10" → 10)           │
└─────────────────────────────────────────────────────────────┘
    ↓ (5,845 rows)
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: OUTLIER REMOVAL                                     │
│ - Domain bounds            (3 outliers)                     │
│ - IQR method               (345 outliers)                   │
└─────────────────────────────────────────────────────────────┘
    ↓ (5,497 rows)
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: MISSING VALUE IMPUTATION                            │
│ - Categorical → "Không xác định"                            │
│ - Numeric → Median (global hoặc group)                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: FEATURE ENGINEERING                                 │
│ - Giá_per_m2, Tổng_phòng, Aspect_ratio, Diện_tích_per_phòng │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: ENCODING                                            │
│ - One-Hot (19 columns)                                      │
│ - Target Encoding (2 columns)                               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: LOG TRANSFORM                                       │
│ - Giá bán, Diện tích, Giá_per_m2                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: TRAIN/TEST SPLIT                                    │
│ - 80% Train (4,397)                                         │
│ - 20% Test (1,100)                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
Model-Ready Data (5,497 rows, 34 features)
```

---

## 2. Log Transform - Chi Tiết

### ❓ **WHY: Tại sao phải dùng Log Transform?**

#### **Vấn đề với dữ liệu giá nhà:**

```
Giá nhà thực tế:
├── 500 triệu - 5 tỷ:     ~60% (nhiều)
├── 5 tỷ - 15 tỷ:         ~30% (trung bình)
├── 15 tỷ - 50 tỷ:        ~8%  (ít)
└── 50 tỷ - 200 tỷ:       ~2%  (rất ít - biệt thự)
```

Phân phối này gọi là **RIGHT-SKEWED (lệch phải)** vì:
- **Đuôi dài bên phải**: Vài căn nhà rất đắt kéo dài biểu đồ
- **Mean > Median**: 7.7 tỷ > 5.9 tỷ (mean bị kéo bởi outliers)
- **Skewness = 49.54**: Cực kỳ lệch (bình thường < 1)

#### **Tại sao điều này là vấn đề?**

1. **Linear Regression giả định residuals tuân theo phân phối chuẩn (Normal Distribution)**
   ```
   Nếu Y (giá) không normal → residuals không normal → model sai
   ```

2. **Outliers có ảnh hưởng không cân đối**
   ```
   Nhà 100 tỷ ảnh hưởng model gấp 20 lần nhà 5 tỷ
   → Model cố fit outliers thay vì đa số dữ liệu
   ```

3. **Scale khác nhau quá lớn**
   ```
   500 triệu vs 100 tỷ = chênh lệch 200 lần
   → Model khó học pattern
   ```

---

### 🔧 **HOW: Log Transform hoạt động như thế nào?**

#### **Công thức:**
```python
# Log1p = log(1 + x) để tránh log(0) = undefined
df['Giá bán_log'] = np.log1p(df['Giá bán_numeric'])
```

#### **Ví dụ cụ thể:**

| Giá gốc (VND) | Log(Giá) | Giải thích |
|---------------|----------|------------|
| 500 triệu     | 20.03    | log1p(500,000,000) |
| 1 tỷ          | 20.72    | Tăng gấp đôi → chỉ +0.69 |
| 5 tỷ          | 22.33    | Tăng 10x → chỉ +2.3 |
| 10 tỷ         | 23.03    | Tăng 20x → chỉ +3.0 |
| 50 tỷ         | 24.63    | Tăng 100x → chỉ +4.6 |
| 100 tỷ        | 25.33    | Tăng 200x → chỉ +5.3 |

**Nhận xét:** Sau log, chênh lệch 500 triệu ↔ 100 tỷ chỉ còn ~5 đơn vị thay vì 200x

#### **Kết quả:**
```
TRƯỚC log:
- Skewness = 49.54 (cực kỳ lệch)
- Range: 0 - 1,250 tỷ

SAU log:
- Skewness = -0.44 (gần symmetric!)
- Range: 18.9 - 23.8 (chỉ ~5 đơn vị)
```

---

#### **Visualization so sánh:**

```
TRƯỚC LOG (Original):                 SAU LOG:
                                      
    │                                     │    ╭─╮
    │                                     │   ╭╯ ╰╮
    │                                     │  ╭╯   ╰╮
    │╭╮                                   │ ╭╯     ╰╮
    ╰╯╰──────────────────────►           ╰─╯       ╰─────►
    
    Đuôi dài bên phải                     Symmetric (hình chuông)
    Outliers rõ ràng                      Outliers giảm đáng kể
```

---

### ⚠️ **LƯU Ý QUAN TRỌNG:**

#### **Khi predict, phải convert ngược:**
```python
# Model predict ra log scale
y_pred_log = model.predict(X_test)

# Convert về VND thực
y_pred_vnd = np.expm1(y_pred_log)  # expm1 = exp(x) - 1 (ngược của log1p)

# Ví dụ:
# y_pred_log = 22.33 → y_pred_vnd = 5,000,000,000 (5 tỷ)
```

---

## 3. Outlier Detection - Chi Tiết

### ❓ **WHY: Tại sao phải xử lý Outliers?**

#### **Outliers là gì?**
Các điểm dữ liệu **bất thường** so với phần còn lại:
- Nhà giá 500 tỷ trong khi đa số < 20 tỷ
- Diện tích 5000m² trong khi đa số < 200m²
- Giá/m² = 5 tỷ/m² (rõ ràng sai)

#### **Nguồn gốc outliers:**
1. **Lỗi nhập liệu**: Nhập sai số (thiếu số 0, thừa số 0)
2. **Đơn vị khác**: Nhầm triệu với tỷ
3. **Outliers thật**: Biệt thự, penthouse (real nhưng rare)
4. **Gian lận**: Giá ảo để SEO/marketing

#### **Tại sao phải xử lý?**

1. **Model bị distort:**
   ```
   Mean BỊ KÉO:
   - Không outlier: Mean = 5 tỷ (đúng)
   - Có 1 outlier 500 tỷ: Mean = 10 tỷ (sai!)
   ```

2. **Linear Regression rất sensitive:**
   ```
   RSS = Σ(y - ŷ)²
   
   Outlier 500 tỷ sai 100 tỷ:
   (100 tỷ)² = 10,000 tỷ² → ảnh hưởng cực lớn!
   ```

3. **Overfitting đến outliers:**
   ```
   Model cố học pattern của outliers
   → Bỏ qua pattern của đa số 99% data
   ```

---

### 🔧 **HOW: Xử lý Outliers như thế nào?**

#### **2 phương pháp kết hợp:**

### **Phương pháp 1: Domain Knowledge Bounds (Kiến thức nghiệp vụ)**

```python
OUTLIER_BOUNDS = {
    'Giá bán_numeric': (100_000_000, 500_000_000_000),
    # 100 triệu - 500 tỷ
    # Giải thích:
    # - < 100 triệu: Không thể là nhà (chắc là đất hoặc lỗi)
    # - > 500 tỷ: Quá đắt, có thể ảo hoặc rất rare
    
    'Diện tích (m2)': (5, 10000),
    # 5m² - 10,000m² (1 hectare)
    # Giải thích:
    # - < 5m²: Không thể là nhà ở
    # - > 10,000m²: Đất dự án, không phải nhà
    
    'Giá_per_m2': (1_000_000, 1_000_000_000),
    # 1 triệu/m² - 1 tỷ/m²
    # Giải thích:
    # - < 1 triệu/m²: Quá rẻ, chắc lỗi
    # - > 1 tỷ/m²: Không hợp lý (đắt nhất VN ~300tr/m²)
}

# Code:
for col, (lower, upper) in OUTLIER_BOUNDS.items():
    df = df[(df[col] >= lower) & (df[col] <= upper)]
```

**Kết quả:** Loại 3 outliers rõ ràng sai

---

### **Phương pháp 2: IQR Method (Thống kê)**

#### **IQR là gì?**
```
IQR = InterQuartile Range = Q3 - Q1

Q1 (25th percentile): 25% data nhỏ hơn giá trị này
Q3 (75th percentile): 75% data nhỏ hơn giá trị này
IQR: Khoảng chứa 50% data ở giữa
```

#### **Công thức phát hiện outlier:**
```
Lower bound = Q1 - k × IQR
Upper bound = Q3 + k × IQR

Nếu x < Lower bound hoặc x > Upper bound → Outlier
```

#### **Tại sao chọn k = 3.0 thay vì 1.5?**

| k | Tên gọi | Ưu điểm | Nhược điểm |
|---|---------|---------|------------|
| **1.5** | Standard | Loại nhiều outlier | Mất data thật (villa) |
| **3.0** | Conservative | Giữ data, chỉ loại extreme | Còn 1 số outlier |

**Với bất động sản, k=3.0 là ĐÚNG vì:**
```
Nhà 50 tỷ:
- Có THỂ là biệt thự cao cấp (REAL)
- k=1.5: Bị loại → MẤT DATA TỐT
- k=3.0: Giữ lại → ĐÚN'G

Nhà 500 tỷ:
- Chắc chắn bất thường (lỗi hoặc siêu outlier)
- Cả k=1.5 và k=3.0 đều loại → ĐÚ'NG
```

#### **Code implementation:**
```python
def remove_outliers_iqr(df, column, multiplier=3.0):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
```

**Kết quả:** Loại 345 outliers (6.3% data)

---

#### **Ví dụ thực tế với giá nhà:**

```
Dữ liệu giá (tỷ VND):
[1.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 15.0, 50.0, 200.0]

Q1 = 3.25 tỷ
Q3 = 8.5 tỷ
IQR = 5.25 tỷ

k = 1.5:
- Lower = 3.25 - 1.5×5.25 = -4.6 tỷ (dưới 0, không áp dụng)
- Upper = 8.5 + 1.5×5.25 = 16.4 tỷ
- → Loại: 50 tỷ, 200 tỷ ✓
- → Nhưng cũng loại 15 tỷ (có thể real!)

k = 3.0:
- Lower = 3.25 - 3×5.25 = -12.5 tỷ
- Upper = 8.5 + 3×5.25 = 24.25 tỷ
- → Loại: 50 tỷ, 200 tỷ ✓
- → Giữ: 15 tỷ (có thể real) ✓
```

---

## 4. Missing Value Imputation - Chi Tiết

### ❓ **WHY: Tại sao phải fill missing values?**

1. **Nhiều ML algorithms không chấp nhận NaN:**
   ```python
   sklearn: ValueError: Input contains NaN
   ```

2. **Nếu drop tất cả missing → mất quá nhiều data:**
   ```
   Cột "Hướng" missing 70%
   Drop all → mất 70% dataset!
   ```

3. **Missing có thể chứa thông tin:**
   ```
   "Hướng" = missing có thể nghĩa: "Người bán không quan tâm hướng"
   → Đây là thông tin hữu ích!
   ```

---

### 🔧 **HOW: Xử lý Missing như thế nào?**

#### **3 chiến lược khác nhau cho 3 loại cột:**

### **Chiến lược 1: Fill "Không xác định" cho Categorical**

**Áp dụng cho:** `Hướng`, `Tình trạng nội thất`

```python
df['Hướng'].fillna('Không xác định', inplace=True)
```

**Tại sao không dùng Mode (giá trị phổ biến nhất)?**
```
Mode của Hướng = "Đông Nam"

Nếu fill Mode:
- Tất cả missing → "Đông Nam"
- Model học: "Đông Nam" = phổ biến (SAI!)
- Thực tế: Nhiều nhà không biết hướng

Nếu fill "Không xác định":
- Model học: Có 1 category riêng cho "không biết"
- Nếu "không biết hướng" ảnh hưởng giá → model sẽ học được!
```

---

### **Chiến lược 2: Fill Median theo Group cho Numeric quan trọng**

**Áp dụng cho:** `Chiều ngang`, `Chiều dài`, `Số tầng`

```python
df['Chiều ngang (m)'] = df.groupby('Loại hình')['Chiều ngang (m)'].transform(
    lambda x: x.fillna(x.median())
)
```

**Tại sao Group theo "Loại hình"?**
```
Loại hình = "Biệt thự":
- Chiều ngang trung bình: 10m
- Nên fill missing = 10m

Loại hình = "Nhà ngõ, hẻm":
- Chiều ngang trung bình: 4m
- Nên fill missing = 4m

→ Group giúp fill PHÙ HỢP với từng loại nhà!
```

**Tại sao dùng Median thay vì Mean?**
```
Dữ liệu chiều ngang: [3, 4, 4, 5, 5, 5, 6, 20]

Mean = 6.5m (bị kéo bởi 20m outlier)
Median = 5m (đúng với đa số)

→ Median ROBUST hơn với outliers!
```

---

### **Chiến lược 3: Fill Global Median cho Numeric ít quan trọng**

**Áp dụng cho:** `Số phòng ngủ`, `Số phòng vệ sinh`

```python
df['Số phòng vệ sinh'].fillna(df['Số phòng vệ sinh'].median(), inplace=True)
```

**Tại sao dùng Global thay vì Group?**
```
Số phòng vệ sinh không phụ thuộc nhiều vào Loại hình:
- Biệt thự: 3-5 WC
- Nhà phố: 2-3 WC
- Nhà ngõ: 1-2 WC

Sự khác biệt không quá lớn → Global median đủ tốt
Và đơn giản hơn Group median
```

---

#### **Bảng tổng hợp:**

| Cột | Strategy | Lý do |
|-----|----------|-------|
| **Hướng** | "Không xác định" | Categorical, missing = thông tin |
| **Nội thất** | "Không xác định" | Categorical, missing = thông tin |
| **Chiều ngang** | Group median | Phụ thuộc Loại hình mạnh |
| **Chiều dài** | Group median | Phụ thuộc Loại hình mạnh |
| **Số tầng** | Group median | Biệt thự vs Nhà ngõ khác nhau |
| **Số phòng ngủ** | Global median | Không khác biệt nhiều giữa groups |
| **Số phòng vệ sinh** | Global median | Không khác biệt nhiều giữa groups |

---

## 5. Feature Engineering - Chi Tiết

### ❓ **WHY: Tại sao phải tạo features mới?**

**"Raw features rarely tell the full story"**

#### **Ví dụ cụ thể:**

```
Nhà A: 50m², giá 5 tỷ, ở Quận 1 HCM
Nhà B: 200m², giá 5 tỷ, ở Đồng Nai

Chỉ nhìn "Giá" và "Diện tích" riêng lẻ:
- Cả 2 đều 5 tỷ → giống nhau?
- A: 50m², B: 200m² → B rộng hơn?

Nhìn "Giá/m²":
- A: 100 triệu/m² → RẺ cho Q1 (avg = 200tr/m²)
- B: 25 triệu/m² → ĐẮT cho Đồng Nai (avg = 20tr/m²)

→ Giá/m² = INSIGHT THỰC SỰ!
```

---

### 🔧 **HOW: Tạo Features như thế nào?**

### **Feature 1: Giá_per_m2 (Giá trên mét vuông)**

```python
df['Giá_per_m2'] = df['Giá bán_numeric'] / df['Diện tích (m2)']
```

**Tại sao quan trọng nhất?**
- Correlation với giá: **+0.349** (cao nhất!)
- Chuẩn hóa giá theo kích thước
- Đại diện cho "giá trị vị trí" (HCM > Tây Ninh)

---

### **Feature 2: Tổng_phòng (Tổng số phòng)**

```python
df['Tổng_phòng'] = df['Số phòng ngủ'].fillna(0) + df['Số phòng vệ sinh'].fillna(0)
```

**Tại sao hữu ích?**
- Phản ánh **quy mô tổng thể** của nhà
- Nhiều phòng → nhà to → giá cao hơn
- Đơn giản hóa 2 features thành 1

---

### **Feature 3: Aspect_ratio (Tỷ lệ hình dạng)**

```python
df['Aspect_ratio'] = df['Chiều ngang (m)'] / df['Chiều dài (m)']
```

**Ý nghĩa:**
```
Aspect_ratio ≈ 1.0: Nhà vuông (square)
Aspect_ratio < 0.5: Nhà dài, hẹp (long, narrow)

Ví dụ:
- 5m x 10m → ratio = 0.5 (hơi dài)
- 4m x 20m → ratio = 0.2 (rất dài, giá thấp hơn)
- 10m x 10m → ratio = 1.0 (vuông, giá cao hơn)
```

**Tại sao ảnh hưởng giá?**
- Nhà vuông dễ thiết kế nội thất
- Nhà dài hẹp ánh sáng kém
- Mặt tiền rộng (ratio cao) giá cao hơn

---

### **Feature 4: Diện_tích_per_phòng (Diện tích mỗi phòng)**

```python
df['Diện_tích_per_phòng'] = df['Diện tích (m2)'] / df['Tổng_phòng']
```

**Ý nghĩa:**
- Cao = phòng rộng rãi (spacious)
- Thấp = phòng chật (cramped)

**Ảnh hưởng:**
- 60m² / 3 phòng = 20m²/phòng (rộng rãi)
- 60m² / 6 phòng = 10m²/phòng (chật)
- Cùng diện tích nhưng giá trị khác nhau!

---

## 6. Encoding - Chi Tiết

### ❓ **WHY: Tại sao phải encode?**

**ML models chỉ hiểu số, không hiểu text:**
```python
# Model nhìn thấy:
"Đông Nam", "Tây Bắc", "Bắc"  → ❌ Không hiểu

# Sau encoding:
[1, 0, 0], [0, 1, 0], [0, 0, 1]  → ✅ Hiểu được
```

---

### 🔧 **HOW: Encode như thế nào?**

### **Phương pháp 1: One-Hot Encoding cho Low Cardinality**

**Áp dụng cho:** `Loại hình` (5 values), `Hướng` (9 values), `Nội thất` (5 values)

```python
# Từ:
Loại hình = ["Nhà phố", "Biệt thự", "Nhà ngõ"]

# Thành:
Loại hình_Nhà phố    Loại hình_Biệt thự    Loại hình_Nhà ngõ
      1                    0                     0
      0                    1                     0
      0                    0                     1
```

**Tại sao chỉ dùng cho <10 categories?**
```
10 categories → 10 columns mới (OK)
100 categories → 100 columns (chấp nhận được)
500 categories → 500 columns (CURSE OF DIMENSIONALITY!)
```

---

### **Phương pháp 2: Target Encoding cho High Cardinality**

**Áp dụng cho:** `Phường/Xã` (~500 values), `Thành phố` (~30 values)

```python
# Ý tưởng:
# Mỗi category → trung bình của target (Giá) trong category đó

# Ví dụ:
Phường Bến Nghé (Q1 HCM):
- Các nhà ở đây có giá trung bình: 15 tỷ
- → encoded = 15,000,000,000

Xã Bình Hưng (Bình Chánh):
- Các nhà ở đây có giá trung bình: 3 tỷ
- → encoded = 3,000,000,000
```

**Ưu điểm:**
- 500 categories → chỉ 1 column mới
- Giữ được thông tin về "giá trị location"
- Không explosion số features

**Nhược điểm:**
- Có thể data leakage nếu không cẩn thận

---

## 7. Train/Test Split - Chi Tiết

### ❓ **WHY: Tại sao phải split?**

**"You can't grade your own exam"**

#### **Vấn đề nếu không split:**

```
Scenario: Train và Test trên 100% data

Bước 1: Model "học thuộc" toàn bộ data
        → Train accuracy = 99%

Bước 2: Test trên data đã học
        → Test accuracy = 99% (tất nhiên!)

Bước 3: Deploy lên production
        → Real accuracy = 40% 💀

Tại sao? OVERFITTING!
Model học thuộc noise, không học pattern
```

---

### 🔧 **HOW: Split như thế nào?**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% cho test
    random_state=42,    # Reproducible
    shuffle=True        # Random shuffle
)
```

**Tại sao 80/20?**
- Đủ data để train: 4,397 samples
- Đủ data để test reliable: 1,100 samples
- Standard practice cho dataset >5000

**Tại sao random_state=42?**
- Reproducibility: Chạy lại cho kết quả giống nhau
- Debug dễ hơn
- 42: Số phổ biến (từ "Hitchhiker's Guide to Galaxy")

---

## 📊 Tổng Kết: Workflow Reasoning

| Step | Action | WHY | HOW |
|------|--------|-----|-----|
| **1. Clean** | Remove empty, parse text | Garbage in = garbage out | dropna(), custom parse |
| **2. Outliers** | Remove extreme values | Model sensitive to outliers | Domain + IQR×3 |
| **3. Missing** | Fill NaN | ML cần numeric | Category→text, Numeric→median |
| **4. Features** | Create new columns | Domain knowledge > raw | Division, sum operations |
| **5. Encode** | Convert text→number | ML không hiểu text | One-hot, Target encoding |
| **6. Log** | Transform skewed | Normal assumption | np.log1p() |
| **7. Split** | Separate train/test | Avoid overfitting | train_test_split |

---

## 🎯 Key Decisions Summary

| Decision | WHY | Alternative & Why Not |
|----------|-----|----------------------|
| **Log transform** | Skewness 49→0 | StandardScaler: không fix skew |
| **IQR×3** | Keep real expensive houses | IQR×1.5: lose too much data |
| **Group median** | Respect data patterns | Global median: ignore groups |
| **Target encoding** | 500 categories→1 column | One-hot: 500 columns |
| **80/20 split** | Industry standard | 90/10: test too small |

---

**Mọi quyết định đều có LÝ DO và dựa trên EVIDENCE từ data!**
