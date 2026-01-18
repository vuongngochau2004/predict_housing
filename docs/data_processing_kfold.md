# 📊 Xử Lý Dữ Liệu & K-Fold Target Encoding

## 📋 Mục Lục

1. [Quy Trình Xử Lý Dữ Liệu](#1-quy-trình-xử-lý-dữ-liệu)
2. [Smoothed K-Fold Target Encoding](#2-smoothed-k-fold-target-encoding)
3. [Tại Sao Phải Dùng K-Fold?](#3-tại-sao-phải-dùng-k-fold)
4. [Code Implementation](#4-code-implementation)

---

## 1. Quy Trình Xử Lý Dữ Liệu

### Tại Sao Không Tách Train/Val/Test Trước?

Có 2 cách tiếp cận:

| Cách | Mô tả | Ưu/Nhược |
|------|-------|----------|
| **Cách A** | Tách Train/Test → Preprocessing trên mỗi set | ❌ Phức tạp, dễ sai |
| **Cách B** | Preprocessing với K-Fold → Tách Train/Test | ✅ **Chúng ta dùng cách này** |

**Lý do dùng Cách B:**
- K-Fold Target Encoding ĐÃ đảm bảo không data leakage
- Mỗi row được encode bằng data từ 4/5 còn lại
- Không cần tách trước vì K-Fold đã xử lý vấn đề này

### Pipeline Overview

```
Raw Data (19,733 rows)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1-4: CLEANING, OUTLIERS, MISSING, FEATURES             │
│ → Các bước này KHÔNG dùng target → OK làm trên toàn bộ data │
└─────────────────────────────────────────────────────────────┘
    │ (5,497 rows)
    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: K-FOLD TARGET ENCODING                              │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ For each row i:                                     │   │
│   │   1. Chia data thành 5 folds                        │   │
│   │   2. Row i thuộc fold k                             │   │
│   │   3. Tính mean từ 4 folds CÒN LẠI (không có fold k) │   │
│   │   4. Encode row i bằng mean đó                      │   │
│   │                                                     │   │
│   │ → Mỗi row KHÔNG thấy giá trị của chính nó!          │   │
│   │ → KHÔNG DATA LEAKAGE!                               │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: LOG TRANSFORM                                       │
│ → Giá, Diện tích, Giá/m²                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: TRAIN/TEST SPLIT (80/20)                            │
│                                                             │
│   Data đã preprocessed                                      │
│       │                                                     │
│       ├── Train (80%) → Dùng để train models                │
│       │                 → K-Fold CV ở đây để tune params    │
│       │                                                     │
│       └── Test (20%)  → Holdout set                         │
│                       → CHỈ DÙNG để đánh giá cuối cùng      │
│                       → KHÔNG được chạm trong quá trình tune│
└─────────────────────────────────────────────────────────────┘
```

### ⚠️ Lưu Ý Quan Trọng

**Q: Tại sao K-Fold Target Encoding làm TRƯỚC split?**

**A:** Vì K-Fold Target Encoding đã tự tách data bên trong!

```
K-Fold Target Encoding với 5 folds:

Fold 1: [████░░░░░░░░░░░░░░░░] ← Encode bởi 2,3,4,5
Fold 2: [░░░░████░░░░░░░░░░░░] ← Encode bởi 1,3,4,5
Fold 3: [░░░░░░░░████░░░░░░░░] ← Encode bởi 1,2,4,5
Fold 4: [░░░░░░░░░░░░████░░░░] ← Encode bởi 1,2,3,5
Fold 5: [░░░░░░░░░░░░░░░░████] ← Encode bởi 1,2,3,4

→ Mỗi row được encode bởi 80% data còn lại
→ Giống như đã "hold out" 20% cho mỗi row!
```

---

## 2. Smoothed K-Fold Target Encoding

### Công Thức

```
Smoothed Mean = (count × category_mean + α × global_mean) / (count + α)

Với:
- count: Số samples trong category
- category_mean: Mean của category (từ train folds)
- global_mean: Mean của toàn bộ data (từ train folds)
- α: Smoothing factor (default = 10)
```

### Ví Dụ

```
Phường A: 500 nhà, mean = 5 tỷ
Global mean = 6 tỷ
α = 10

Smoothed = (500 × 5 + 10 × 6) / (500 + 10)
         = (2500 + 60) / 510
         = 5.02 tỷ  ← Gần như unchanged

Phường B: 2 nhà, mean = 50 tỷ (outlier!)
Smoothed = (2 × 50 + 10 × 6) / (2 + 10)
         = (100 + 60) / 12
         = 13.33 tỷ  ← Bị kéo về global mean
```

---

## 3. Tại Sao Phải Dùng K-Fold?

### So Sánh Naive vs K-Fold

| Aspect | Naive Target Encoding | K-Fold Target Encoding |
|--------|----------------------|------------------------|
| Data Leakage | ❌ CÓ | ✅ KHÔNG |
| Row i được encode bằng | Mean CHỨA row i | Mean KHÔNG CHỨA row i |
| Train/Val split cần trước? | ✅ Bắt buộc | ❌ Không cần |
| Performance thực tế | Ảo cao | Đúng |

---

## 4. Code Implementation

```python
from sklearn.model_selection import KFold
import numpy as np

def smoothed_kfold_target_encoding(df, cat_col, target_col, n_folds=5, smoothing=10):
    """
    K-Fold Target Encoding không data leakage
    """
    encoded = np.zeros(len(df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        
        # Tính từ TRAIN folds only
        global_mean = train[target_col].mean()
        agg = train.groupby(cat_col)[target_col].agg(['mean', 'count'])
        
        # Smoothed
        smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        
        # Apply cho VAL fold
        encoded[val_idx] = df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean).values
    
    return encoded
```

---

## 5. Workflow Khi Training

```
Sau preprocessing, data đã chia:
├── Train (4,397 rows)
└── Test (1,100 rows)

Khi training:

1. OPTIONAL: K-Fold CV trên Train set để tune hyperparameters
   for fold in 5-fold:
       train_fold, val_fold = split(Train)
       model.fit(train_fold)
       score = model.evaluate(val_fold)
   best_params = average(scores)

2. Train final model trên TOÀN BỘ Train set
   model.fit(Train, best_params)

3. Evaluate trên Test set (CHƯA TỪNG THẤY)
   final_score = model.evaluate(Test)
```

---

## 🎓 Tóm Tắt

| Bước | Mục đích | Data Leakage? |
|------|----------|---------------|
| Step 1-4 | Clean, outliers, missing | ✅ Không (không dùng target) |
| Step 5 | K-Fold Target Encoding | ✅ Không (K-Fold xử lý) |
| Step 6 | Log transform | ✅ Không |
| Step 7 | Train/Test split | ✅ Không |
| Training | K-Fold CV để tune | ✅ Không |
| Final | Evaluate trên Test | ✅ Không |
