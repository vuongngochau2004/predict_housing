# 📘 Tài liệu Kỹ thuật: Training Model & Data Crawling

Tài liệu này giải thích chi tiết về chiến lược huấn luyện mô hình dự đoán giá nhà và quy trình thu thập dữ liệu, bao gồm lý do (Why) và cách thức thực hiện (How) cho từng kỹ thuật.

---

## 📑 Mục lục

1. [Thu thập Dữ liệu (Data Crawling)](#1-thu-thập-dữ-liệu-data-crawling)
2. [Chiến lược Huấn luyện (Training Strategy)](#2-chiến-lược-huấn-luyện-training-strategy)
3. [Optuna - Tối ưu Hyperparameters](#3-optuna---tối-ưu-hyperparameters)
4. [K-Fold Cross Validation](#4-k-fold-cross-validation)

---

## 1. Thu thập Dữ liệu (Data Crawling)

### 📍 WHY - Tại sao cần crawl dữ liệu từ Nhatot.com?

1. **Nguồn dữ liệu thực tế**: Nhatot.com là một trong những trang web mua bán bất động sản lớn nhất Việt Nam, cung cấp dữ liệu thực về giá nhà đất.

2. **Dữ liệu phong phú**: Mỗi tin đăng chứa nhiều đặc trưng quan trọng:
   - Giá bán
   - Diện tích, chiều ngang, chiều dài
   - Vị trí (Thành phố, Phường/Xã)
   - Loại hình bất động sản
   - Số phòng ngủ, số phòng vệ sinh, số tầng
   - Giấy tờ pháp lý, hướng, tình trạng nội thất

3. **Không có API công khai**: Nhatot.com không cung cấp API để lấy dữ liệu, do đó cần phải crawl trực tiếp từ trang web.

### 🔧 HOW - Cách thức crawl dữ liệu

#### Công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **Crawl4AI** | Framework async crawler hiện đại với browser automation |
| **BeautifulSoup** | Parse HTML để trích xuất thông tin |
| **asyncio** | Xử lý bất đồng bộ, tăng tốc độ crawl |

#### Các kỹ thuật chính

**1. Async Crawling với Browser Pooling**

```python
async with AsyncWebCrawler(config=browser_config) as crawler:
    # Crawl nhiều trang đồng thời
    tasks = [self.parse_detail_page(crawler, url) for url in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

> **Why?** Sử dụng async giúp crawl đồng thời nhiều trang, giảm thời gian chờ từ hàng giờ xuống còn vài phút.

**2. Stealth Mode - Chế độ ẩn danh**

```python
browser_config = BrowserConfig(
    headless=True,
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
    extra_args=[
        "--disable-blink-features=AutomationControlled",
        "--disable-dev-shm-usage",
    ],
    use_managed_browser=True,
)
```

> **Why?** Tránh bị phát hiện là bot bởi anti-bot systems, giả lập hành vi trình duyệt thật.

**3. JSON-LD Extraction**

```python
json_scripts = soup.find_all('script', type='application/ld+json')
for script in json_scripts:
    data = json.loads(script.string)
    if data.get('@type') == 'ItemList':
        # Extract listing URLs
```

> **Why?** JSON-LD là structured data được nhúng sẵn trong HTML, dễ parse và ổn định hơn so với selector CSS.

**4. Concurrency Control**

```python
MAX_CONCURRENT = 10  # Số trang crawl đồng thời
for i in range(0, len(listing_urls), self.max_concurrent):
    batch = listing_urls[i:i + self.max_concurrent]
    # Process batch
```

> **Why?** Kiểm soát số lượng request đồng thời để không gây quá tải server và tránh bị chặn IP.

**5. Periodic Saving**

```python
if (i + self.max_concurrent) % 20 == 0:
    self._save_to_csv()
```

> **Why?** Lưu dữ liệu định kỳ để tránh mất dữ liệu nếu crawler gặp lỗi giữa chừng.

### 📊 Kết quả

- **Input**: URL trang tìm kiếm Nhatot.com (`https://www.nhatot.com/mua-ban-nha-dat`)
- **Output**: File CSV chứa thông tin bất động sản với 13 features
- **Hiệu suất**: ~2-3 giây/tin đăng với 10 concurrent pages

---

## 2. Chiến lược Huấn luyện (Training Strategy)

### 📍 WHY - Tại sao cần chiến lược huấn luyện đặc biệt?

1. **So sánh nhiều mô hình**: Sử dụng 3 mô hình khác nhau (LightGBM, RandomForest, CatBoost) để tìm ra mô hình tốt nhất.

2. **Tối ưu hyperparameters**: Mỗi mô hình có nhiều hyperparameters cần tinh chỉnh để đạt hiệu suất tốt nhất.

3. **Đánh giá khách quan**: Cần phương pháp đánh giá robust để đảm bảo mô hình generalize tốt.

### 🔧 HOW - Pipeline huấn luyện

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                          │
│  • Load train/test data                                      │
│  • Clean feature names                                       │
│  • Convert categorical columns                               │
│  • Label encode for sklearn models                           │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 OPTUNA HYPERPARAMETER TUNING                 │
│  • 30 trials per model                                       │
│  • 5-Fold CV per trial                                       │
│  • Optimize for minimum RMSE                                 │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               K-FOLD CROSS VALIDATION                        │
│  • Train with optimized params                               │
│  • Compute CV metrics (RMSE, MAE, R²)                        │
│  • Evaluate on test set                                      │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               MODEL SELECTION & SAVING                       │
│  • Compare models, select best                               │
│  • Save all models and metadata                              │
└─────────────────────────────────────────────────────────────┘
```

### 📊 Các mô hình sử dụng

| Model | Library | Xử lý Categorical |
|-------|---------|-------------------|
| **LightGBM** | `lightgbm` | Native `category` dtype |
| **RandomForest** | `sklearn` | Label Encoding |
| **CatBoost** | `catboost` | Native với `cat_features` |

---

## 3. Optuna - Tối ưu Hyperparameters

### 📍 WHY - Tại sao sử dụng Optuna?

1. **Hiệu quả cao hơn Grid Search**: Grid Search thử tất cả tổ hợp → O(n^k). Optuna sử dụng TPE (Tree-structured Parzen Estimator) thông minh hơn.

2. **Tự động pruning**: Dừng sớm các trial kém hiệu quả, tiết kiệm thời gian.

3. **Dễ định nghĩa search space**: API đơn giản với `suggest_int`, `suggest_float`, `suggest_categorical`.

4. **Tích hợp CV**: Kết hợp với Cross-Validation để đánh giá mỗi trial.

### 🔧 HOW - Cách triển khai Optuna

**1. Tạo Objective Function cho mỗi mô hình**

```python
def create_lightgbm_objective(X_train, y_train, n_folds=5):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            # ... more params
        }
        
        model = LGBMRegressor(**param)
        scores = cross_val_score(model, X_train, y_train, cv=n_folds, 
                                 scoring='neg_root_mean_squared_error')
        return -scores.mean()  # Minimize RMSE
    
    return objective
```

**2. Sử dụng TPE Sampler**

```python
sampler = TPESampler(seed=42)  # Reproducible
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=30, show_progress_bar=True)
```

> **Why TPE?** TPE sử dụng Bayesian optimization, học từ các trial trước để chọn hyperparameters cho trial sau thông minh hơn.

**3. CatBoost Native CV**

```python
# CatBoost có CV function riêng, tối ưu hơn sklearn
cv_results = catboost_cv(
    pool=Pool(X_train, y_train, cat_features=cat_features),
    params=param,
    fold_count=5,
    early_stopping_rounds=50  # Dừng sớm nếu không cải thiện
)
```

### 📊 Search Space cho mỗi mô hình

| Model | Hyperparameter | Range | Type |
|-------|----------------|-------|------|
| **LightGBM** | n_estimators | 200 - 2000 | int |
| | learning_rate | 0.001 - 0.3 | log float |
| | max_depth | 3 - 20 | int |
| | num_leaves | 15 - 500 | int |
| **RandomForest** | n_estimators | 100 - 1000 | int |
| | max_depth | 5 - 30 | int |
| | max_features | sqrt, log2, None | categorical |
| **CatBoost** | iterations | 200 - 2000 | int |
| | depth | 4 - 12 | int |
| | l2_leaf_reg | 1e-8 - 100 | log float |

### 🔄 Caching Hyperparameters

```python
# Nếu đã có hyperparams từ lần chạy trước → dùng lại
if os.path.exists('models/best_hyperparams.json'):
    best_params = json.load(f)
    print("✅ Loaded cached hyperparameters")
else:
    # Chạy Optuna optimization
    best_params, studies = optimize_all_models(...)
```

> **Why?** Tiết kiệm thời gian khi re-train, chỉ cần xóa file JSON nếu muốn tìm hyperparams mới.

---

## 4. K-Fold Cross Validation

### 📍 WHY - Tại sao sử dụng K-Fold CV?

1. **Đánh giá robust**: Thay vì split cố định 1 lần, K-Fold CV đánh giá model K lần trên K tập khác nhau.

2. **Tận dụng toàn bộ dữ liệu**: Mỗi sample được dùng làm validation đúng 1 lần.

3. **Ước lượng variance**: Độ lệch chuẩn giữa các fold cho biết model có ổn định không.

4. **Tránh overfitting**: Model không thể "nhớ" validation set vì mỗi fold có validation khác nhau.

### 🔧 HOW - Cách triển khai K-Fold CV

**1. Cấu hình K-Fold**

```python
N_FOLDS = 5
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
```

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| `n_splits` | 5 | Chia dữ liệu thành 5 phần |
| `shuffle` | True | Xáo trộn dữ liệu trước khi chia |
| `random_state` | 42 | Đảm bảo reproducibility |

**2. Training Loop**

```python
cv_scores = {'rmse': [], 'mae': [], 'r2': []}

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model = LGBMRegressor(**params)
    model.fit(X_fold_train, y_fold_train)
    
    y_pred = model.predict(X_fold_val)
    
    cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
    cv_scores['r2'].append(r2_score(y_fold_val, y_pred))
    
print(f"Mean R²: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
```

**3. Minh họa K-Fold (K=5)**

```
Fold 1: [Val] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Val] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Val] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Val] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Val]
```

### 📊 So sánh với các phương pháp khác

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| **Hold-out** | Nhanh | Kết quả phụ thuộc vào cách chia |
| **K-Fold CV** | Robust, ước lượng variance | Chậm hơn K lần |
| **LOO (Leave-One-Out)** | Dùng tối đa dữ liệu | Rất chậm (N lần training) |
| **Stratified K-Fold** | Giữ tỷ lệ class | Chỉ dùng cho classification |

> **Kết luận**: 5-Fold CV là lựa chọn cân bằng giữa tốc độ và độ tin cậy.

### 📈 Kết quả mẫu

```
📊 Training LightGBM...
--------------------------------------------------
   Fold 1: RMSE=0.8234, R²=0.8712
   Fold 2: RMSE=0.7891, R²=0.8845
   Fold 3: RMSE=0.8012, R²=0.8789
   Fold 4: RMSE=0.8456, R²=0.8634
   Fold 5: RMSE=0.7923, R²=0.8801
--------------------------------------------------
   Mean: RMSE=0.8103 ± 0.0217
         R²=0.8756 ± 0.0075
```

---

## 📁 Output Files

### Models (`models/`)

| File | Mô tả |
|------|-------|
| `model.joblib` | Model tốt nhất (production) |
| `lightgbm_optuna_model.joblib` | LightGBM đã optimize |
| `randomforest_optuna_model.joblib` | RandomForest đã optimize |
| `catboost_optuna_model.joblib` | CatBoost đã optimize |
| `best_hyperparams.json` | Hyperparameters tối ưu |
| `cv_scores.json` | Điểm CV cho từng fold |

### Visualizations (`outputs/`)

| File | Mô tả |
|------|-------|
| `optuna_optimization_history.png` | Quá trình tối ưu của Optuna |
| `model_comparison.png` | So sánh RMSE/MAE/R² |
| `cv_scores.png` | R² theo từng fold |
| `training_summary.png` | Tổng kết training |

---

## 🚀 Sử dụng

```bash
# 1. Crawl dữ liệu (nếu cần)
python crawl_nhatot_crawl4ai.py

# 2. Tiền xử lý dữ liệu
# (xem docs/data_processing_guide.md)

# 3. Huấn luyện model
python src/train_model.py

# 4. Chạy ứng dụng
streamlit run app.py
```

---

## 📚 References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Crawl4AI GitHub](https://github.com/unclecode/crawl4ai)
- [scikit-learn Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [CatBoost CV](https://catboost.ai/en/docs/concepts/python-reference_cv)
