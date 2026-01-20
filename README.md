# 🏠 Hướng Dẫn Sử Dụng - House Price Prediction

## 1. Cài Đặt Dependencies

```bash
# Tạo virtual environment (khuyến nghị)
uv venv
source .venv/bin/activate  # macOS/Linux

# Cài đặt packages
uv sync
```

## 2. Train Model

### Chuẩn bị data
Đảm bảo có các file sau trong thư mục `data/`:
- `train_data.csv` - Dữ liệu training
- `test_data.csv` - Dữ liệu test

### Chạy training

```bash
uv run python src/train_model.py
```

**Thời gian ước tính:** ~15-30 phút (tùy thuộc vào cấu hình máy)

**Output:** Thư mục `models/` được tạo với:
- `model.joblib` - Model tốt nhất
- `lightgbm_optuna_model.joblib`
- `randomforest_optuna_model.joblib`
- `catboost_optuna_model.joblib`
- `best_hyperparams.json` - Hyperparameters đã tối ưu
- `cv_scores.json` - Điểm K-Fold CV
- `metrics.json` - Metrics của model tốt nhất

## 3. Chạy Ứng Dụng Streamlit

```bash
uv run streamlit run app.py
```

App sẽ tự động mở trong browser tại `http://localhost:8501`

## 4. Sử Dụng App

1. **Nhập thông tin căn nhà:**
   - Diện tích, chiều ngang, chiều dài
   - Số phòng ngủ, phòng vệ sinh, số tầng
   - Loại hình, hướng, tình trạng nội thất

2. **Chọn mức độ đắt đỏ khu vực** (1-10):
   - 1-3: Tỉnh lẻ
   - 4-6: Thành phố cấp 2
   - 7-10: Hà Nội/TP.HCM

3. **Nhấn "Dự Đoán Giá"**

## 5. Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `FileNotFoundError: model.joblib` | Chạy `python src/train_model.py` trước |
| `ModuleNotFoundError: lightgbm` | Chạy `pip install lightgbm` |
| Prediction quá cao/thấp | Điều chỉnh "Mức độ đắt đỏ khu vực" |

## 6. Cấu Hình Training

Sửa các tham số trong `src/train_model.py`:

```python
N_FOLDS = 5           # Số fold cho Cross-Validation
N_OPTUNA_TRIALS = 30  # Số trials Optuna (tăng = chính xác hơn, lâu hơn)
RANDOM_STATE = 42     # Random seed
```

