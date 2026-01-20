# 🚀 Cách Deploy lên Streamlit Cloud

Dưới đây là hướng dẫn chi tiết để đưa ứng dụng **Dự Đoán Giá Nhà** lên internet để mọi người cùng sử dụng.

## 1. Chuẩn bị

Đảm bảo project của bạn đã có:
1. **Source code**: `app.py`, folder `src/`, `models/` (chứa file `.joblib` và `.json`).
2. **Dependencies**: Đảm bảo khai báo thư viện (ví dụ: `requirements.txt` hoặc `pyproject.toml`).
   - *Lưu ý: Bạn đã chọn tự quản lý file này.*

## 2. Đẩy code lên GitHub

Nếu bạn chưa có repository trên GitHub:

1. Tạo repository mới trên [GitHub](https://github.com/new).
2. Chạy các lệnh sau tại thư mục dự án của bạn (Terminal):

```bash
# Khởi tạo git (nếu chưa có)
git init

# Thêm tất cả file (lưu ý .gitignore đã được cấu hình để gửi file model)
git add .

# Commit code
git commit -m "Deploy housing prediction app"

# Link tới repository GitHub của bạn (thay URL bên dưới bằng URL của bạn)
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 3. Deploy trên Streamlit Cloud

Streamlit Cloud là nền tảng miễn phí và dễ nhất để host ứng dụng Streamlit.

1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
2. Nhấn nút **"New app"**.
3. Chọn Repository bạn vừa đẩy lên.
4. Cấu hình:
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Nhấn **"Deploy!"**.

## 4. Xử lý sự cố thường gặp

### Lỗi "ModuleNotFoundError"
Nếu app báo lỗi thiếu thư viện (ví dụ `ModuleNotFoundError: No module named 'catboost'`), nghĩa là file khai báo dependencies của bạn (requirements.txt/pyproject.toml) thiếu thư viện đó. Hãy bổ sung và push lại lên GitHub.

### Lỗi không tìm thấy Model
Đảm bảo folder `models/` và file `model.joblib` đã được push lên GitHub. Kiểm tra trên website GitHub xem folder này có tồn tại không.

### App chạy chậm
Model CatBoost/LightGBM load lần đầu có thể mất vài giây. Streamlit Cloud sẽ cache lại (nhờ decorator `@st.cache_resource`) nên các lần sau sẽ nhanh hơn.
