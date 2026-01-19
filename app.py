"""
🏠 Streamlit Demo: House Price Prediction
Compatible with train_model.py (Optuna + K-Fold CV)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🏠 Dự Đoán Giá Nhà",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL & DATA
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained model"""
    model = joblib.load('models/model.joblib')
    return model

@st.cache_data
def load_metadata():
    """Load feature names, metrics, and column mapping"""
    with open('models/feature_names.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    with open('models/metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    # Load column mapping if exists
    col_mapping = {}
    if os.path.exists('models/column_mapping.json'):
        with open('models/column_mapping.json', 'r', encoding='utf-8') as f:
            col_mapping = json.load(f)
    
    return feature_names, metrics, col_mapping

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_price(price_ty):
    """Format price in Vietnamese style (tỷ VND)"""
    if price_ty >= 1:
        return f"{price_ty:.2f} tỷ VND"
    else:
        return f"{price_ty * 1000:.0f} triệu VND"

def clean_col_name(name):
    """Clean column name like in training"""
    import re
    new_col = name.replace('(', '_').replace(')', '_').replace(' ', '_')
    new_col = new_col.replace('/', '_').replace(',', '_').replace('.', '_')
    new_col = re.sub(r'[^a-zA-Z0-9_]', '', new_col)
    new_col = re.sub(r'_+', '_', new_col).strip('_')
    return new_col

def create_input_features(inputs, feature_names, col_mapping):
    """Create feature DataFrame from user inputs matching training format"""
    # Initialize features dict
    features = {}
    
    # Basic numeric features
    features['Diện tích (m2)'] = inputs['dien_tich']
    features['Chiều ngang (m)'] = inputs['chieu_ngang']
    features['Chiều dài (m)'] = inputs['chieu_dai']
    features['Số phòng ngủ'] = float(inputs['so_phong_ngu'])
    features['Số phòng vệ sinh'] = float(inputs['so_phong_ve_sinh'])
    features['Số tầng'] = float(inputs['so_tang'])
    
    # Categorical features
    features['Hướng'] = inputs['huong']
    features['Tình trạng nội thất'] = inputs['tinh_trang_noi_that']
    
    # Engineered features (NO Giá_per_m2 - that was data leakage!)
    tong_phong = inputs['so_phong_ngu'] + inputs['so_phong_ve_sinh']
    features['Tổng_phòng'] = tong_phong
    features['Aspect_ratio'] = inputs['chieu_ngang'] / max(inputs['chieu_dai'], 0.1)
    features['Diện_tích_per_phòng'] = inputs['dien_tich'] / max(tong_phong, 1)
    
    # Location encoded features (use location factor as proxy)
    features['Thành phố_encoded'] = inputs['location_factor'] * 2
    features['Phường/Xã_encoded'] = inputs['location_factor'] * 1.5
    
    # One-hot encoding for Loại hình
    loai_hinh_types = ['Nhà biệt thự', 'Nhà mặt phố, mặt tiền', 'Nhà ngõ, hẻm', 'Nhà phố liền kề']
    for lh in loai_hinh_types:
        col_name = f'Loại hình_{lh}'
        features[col_name] = inputs['loai_hinh'] == lh
    
    # Legal document encoding
    giay_to_map = {
        'Sổ đỏ/Sổ hồng': 4,
        'Hợp đồng mua bán': 3,
        'Đang chờ sổ': 2,
        'Giấy tờ khác': 1,
        'Không xác định': 0
    }
    features['Giấy tờ pháp lý_encoded'] = giay_to_map.get(inputs['giay_to'], 4)
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Clean column names to match training
    df_cleaned, _ = clean_feature_names_df(df)
    
    # Ensure all required features exist (fill missing with 0)
    cleaned_feature_names = [clean_col_name(f) for f in feature_names]
    for col in cleaned_feature_names:
        if col not in df_cleaned.columns:
            df_cleaned[col] = 0
    
    # Reorder columns to match training
    df_cleaned = df_cleaned[cleaned_feature_names]
    
    # Convert categorical columns
    cat_cols = ['Hng', 'Tnh_trng_ni_tht']
    for col in cat_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('category')
    
    return df_cleaned

def clean_feature_names_df(df):
    """Clean all column names in DataFrame"""
    import re
    new_columns = {}
    for col in df.columns:
        new_col = col.replace('(', '_').replace(')', '_').replace(' ', '_')
        new_col = new_col.replace('/', '_').replace(',', '_').replace('.', '_')
        new_col = re.sub(r'[^a-zA-Z0-9_]', '', new_col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        new_columns[col] = new_col
    return df.rename(columns=new_columns), new_columns

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("🏠 Dự Đoán Giá Nhà Việt Nam")
    st.markdown("*Sử dụng Machine Learning để ước tính giá nhà*")
    
    # Load resources
    try:
        model = load_model()
        feature_names, metrics, col_mapping = load_metadata()
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        st.info("Hãy chạy `python src/train_model.py` trước để train model!")
        st.code("""
# Bước 1: Train model
python src/train_model.py

# Bước 2: Chạy app
streamlit run app.py
        """)
        return
    
    # Sidebar - Model info
    with st.sidebar:
        st.header("📊 Thông Tin Model")
        best_model = metrics.get('best_model', 'Unknown')
        st.info(f"🏆 Best: **{best_model}**")
        
        st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f} tỷ")
        st.metric("MAE", f"{metrics.get('mae', 0):.4f} tỷ")
        
        if 'cv_folds' in metrics:
            st.divider()
            st.markdown(f"**Cross-Validation:** {metrics['cv_folds']}-Fold")
            st.markdown(f"**Optuna Trials:** {metrics.get('optuna_trials', 'N/A')}")
    
    # Main content - Input form
    st.header("📝 Nhập Thông Tin Căn Nhà")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📐 Kích Thước")
        dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=80.0, step=5.0)
        chieu_ngang = st.number_input("Chiều ngang (m)", min_value=2.0, max_value=50.0, value=5.0, step=0.5)
        chieu_dai = st.number_input("Chiều dài (m)", min_value=5.0, max_value=100.0, value=16.0, step=1.0)
    
    with col2:
        st.subheader("🏠 Cấu Trúc")
        so_phong_ngu = st.number_input("Số phòng ngủ", min_value=1, max_value=10, value=3)
        so_phong_ve_sinh = st.number_input("Số phòng vệ sinh", min_value=1, max_value=10, value=2)
        so_tang = st.number_input("Số tầng", min_value=1, max_value=10, value=3)
    
    with col3:
        st.subheader("📍 Thông Tin Khác")
        loai_hinh = st.selectbox("Loại hình", [
            "Nhà ngõ, hẻm",
            "Nhà mặt phố, mặt tiền", 
            "Nhà phố liền kề",
            "Nhà biệt thự"
        ])
        huong = st.selectbox("Hướng", [
            "Không xác định", "Đông", "Tây", "Nam", "Bắc",
            "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"
        ])
        tinh_trang = st.selectbox("Tình trạng nội thất", [
            "Không xác định",
            "Bàn giao thô",
            "Hoàn thiện cơ bản",
            "Nội thất đầy đủ",
            "Nội thất cao cấp"
        ])
    
    # Additional info
    col4, col5 = st.columns(2)
    
    with col4:
        giay_to = st.selectbox("Giấy tờ pháp lý", [
            "Sổ đỏ/Sổ hồng",
            "Hợp đồng mua bán",
            "Đang chờ sổ",
            "Giấy tờ khác",
            "Không xác định"
        ])
    
    with col5:
        location_factor = st.slider(
            "Mức độ đắt đỏ khu vực",
            min_value=1.0, max_value=10.0, value=6.5, step=0.5,
            help="1=Tỉnh lẻ, 5=Thành phố cấp 2, 10=HN/HCM trung tâm"
        )
    
    st.divider()
    
    # Predict button
    if st.button("🎯 Dự Đoán Giá", type="primary", use_container_width=True):
        inputs = {
            'dien_tich': dien_tich,
            'chieu_ngang': chieu_ngang,
            'chieu_dai': chieu_dai,
            'so_phong_ngu': so_phong_ngu,
            'so_phong_ve_sinh': so_phong_ve_sinh,
            'so_tang': so_tang,
            'loai_hinh': loai_hinh,
            'huong': huong,
            'tinh_trang_noi_that': tinh_trang,
            'giay_to': giay_to,
            'location_factor': location_factor
        }
        
        try:
            # Create feature vector
            X = create_input_features(inputs, feature_names, col_mapping)
            
            # Predict (model outputs directly in tỷ VND)
            y_pred = model.predict(X)[0]
            
            # Ensure positive prediction
            y_pred = max(0.1, y_pred)
            
            # Display result
            st.success("✅ Dự đoán thành công!")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric(
                    label="💰 Giá Dự Đoán",
                    value=format_price(y_pred)
                )
            
            with col_result2:
                price_per_m2 = y_pred / dien_tich * 1_000_000_000  # Convert to VND/m²
                st.metric(
                    label="📊 Giá/m²",
                    value=f"{price_per_m2/1e6:.1f} triệu/m²"
                )
            
            # Price range (±15%)
            st.info(f"""
            📈 **Khoảng giá ước tính:** {format_price(y_pred * 0.85)} - {format_price(y_pred * 1.15)}
            
            ⚠️ *Đây chỉ là ước tính dựa trên dữ liệu học máy. Giá thực tế có thể khác.*
            """)
            
        except Exception as e:
            st.error(f"❌ Lỗi dự đoán: {e}")
            with st.expander("Chi tiết lỗi"):
                st.exception(e)
    
    # Footer
    st.divider()
    st.caption("🏠 House Price Prediction | Built with Streamlit & LightGBM/CatBoost/RandomForest")

if __name__ == "__main__":
    main()
