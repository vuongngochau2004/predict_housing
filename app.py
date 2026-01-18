"""
🏠 Streamlit Demo: House Price Prediction
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
    """Load feature names and metrics"""
    with open('models/feature_names.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    with open('models/metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return feature_names, metrics

@st.cache_data
def load_reference_data():
    """Load data for reference (city prices, etc)"""
    df = pd.read_csv('data/gia_nha_train.csv')
    return df

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_price(price_vnd):
    """Format price in Vietnamese style"""
    if price_vnd >= 1e9:
        return f"{price_vnd/1e9:.2f} tỷ VND"
    else:
        return f"{price_vnd/1e6:.0f} triệu VND"

def create_input_features(inputs, feature_names, ref_df):
    """Create feature vector from user inputs"""
    # Start with zeros
    features = {name: 0.0 for name in feature_names}
    
    # Basic numeric features
    features['Diện tích (m2)'] = inputs['dien_tich']
    features['Chiều ngang (m)'] = inputs['chieu_ngang']
    features['Chiều dài (m)'] = inputs['chieu_dai']
    features['Số phòng ngủ'] = inputs['so_phong_ngu']
    features['Số phòng vệ sinh'] = inputs['so_phong_ve_sinh']
    features['Số tầng'] = inputs['so_tang']
    
    # Engineered features
    features['Tổng_phòng'] = inputs['so_phong_ngu'] + inputs['so_phong_ve_sinh']
    features['Aspect_ratio'] = inputs['chieu_ngang'] / max(inputs['chieu_dai'], 0.1)
    features['Diện_tích_per_phòng'] = inputs['dien_tich'] / max(features['Tổng_phòng'], 1)
    
    # One-hot encoding for Loại hình
    loai_hinh_cols = [c for c in feature_names if c.startswith('Loại hình_')]
    for col in loai_hinh_cols:
        if inputs['loai_hinh'] in col:
            features[col] = True
    
    # One-hot encoding for Hướng
    huong_cols = [c for c in feature_names if c.startswith('Hướng_')]
    for col in huong_cols:
        if inputs['huong'] in col:
            features[col] = True
    
    # One-hot encoding for Giấy tờ
    giay_to_cols = [c for c in feature_names if c.startswith('Giấy tờ pháp lý_')]
    for col in giay_to_cols:
        if inputs['giay_to'] in col:
            features[col] = True
    
    # One-hot encoding for Nội thất
    noi_that_cols = [c for c in feature_names if c.startswith('Tình trạng nội thất_')]
    for col in noi_that_cols:
        if inputs['noi_that'] in col:
            features[col] = True
    
    # Target encoding for Thành phố
    if 'Thành phố_encoded' in feature_names:
        # Use average from training data
        city_avg = ref_df.groupby('Thành phố_encoded').size().index.mean()
        features['Thành phố_encoded'] = inputs.get('thanh_pho_encoded', city_avg)
    
    # Target encoding for Phường/Xã
    if 'Phường/Xã_encoded' in feature_names:
        phuong_avg = ref_df.groupby('Phường/Xã_encoded').size().index.mean() if 'Phường/Xã_encoded' in ref_df.columns else 5e9
        features['Phường/Xã_encoded'] = phuong_avg
    
    # Log transformed features (will be computed from prediction)
    features['Diện tích (m2)_log'] = np.log1p(inputs['dien_tich'])
    
    # Return as DataFrame with correct column order
    return pd.DataFrame([features])[feature_names]

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
        feature_names, metrics = load_metadata()
        ref_df = load_reference_data()
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        st.info("Hãy chạy `python src/train_model.py` trước!")
        return
    
    # Sidebar - Model info
    with st.sidebar:
        st.header("📊 Thông Tin Model")
        st.metric("MAE", f"{metrics['mae_billion']:.2f} tỷ")
        st.metric("R² Score", f"{metrics['r2']:.3f}")
        st.metric("MAPE", f"{metrics['mape']:.1f}%")
        
        st.divider()
        st.markdown("**Được train trên:**")
        st.markdown("- 4,397 căn nhà")
        st.markdown("- 33 features")
        st.markdown("- RandomForest model")
    
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
            "Biệt thự"
        ])
        huong = st.selectbox("Hướng", [
            "Không xác định", "Đông", "Tây", "Nam", "Bắc",
            "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"
        ])
        giay_to = st.selectbox("Giấy tờ pháp lý", [
            "Đã có sổ", "Đang chờ sổ", "Sổ chung / công chứng vi bằng", "Không có sổ"
        ])
        noi_that = st.selectbox("Tình trạng nội thất", [
            "Không xác định", "Hoàn thiện cơ bản", "Nội thất đầy đủ", "Nội thất cao cấp"
        ])
    
    # Location price factor
    st.subheader("📍 Vị Trí (Ảnh hưởng lớn đến giá)")
    location_factor = st.slider(
        "Mức độ đắt đỏ của khu vực (1=Tỉnh lẻ, 5=Trung tâm HN/HCM)",
        min_value=1, max_value=5, value=3
    )
    
    # Map location factor to approximate encoded value
    location_encoded_map = {
        1: 2e9,   # Tỉnh lẻ
        2: 4e9,   # Ngoại thành
        3: 6e9,   # Thành phố cấp 2
        4: 10e9,  # HN/HCM ngoại thành
        5: 15e9   # HN/HCM trung tâm
    }
    
    st.divider()
    
    # Predict button
    if st.button("🎯 Dự Đoán Giá", type="primary", use_container_width=True):
        # Prepare inputs
        inputs = {
            'dien_tich': dien_tich,
            'chieu_ngang': chieu_ngang,
            'chieu_dai': chieu_dai,
            'so_phong_ngu': so_phong_ngu,
            'so_phong_ve_sinh': so_phong_ve_sinh,
            'so_tang': so_tang,
            'loai_hinh': loai_hinh,
            'huong': huong,
            'giay_to': giay_to,
            'noi_that': noi_that,
            'thanh_pho_encoded': location_encoded_map[location_factor]
        }
        
        try:
            # Create feature vector
            X = create_input_features(inputs, feature_names, ref_df)
            
            # Predict (log scale)
            y_pred_log = model.predict(X)[0]
            
            # Convert to VND
            y_pred_vnd = np.expm1(y_pred_log)
            
            # Adjust by location factor
            y_pred_vnd = y_pred_vnd * (0.5 + location_factor * 0.2)
            
            # Display result
            st.success("✅ Dự đoán thành công!")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.metric(
                    label="💰 Giá Dự Đoán",
                    value=format_price(y_pred_vnd),
                    delta=None
                )
            
            with col_result2:
                price_per_m2 = y_pred_vnd / dien_tich
                st.metric(
                    label="📊 Giá/m²",
                    value=format_price(price_per_m2).replace(" VND", "/m²")
                )
            
            # Price range
            st.info(f"""
            📈 **Khoảng giá ước tính:** {format_price(y_pred_vnd * 0.85)} - {format_price(y_pred_vnd * 1.15)}
            
            ⚠️ *Đây chỉ là ước tính dựa trên dữ liệu thị trường. Giá thực tế có thể khác tùy thuộc vào nhiều yếu tố.*
            """)
            
        except Exception as e:
            st.error(f"❌ Lỗi dự đoán: {e}")
            st.exception(e)
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **🏠 House Price Prediction Demo**  
    Built with Streamlit & Scikit-learn | Data: Vietnam Real Estate  
    """)

if __name__ == "__main__":
    main()
