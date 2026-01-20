"""
🏠 Streamlit Demo: House Price Prediction
Minimalist UI/UX Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import time

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
# CUSTOM CSS - MINIMALIST STYLE
# ============================================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.03);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        letter-spacing: -0.025em;
    }
    
    h2 {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
    }
    
    h3 {
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    /* Input labels */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    
    /* Input fields */
    .stNumberInput input, .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid #e2e8f0 !important;
        background: #f8fafc !important;
        transition: all 0.2s ease !important;
    }
    
    .stNumberInput input:focus, .stSelectbox > div > div:focus-within {
        border-color: #14b8a6 !important;
        box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.1) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.25);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.35) !important;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Metric container */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        border-radius: 12px !important;
        border: none !important;
    }
    
    /* Divider */
    hr {
        border-color: #e2e8f0 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Custom card styling */
    .prediction-card {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(20, 184, 166, 0.3);
    }
    
    .model-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .model-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #14b8a6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & DATA
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'LightGBM': 'models/lightgbm_optuna_model.joblib',
        'RandomForest': 'models/randomforest_optuna_model.joblib',
        'CatBoost': 'models/catboost_optuna_model.joblib'
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_metadata():
    """Load feature names, metrics, and column mapping"""
    with open('models/feature_names.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    with open('models/metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
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
        return f"{price_ty:.2f} tỷ"
    else:
        return f"{price_ty * 1000:.0f} triệu"

def clean_col_name(name):
    """Clean column name like in training"""
    import re
    new_col = name.replace('(', '_').replace(')', '_').replace(' ', '_')
    new_col = new_col.replace('/', '_').replace(',', '_').replace('.', '_')
    new_col = re.sub(r'[^a-zA-Z0-9_]', '', new_col)
    new_col = re.sub(r'_+', '_', new_col).strip('_')
    return new_col

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

def create_input_features(inputs, feature_names, col_mapping):
    """Create feature DataFrame from user inputs matching training format"""
    features = {}
    
    # Basic numeric features
    features['Diện tích (m2)'] = inputs['dien_tich']
    features['Chiều ngang (m)'] = inputs['chieu_ngang']
    features['Chiều dài (m)'] = inputs['chieu_dai']
    features['Số tầng'] = float(inputs['so_tang'])
    
    # Categorical features
    features['Hướng'] = inputs['huong']
    features['Tình trạng nội thất'] = inputs['tinh_trang_noi_that']
    
    # Engineered features
    features['Tổng_phòng'] = inputs['tong_phong']
    
    # Location encoded features
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
    
    df = pd.DataFrame([features])
    df_cleaned, _ = clean_feature_names_df(df)
    
    cleaned_feature_names = [clean_col_name(f) for f in feature_names]
    for col in cleaned_feature_names:
        if col not in df_cleaned.columns:
            df_cleaned[col] = 0
    
    df_cleaned = df_cleaned[cleaned_feature_names]
    
    cat_cols = ['Hng', 'Tnh_trng_ni_tht']
    for col in cat_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('category')
    
    return df_cleaned

def create_price_trend_chart(predictions, dien_tich):
    """Create a clear and beautiful price comparison chart"""
    import plotly.graph_objects as go
    
    models = list(predictions.keys())
    prices = [predictions[m] for m in models]
    
    # Model icons and colors
    model_config = {
        'LightGBM': {'color': '#14b8a6', 'icon': '⚡'},
        'RandomForest': {'color': '#0ea5e9', 'icon': '🌲'},
        'CatBoost': {'color': '#8b5cf6', 'icon': '🐱'}
    }
    
    colors = [model_config.get(m, {'color': '#64748b'})['color'] for m in models]
    icons = [model_config.get(m, {'icon': '🤖'})['icon'] for m in models]
    
    # Create labels with icons
    labels = [f"{icons[i]} {m}" for i, m in enumerate(models)]
    
    fig = go.Figure()
    
    # Horizontal bar chart - easier to read
    fig.add_trace(go.Bar(
        y=labels,
        x=prices,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=8
        ),
        text=[f"<b>{p:.2f} tỷ</b>  ({p/dien_tich*1000:.1f} tr/m²)" for p in prices],
        textposition='outside',
        textfont=dict(size=14, color='#1e293b', family='Inter'),
        hovertemplate='<b>%{y}</b><br>Giá: %{x:.2f} tỷ VND<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Inter', color='#334155', size=13),
        showlegend=False,
        margin=dict(l=10, r=120, t=30, b=30),
        height=200,
        xaxis=dict(
            showgrid=True,
            gridcolor='#f1f5f9',
            gridwidth=1,
            title=dict(text='Giá (tỷ VND)', font=dict(size=12, color='#64748b')),
            zeroline=False,
            tickfont=dict(size=11, color='#64748b'),
        ),
        yaxis=dict(
            showgrid=False,
            title='',
            tickfont=dict(size=14, color='#1e293b'),
            automargin=True,
        ),
        bargap=0.35,
    )
    
    # Add range annotation
    min_price = min(prices)
    max_price = max(prices)
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # ===== SIDEBAR - INPUT FORM =====
    with st.sidebar:
        st.markdown("### 🏠 Thông tin căn nhà")
        st.markdown("---")
        
        # Size section
        st.markdown("##### 📐 Kích thước")
        dien_tich = st.number_input(
            "Diện tích (m²)", 
            min_value=10.0, max_value=1000.0, value=80.0, step=5.0
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            chieu_ngang = st.number_input(
                "Ngang (m)", 
                min_value=2.0, max_value=50.0, value=5.0, step=0.5
            )
        with col_b:
            chieu_dai = st.number_input(
                "Dài (m)", 
                min_value=5.0, max_value=100.0, value=16.0, step=1.0
            )
        
        st.markdown("")
        
        # Structure section
        st.markdown("##### 🏗️ Cấu trúc")
        col_c, col_d = st.columns(2)
        with col_c:
            tong_phong = st.number_input(
                "Số phòng", 
                min_value=1, max_value=20, value=5,
                help="Tổng số phòng ngủ + WC"
            )
        with col_d:
            so_tang = st.number_input(
                "Số tầng", 
                min_value=1, max_value=10, value=3
            )
        
        st.markdown("")
        
        # Type section
        st.markdown("##### 🏘️ Loại hình & Pháp lý")
        loai_hinh = st.selectbox(
            "Loại hình nhà",
            ["Nhà ngõ, hẻm", "Nhà mặt phố, mặt tiền", "Nhà phố liền kề", "Nhà biệt thự"],
            label_visibility="collapsed"
        )
        
        giay_to = st.selectbox(
            "Giấy tờ pháp lý",
            ["Sổ đỏ/Sổ hồng", "Hợp đồng mua bán", "Đang chờ sổ", "Giấy tờ khác", "Không xác định"],
            label_visibility="collapsed"
        )
        
        st.markdown("")
        
        # Other info
        st.markdown("##### 🧭 Thông tin khác")
        huong = st.selectbox(
            "Hướng nhà",
            ["Không xác định", "Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"],
            label_visibility="collapsed"
        )
        
        tinh_trang = st.selectbox(
            "Nội thất",
            ["Không xác định", "Bàn giao thô", "Hoàn thiện cơ bản", "Nội thất đầy đủ", "Nội thất cao cấp"],
            label_visibility="collapsed"
        )
        
        st.markdown("")
        
        # Location factor
        st.markdown("##### 📍 Vị trí")
        location_factor = st.slider(
            "Mức độ đắt đỏ khu vực",
            min_value=1.0, max_value=10.0, value=6.5, step=0.5,
            help="1 = Tỉnh lẻ → 10 = Trung tâm HN/HCM"
        )
        
        st.markdown("---")
        
        # Predict button
        predict_clicked = st.button("🎯 Dự đoán giá", type="primary", use_container_width=True)
    
    # ===== MAIN CONTENT - RESULTS =====
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="margin-bottom: 0.5rem;">🏠 Dự Đoán Giá Nhà</h1>
        <p style="color: #64748b; font-size: 1rem;">
            Sử dụng Machine Learning để ước tính giá trị bất động sản
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    try:
        models = load_models()
        feature_names, metrics, col_mapping = load_metadata()
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        st.info("Hãy chạy `python src/train_model.py` trước để train model!")
        return
    
    if not models:
        st.error("❌ Không tìm thấy model nào!")
        return
    
    # Show model info
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("🏆 Best Model", metrics.get('best_model', 'N/A'))
    with col_info2:
        st.metric("📊 R² Score", f"{metrics.get('r2', 0):.4f}")
    with col_info3:
        st.metric("📉 RMSE", f"{metrics.get('rmse', 0):.3f} tỷ")
    
    st.markdown("---")
    
    # Results area
    if predict_clicked:
        inputs = {
            'dien_tich': dien_tich,
            'chieu_ngang': chieu_ngang,
            'chieu_dai': chieu_dai,
            'tong_phong': tong_phong,
            'so_tang': so_tang,
            'loai_hinh': loai_hinh,
            'huong': huong,
            'tinh_trang_noi_that': tinh_trang,
            'giay_to': giay_to,
            'location_factor': location_factor
        }
        
        # Loading effect
        with st.spinner("🔄 Đang phân tích dữ liệu..."):
            time.sleep(0.8)  # Simulate processing
            
            try:
                X = create_input_features(inputs, feature_names, col_mapping)
                
                predictions = {}
                cat_cols = ['Hng', 'Tnh_trng_ni_tht']
                
                for model_name, model in models.items():
                    if model_name == 'RandomForest':
                        X_rf = X.copy()
                        for col in cat_cols:
                            if col in X_rf.columns:
                                X_rf[col] = X_rf[col].astype('category').cat.codes
                        y_pred = model.predict(X_rf)[0]
                    else:
                        y_pred = model.predict(X)[0]
                    predictions[model_name] = max(0.1, y_pred)
                
                # Best prediction (from best model)
                best_model = metrics.get('best_model', list(predictions.keys())[0])
                best_pred = predictions.get(best_model, list(predictions.values())[0])
                
                # Main result card
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
                    border-radius: 24px;
                    padding: 2.5rem;
                    text-align: center;
                    color: white;
                    box-shadow: 0 20px 60px rgba(20, 184, 166, 0.3);
                    margin-bottom: 2rem;
                ">
                    <p style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">
                        💰 Giá dự đoán ({best_model})
                    </p>
                    <h1 style="
                        font-size: 3.5rem; 
                        font-weight: 700; 
                        margin: 0.5rem 0;
                        color: white !important;
                        letter-spacing: -0.02em;
                    ">
                        {format_price(best_pred)} VND
                    </h1>
                    <p style="font-size: 1.1rem; opacity: 0.85; margin-top: 1rem;">
                        📐 {best_pred / dien_tich * 1000:.1f} triệu/m² • {dien_tich:.0f}m²
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model comparison
                st.markdown("### 📊 So sánh các mô hình")
                
                cols = st.columns(3)
                model_icons = {'LightGBM': '⚡', 'RandomForest': '🌲', 'CatBoost': '🐱'}
                
                for i, (model_name, y_pred) in enumerate(predictions.items()):
                    with cols[i]:
                        icon = model_icons.get(model_name, '🤖')
                        is_best = model_name == best_model
                        border_style = "2px solid #14b8a6" if is_best else "1px solid #e2e8f0"
                        
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 16px;
                            padding: 1.5rem;
                            border: {border_style};
                            text-align: center;
                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
                        ">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.25rem;">
                                {model_name} {"🏆" if is_best else ""}
                            </div>
                            <div style="color: #0f172a; font-size: 1.5rem; font-weight: 600;">
                                {format_price(y_pred)}
                            </div>
                            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.25rem;">
                                {y_pred / dien_tich * 1000:.1f} triệu/m²
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("")
                
                # Price trend chart
                st.markdown("### 📈 Biểu đồ so sánh giá")
                fig = create_price_trend_chart(predictions, dien_tich)
                st.plotly_chart(fig, use_container_width=True)
                
                # Range info
                min_pred = min(predictions.values())
                max_pred = max(predictions.values())
                
                st.markdown(f"""
                <div style="
                    background: #f1f5f9;
                    border-radius: 12px;
                    padding: 1rem 1.5rem;
                    text-align: center;
                    color: #475569;
                    margin-top: 1rem;
                ">
                    📊 <strong>Khoảng giá:</strong> {format_price(min_pred)} - {format_price(max_pred)} VND
                    <br>
                    <span style="font-size: 0.85rem; color: #94a3b8;">
                        ⚠️ Đây chỉ là ước tính từ ML. Giá thực tế phụ thuộc nhiều yếu tố khác.
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Lỗi dự đoán: {e}")
                with st.expander("Chi tiết lỗi"):
                    st.exception(e)
    else:
        # Empty state
        st.markdown("""
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            background: white;
            border-radius: 20px;
            border: 2px dashed #e2e8f0;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏡</div>
            <h3 style="color: #334155; margin-bottom: 0.5rem;">Nhập thông tin căn nhà</h3>
            <p style="color: #94a3b8;">
                Điền thông tin ở thanh bên trái và nhấn "Dự đoán giá" để xem kết quả
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem 0;">
        🏠 House Price Prediction • Built with Streamlit & ML
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
