"""
📊 HƯỚNG DẪN VISUALIZATION VỚI SEABORN
Dự án: Phân Tích Giá Bất Động Sản

Author: Data Science Expert
Purpose: Visualization để hiểu data và quyết định preprocessing strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Cấu hình font tiếng Việt
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Cấu hình style
sns.set_palette("husl")
sns.set_style("whitegrid")

# ============================================================================
# BƯỚC 1: LOAD & CLEAN DỮ LIỆU
# ============================================================================

def parse_price(price_str):
    """
    Parse giá từ text tiếng Việt sang số (VND)
    
    Examples:
        "3,5 tỷ" -> 3_500_000_000
        "850 triệu" -> 850_000_000
        "12 tỷ" -> 12_000_000_000
    """
    if pd.isna(price_str) or price_str == '':
        return np.nan
    
    price_str = str(price_str).strip().lower()
    
    try:
        # Tách số và đơn vị
        if 'tỷ' in price_str:
            number = price_str.replace('tỷ', '').replace(',', '.').strip()
            return float(number) * 1_000_000_000
        elif 'triệu' in price_str:
            number = price_str.replace('triệu', '').replace(',', '.').strip()
            return float(number) * 1_000_000
        else:
            # Trường hợp chỉ có số (giả định tỷ)
            return float(price_str.replace(',', '.'))
    except:
        return np.nan


def clean_numeric_column(series):
    """
    Clean numeric columns that may contain string values
    
    Examples:
        "nhiều hơn 10" -> 10
        "5" -> 5
        np.nan -> np.nan
    """
    def convert_value(val):
        if pd.isna(val):
            return np.nan
        
        # Convert to string and check for special cases
        val_str = str(val).strip().lower()
        
        # Handle "nhiều hơn X" pattern
        if 'nhiều hơn' in val_str or 'nhieu hon' in val_str:
            # Extract number after "nhiều hơn"
            import re
            numbers = re.findall(r'\d+', val_str)
            if numbers:
                return float(numbers[0])
            return np.nan
        
        # Try to convert directly to float
        try:
            return float(val_str)
        except:
            return np.nan
    
    return series.apply(convert_value)


def load_and_clean_data(filepath):
    """
    Load và clean data cơ bản
    
    Returns:
        df: DataFrame đã clean
    """
    print("📁 Đang load dữ liệu...")
    df = pd.read_csv(filepath)
    
    # Xóa các dòng hoàn toàn rỗng
    df = df.dropna(how='all')
    
    # Parse giá bán
    print("💰 Đang parse giá bán...")
    df['Giá bán_numeric'] = df['Giá bán'].apply(parse_price)
    
    # Rename columns để dễ làm việc
    column_mapping = {
        'Diện tích (m2)': 'Diện tích',
        'Chiều ngang (m)': 'Chiều ngang',
        'Chiều dài (m)': 'Chiều dài',
        'Giấy tờ pháp lý': 'Giấy tờ',
        'Tình trạng nội thất': 'Nội thất'
    }
    df = df.rename(columns=column_mapping)
    
    # Drop nếu thiếu giá hoặc diện tích
    initial_count = len(df)
    df = df.dropna(subset=['Giá bán_numeric', 'Diện tích'])
    print(f"✂️ Đã loại bỏ {initial_count - len(df)} dòng thiếu giá/diện tích")
    
    # Clean numeric columns that may have string values
    print("🧹 Đang clean các cột numeric...")
    numeric_cols = ['Số phòng ngủ', 'Số phòng vệ sinh', 'Số tầng', 'Chiều ngang', 'Chiều dài']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Tạo feature Giá/m²
    df['Giá_per_m2'] = df['Giá bán_numeric'] / df['Diện tích']
    
    # Log transform
    df['Giá bán_log'] = np.log1p(df['Giá bán_numeric'])
    df['Diện tích_log'] = np.log1p(df['Diện tích'])
    
    print(f"✅ Dữ liệu sạch: {len(df)} records")
    print(f"📊 Số cột: {len(df.columns)}")
    
    return df


# ============================================================================
# BƯỚC 2: PHÂN PHỐI GIÁ (Để quyết định scaling)
# ============================================================================

def plot_price_distribution(df):
    """
    📈 VISUALIZATION 1: PHÂN PHỐI GIÁ
    
    MỤC ĐÍCH:
    - Xem phân phối giá bán có skewed không
    - Quyết định dùng Log Transform hay StandardScaler
    - Phát hiện outliers
    
    GIẢI THÍCH:
    - Histogram: Xem tần suất của mỗi khoảng giá
    - KDE (Kernel Density Estimation): Smooth version của histogram
    - Boxplot: Phát hiện outliers (điểm ngoài whiskers)
    - Q-Q plot: So sánh với phân phối chuẩn
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram + KDE - Giá gốc
    axes[0, 0].set_title('Phân phối Giá bán (Original)', fontsize=14, weight='bold')
    sns.histplot(data=df, x='Giá bán_numeric', bins=50, kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_xlabel('Giá bán (VND)', fontsize=12)
    axes[0, 0].set_ylabel('Tần suất', fontsize=12)
    axes[0, 0].ticklabel_format(style='plain', axis='x')
    
    # Thêm thống kê
    mean_price = df['Giá bán_numeric'].mean()
    median_price = df['Giá bán_numeric'].median()
    axes[0, 0].axvline(mean_price, color='red', linestyle='--', label=f'Mean: {mean_price/1e9:.2f} tỷ')
    axes[0, 0].axvline(median_price, color='green', linestyle='--', label=f'Median: {median_price/1e9:.2f} tỷ')
    axes[0, 0].legend()
    
    # 2. Histogram + KDE - Giá log
    axes[0, 1].set_title('Phân phối Giá bán (Log Transform)', fontsize=14, weight='bold')
    sns.histplot(data=df, x='Giá bán_log', bins=50, kde=True, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_xlabel('log(Giá bán)', fontsize=12)
    axes[0, 1].set_ylabel('Tần suất', fontsize=12)
    
    # 3. Boxplot - Giá gốc
    axes[1, 0].set_title('Boxplot Giá bán (Original)', fontsize=14, weight='bold')
    sns.boxplot(data=df, y='Giá bán_numeric', ax=axes[1, 0], color='lightblue')
    axes[1, 0].set_ylabel('Giá bán (VND)', fontsize=12)
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    
    # 4. Boxplot - Giá log
    axes[1, 1].set_title('Boxplot Giá bán (Log Transform)', fontsize=14, weight='bold')
    sns.boxplot(data=df, y='Giá bán_log', ax=axes[1, 1], color='lightsalmon')
    axes[1, 1].set_ylabel('log(Giá bán)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('outputs/1_price_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: outputs/1_price_distribution.png")
    plt.show()
    
    # In thống kê
    print("\n" + "="*60)
    print("📊 THỐNG KÊ GIÁ BÁN")
    print("="*60)
    print(f"Mean (Trung bình):     {df['Giá bán_numeric'].mean()/1e9:.2f} tỷ")
    print(f"Median (Trung vị):     {df['Giá bán_numeric'].median()/1e9:.2f} tỷ")
    print(f"Std (Độ lệch chuẩn):   {df['Giá bán_numeric'].std()/1e9:.2f} tỷ")
    print(f"Min:                   {df['Giá bán_numeric'].min()/1e9:.2f} tỷ")
    print(f"Max:                   {df['Giá bán_numeric'].max()/1e9:.2f} tỷ")
    print(f"\n🔍 Skewness (Độ lệch): {df['Giá bán_numeric'].skew():.2f}")
    print(f"   (> 1: right-skewed → NÊN DÙNG LOG TRANSFORM)")
    print(f"\n🔍 Skewness sau Log:   {df['Giá bán_log'].skew():.2f}")
    print(f"   (gần 0: symmetric → phân phối chuẩn hơn)")
    print("="*60)


# ============================================================================
# BƯỚC 3: GIÁ VS DIỆN TÍCH THEO THÀNH PHỐ
# ============================================================================

def plot_price_vs_area_by_city(df):
    """
    📈 VISUALIZATION 2: GIÁ VS DIỆN TÍCH THEO THÀNH PHỐ
    
    MỤC ĐÍCH:
    - Xem mối quan hệ giữa Giá và Diện tích
    - So sánh giá giữa các thành phố
    - Hiểu slope khác nhau (giá/m² khác nhau)
    
    GIẢI THÍCH:
    - Scatterplot: Mỗi điểm = 1 căn nhà
    - Regression line: Xu hướng tuyến tính
    - Color by city: So sánh các thành phố
    - Log scale: Dễ nhìn hơn với data skewed
    """
    
    # Lấy top 5 thành phố có nhiều listing nhất
    top_cities = df['Thành phố'].value_counts().head(5).index.tolist()
    df_top = df[df['Thành phố'].isin(top_cities)].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1. Scatterplot - Original scale
    axes[0].set_title('Giá vs Diện tích theo Thành phố (Original)', fontsize=14, weight='bold')
    for city in top_cities:
        city_data = df_top[df_top['Thành phố'] == city]
        sns.scatterplot(data=city_data, x='Diện tích', y='Giá bán_numeric', 
                       label=city, alpha=0.6, s=50, ax=axes[0])
    
    axes[0].set_xlabel('Diện tích (m²)', fontsize=12)
    axes[0].set_ylabel('Giá bán (VND)', fontsize=12)
    axes[0].legend(title='Thành phố', fontsize=10)
    axes[0].ticklabel_format(style='plain', axis='y')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Scatterplot - Log scale với regression line
    axes[1].set_title('Giá vs Diện tích (Log scale) + Regression', fontsize=14, weight='bold')
    for city in top_cities:
        city_data = df_top[df_top['Thành phố'] == city]
        sns.regplot(data=city_data, x='Diện tích_log', y='Giá bán_log', 
                   label=city, scatter_kws={'alpha': 0.5, 's': 40}, 
                   line_kws={'linewidth': 2}, ax=axes[1])
    
    axes[1].set_xlabel('log(Diện tích)', fontsize=12)
    axes[1].set_ylabel('log(Giá bán)', fontsize=12)
    axes[1].legend(title='Thành phố', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/2_price_vs_area_by_city.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: outputs/2_price_vs_area_by_city.png")
    plt.show()
    
    # In thống kê theo thành phố
    print("\n" + "="*60)
    print("📊 THỐNG KÊ THEO THÀNH PHỐ (Top 5)")
    print("="*60)
    for city in top_cities:
        city_data = df_top[df_top['Thành phố'] == city]
        avg_price_per_m2 = city_data['Giá_per_m2'].median()
        print(f"\n{city}:")
        print(f"  - Số lượng: {len(city_data)} listings")
        print(f"  - Giá TB: {city_data['Giá bán_numeric'].mean()/1e9:.2f} tỷ")
        print(f"  - Giá/m² (median): {avg_price_per_m2/1e6:.1f} triệu/m²")
    print("="*60)


# ============================================================================
# BƯỚC 4: CORRELATION HEATMAP
# ============================================================================

def plot_correlation_heatmap(df):
    """
    📈 VISUALIZATION 3: CORRELATION HEATMAP
    
    MỤC ĐÍCH:
    - Xem feature nào tương quan mạnh với Giá bán
    - Phát hiện multicollinearity (features tương quan với nhau)
    - Quyết định features nào nên giữ/bỏ
    
    GIẢI THÍCH:
    - Heatmap: Màu đậm = tương quan mạnh
    - Số trong ô: Pearson correlation coefficient (-1 to 1)
    - Diagonal = 1: mỗi feature tương quan hoàn toàn với chính nó
    
    ĐỌC KẾT QUẢ:
    - > 0.7: Tương quan mạnh dương (tăng cùng nhau)
    - < -0.7: Tương quan mạnh âm (nghịch biến)
    - -0.3 to 0.3: Tương quan yếu
    """
    
    # Chọn các features numeric
    numeric_features = [
        'Giá bán_numeric', 'Diện tích', 'Chiều ngang', 'Chiều dài',
        'Số phòng ngủ', 'Số phòng vệ sinh', 'Số tầng', 'Giá_per_m2'
    ]
    
    # Tạo correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap của các Features Numerical', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: outputs/3_correlation_heatmap.png")
    plt.show()
    
    # In top correlations với Giá bán
    print("\n" + "="*60)
    print("📊 TOP FEATURES TƯƠNG QUAN VỚI GIÁ BÁN")
    print("="*60)
    price_corr = corr_matrix['Giá bán_numeric'].sort_values(ascending=False)
    for feature, corr in price_corr.items():
        if feature != 'Giá bán_numeric':
            print(f"{feature:25s}: {corr:+.3f}")
    print("="*60)
    
    # Cảnh báo multicollinearity
    print("\n⚠️ CẢNH BÁO MULTICOLLINEARITY:")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        for feat1, feat2, corr in high_corr:
            print(f"  - {feat1} <-> {feat2}: {corr:.3f}")
        print("  → Cần xem xét loại bỏ 1 trong 2 features để tránh redundancy")
    else:
        print("  ✅ Không có multicollinearity nghiêm trọng")


# ============================================================================
# BƯỚC 5: BONUS - MISSING DATA VISUALIZATION
# ============================================================================

def plot_missing_data(df):
    """
    📈 VISUALIZATION BONUS: MISSING DATA PATTERN
    
    MỤC ĐÍCH:
    - Xem cột nào thiếu nhiều
    - Hiểu pattern của missing data
    """
    
    # Tính % missing
    missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_percent = missing_percent[missing_percent > 0]
    
    if len(missing_percent) == 0:
        print("✅ Không có missing data!")
        return
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_percent.values, y=missing_percent.index, palette='Reds_r')
    plt.xlabel('Tỷ lệ Missing (%)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Phân tích Missing Data', fontsize=16, weight='bold')
    
    # Thêm số % vào bars
    for i, v in enumerate(missing_percent.values):
        plt.text(v + 1, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('outputs/4_missing_data.png', dpi=300, bbox_inches='tight')
    print("✅ Đã lưu: outputs/4_missing_data.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🏠 PHÂN TÍCH DỮ LIỆU BẤT ĐỘNG SẢN VỚI SEABORN")
    print("="*60)
    
    # Load data
    filepath = 'data/gia_nha.csv'
    df = load_and_clean_data(filepath)
    
    # Visualization 1: Phân phối giá
    print("\n📊 VISUALIZATION 1: Phân phối giá bán")
    plot_price_distribution(df)
    
    # Visualization 2: Giá vs Diện tích
    print("\n📊 VISUALIZATION 2: Giá vs Diện tích theo Thành phố")
    plot_price_vs_area_by_city(df)
    
    # Visualization 3: Correlation
    print("\n📊 VISUALIZATION 3: Correlation Heatmap")
    plot_correlation_heatmap(df)
    
    # Bonus: Missing data
    print("\n📊 BONUS: Missing Data Analysis")
    plot_missing_data(df)
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH! Kiểm tra các file PNG đã tạo.")
    print("="*60)
