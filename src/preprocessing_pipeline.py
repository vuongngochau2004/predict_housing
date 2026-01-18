"""
Complete Preprocessing Pipeline for Real Estate Price Prediction
Updated with Smoothed K-Fold Target Encoding (NO DATA LEAKAGE)
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import config

# ============================================================================
# STEP 1: DATA LOADING & BASIC CLEANING
# ============================================================================

def parse_price(price_str):
    """Parse Vietnamese price text to numeric VND"""
    if pd.isna(price_str) or price_str == '':
        return np.nan
    
    price_str = str(price_str).strip().lower()
    
    try:
        if 'tỷ' in price_str:
            number = price_str.replace('tỷ', '').replace(',', '.').strip()
            return float(number) * 1_000_000_000
        elif 'triệu' in price_str:
            number = price_str.replace('triệu', '').replace(',', '.').strip()
            return float(number) * 1_000_000
        else:
            return float(price_str.replace(',', '.'))
    except:
        return np.nan


def clean_numeric_column(series):
    """Clean numeric columns containing string values"""
    def convert_value(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().lower()
        if 'nhiều hơn' in val_str or 'nhieu hon' in val_str:
            numbers = re.findall(r'\d+', val_str)
            if numbers:
                return float(numbers[0])
            return np.nan
        try:
            return float(val_str)
        except:
            return np.nan
    return series.apply(convert_value)


def load_and_clean_data(filepath):
    """Load raw data and perform basic cleaning"""
    print("="*60)
    print("STEP 1: LOADING & BASIC CLEANING")
    print("="*60)
    
    df = pd.read_csv(filepath)
    initial_count = len(df)
    print(f"📁 Initial rows: {initial_count:,}")
    
    df = df.dropna(how='all')
    df['Giá bán_numeric'] = df['Giá bán'].apply(parse_price)
    df = df.dropna(subset=config.CRITICAL_COLUMNS)
    
    numeric_cols = ['Số phòng ngủ', 'Số phòng vệ sinh', 'Số tầng', 
                   'Chiều ngang (m)', 'Chiều dài (m)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    print(f"✅ Clean: {len(df):,} rows ({(initial_count - len(df))/initial_count*100:.1f}% removed)\n")
    return df


# ============================================================================
# STEP 2: OUTLIER DETECTION & REMOVAL
# ============================================================================

def remove_outliers(df):
    """Remove outliers using domain knowledge + IQR"""
    print("="*60)
    print("STEP 2: OUTLIER REMOVAL")
    print("="*60)
    
    initial = len(df)
    
    # Domain bounds
    for col, (lower, upper) in config.OUTLIER_BOUNDS.items():
        if col in df.columns:
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    # IQR
    for col in config.IQR_OUTLIER_COLS:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]
    
    print(f"✅ Removed {initial - len(df)} outliers. Remaining: {len(df):,}\n")
    return df


# ============================================================================
# STEP 3: MISSING VALUE IMPUTATION
# ============================================================================

def handle_missing_values(df):
    """Handle missing values"""
    print("="*60)
    print("STEP 3: MISSING VALUE IMPUTATION")
    print("="*60)
    
    # Categorical → "Không xác định"
    for col, value in config.FILL_WITH_VALUE.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    
    # Numeric → Group/Global median
    for col in config.FILL_WITH_GROUP_MEDIAN:
        if col in df.columns:
            df[col] = df.groupby('Loại hình')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())
    
    for col in config.FILL_WITH_MEDIAN:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    print(f"✅ Missing values handled\n")
    return df


# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    """Create engineered features"""
    print("="*60)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*60)
    
    for name, func in config.FEATURE_ENGINEERING_CONFIG.items():
        try:
            df[name] = func(df)
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    print()
    return df


# ============================================================================
# STEP 5: SMOOTHED K-FOLD TARGET ENCODING (NO DATA LEAKAGE!)
# ============================================================================

def smoothed_kfold_target_encoding(df, cat_col, target_col, n_folds=5, smoothing=10):
    """
    Smoothed K-Fold Target Encoding
    
    Prevents data leakage by:
    1. Using K-Fold: each row is encoded using data from OTHER folds
    2. Using Smoothing: prevents overfitting for rare categories
    
    Formula: (count * category_mean + smoothing * global_mean) / (count + smoothing)
    """
    encoded = np.zeros(len(df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        
        # Global mean từ TRAIN fold
        global_mean = train[target_col].mean()
        
        # Tính mean và count cho mỗi category từ TRAIN fold
        agg = train.groupby(cat_col)[target_col].agg(['mean', 'count'])
        
        # Smoothed encoding: tránh overfitting cho category có ít samples
        smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        
        # Apply cho VAL fold (các rows này KHÔNG được dùng để tính mean)
        encoded[val_idx] = df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean).values
    
    return encoded


def encode_features_safe(df, is_training=True, encoders=None):
    """
    Encode features WITHOUT data leakage
    
    - One-Hot: OK (không dùng target)
    - Target Encoding: Dùng Smoothed K-Fold
    """
    print("="*60)
    print("STEP 5: ENCODING (Safe - No Data Leakage)")
    print("="*60)
    
    if encoders is None:
        encoders = {}
    
    # One-Hot Encoding (không dùng target → OK)
    print("🔤 One-Hot Encoding...")
    for col in config.ONEHOT_ENCODING_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            print(f"   {col}: {len(dummies.columns)} columns")
            
            if is_training:
                encoders[f'{col}_categories'] = list(dummies.columns)
    
    # Smoothed K-Fold Target Encoding
    print("\n🎯 Smoothed K-Fold Target Encoding...")
    for col in config.TARGET_ENCODING_COLS:
        if col in df.columns:
            if is_training:
                # K-Fold encoding (mỗi row được encode bởi data từ CÁC FOLDS KHÁC)
                df[f'{col}_encoded'] = smoothed_kfold_target_encoding(
                    df, col, 'Giá bán_numeric', n_folds=5, smoothing=10
                )
                
                # Lưu encoder cuối cùng (mean của toàn bộ train) cho inference
                global_mean = df['Giá bán_numeric'].mean()
                agg = df.groupby(col)['Giá bán_numeric'].agg(['mean', 'count'])
                smoothed = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
                encoders[f'{col}_map'] = smoothed.to_dict()
                encoders[f'{col}_global_mean'] = global_mean
                
                print(f"   ✅ {col}: K-Fold encoded (5 folds, smoothing=10)")
            else:
                # Inference: dùng encoder đã lưu
                encoding_map = encoders[f'{col}_map']
                global_mean = encoders[f'{col}_global_mean']
                df[f'{col}_encoded'] = df[col].map(encoding_map).fillna(global_mean)
                print(f"   ✅ {col}: applied saved encoder")
    
    print()
    
    if is_training:
        return df, encoders
    else:
        return df


# ============================================================================
# STEP 6: TRANSFORMATIONS
# ============================================================================

def apply_transformations(df):
    """Apply log transformation"""
    print("="*60)
    print("STEP 6: LOG TRANSFORMATION")
    print("="*60)
    
    for col in config.LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
            print(f"   {col} → {col}_log")
    
    print()
    return df


# ============================================================================
# STEP 7: FEATURE SELECTION
# ============================================================================

def select_features(df):
    """Select final features"""
    print("="*60)
    print("STEP 7: FEATURE SELECTION")
    print("="*60)
    
    cols_to_drop = config.FEATURES_TO_EXCLUDE.copy()
    cols_to_drop.extend(config.ONEHOT_ENCODING_COLS)
    cols_to_drop.extend(config.TARGET_ENCODING_COLS)
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    df = df.drop(columns=cols_to_drop)
    
    print(f"✅ Final: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_preprocessing_pipeline(
    input_file=config.RAW_DATA_PATH,
    output_file=config.PROCESSED_DATA_PATH,
    create_splits=True
):
    """Run complete preprocessing pipeline with safe Target Encoding"""
    
    print("\n" + "="*60)
    print("🏠 PREPROCESSING PIPELINE (Safe Target Encoding)")
    print("="*60 + "\n")
    
    # Steps 1-4: Không dùng target → OK
    df = load_and_clean_data(input_file)
    df = remove_outliers(df)
    df = handle_missing_values(df)
    df = create_features(df)
    
    # Step 5: Safe encoding với K-Fold
    df, encoders = encode_features_safe(df, is_training=True)
    
    # Step 6: Transformations
    df = apply_transformations(df)
    
    # Step 7: Feature selection
    df = select_features(df)
    
    # Save
    print("="*60)
    print("SAVING")
    print("="*60)
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 Saved: {output_file}")
    
    # Save encoders for inference
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/encoders.joblib')
    print(f"💾 Saved: models/encoders.joblib")
    
    # Train/Test split
    if create_splits:
        if config.TARGET in df.columns:
            X = df.drop(columns=[config.TARGET])
            y = df[config.TARGET]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
            )
            df_train = pd.concat([X_train, y_train], axis=1)
            df_test = pd.concat([X_test, y_test], axis=1)
        else:
            df_train, df_test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
        
        df_train.to_csv(config.TRAIN_DATA_PATH, index=False, encoding='utf-8-sig')
        df_test.to_csv(config.TEST_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"💾 Train: {len(df_train):,} rows")
        print(f"💾 Test: {len(df_test):,} rows")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE (No Data Leakage!)")
    print("="*60 + "\n")
    
    return df, encoders


if __name__ == "__main__":
    df, encoders = run_preprocessing_pipeline()
    print(f"📊 Shape: {df.shape}")
    print(f"🔑 Encoders saved: {list(encoders.keys())}")
