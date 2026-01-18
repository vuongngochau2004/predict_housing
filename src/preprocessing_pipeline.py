"""
Complete Preprocessing Pipeline for Real Estate Price Prediction
Implements all strategies from preprocessing_guide.md
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import config

# ============================================================================
# STEP 1: DATA LOADING & BASIC CLEANING
# ============================================================================

def parse_price(price_str):
    """
    Parse Vietnamese price text to numeric VND
    
    Examples:
        "3,5 tỷ" -> 3_500_000_000
        "850 triệu" -> 850_000_000
    """
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
    """
    Clean numeric columns containing string values
    
    Examples:
        "nhiều hơn 10" -> 10.0
        "5" -> 5.0
    """
    def convert_value(val):
        if pd.isna(val):
            return np.nan
        
        val_str = str(val).strip().lower()
        
        # Handle "nhiều hơn X" pattern
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
    """
    Load raw data and perform basic cleaning
    """
    print("="*60)
    print("STEP 1: LOADING & BASIC CLEANING")
    print("="*60)
    
    print(f"📁 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    initial_count = len(df)
    print(f"   Initial rows: {initial_count:,}")
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    print(f"   After removing empty rows: {len(df):,}")
    
    # Parse price
    print("💰 Parsing price column...")
    df['Giá bán_numeric'] = df['Giá bán'].apply(parse_price)
    
    # Drop critical missing
    print(f"🚫 Dropping rows with missing critical columns...")
    df = df.dropna(subset=config.CRITICAL_COLUMNS)
    print(f"   After dropping critical missing: {len(df):,}")
    
    # Clean numeric columns
    print("🧹 Cleaning numeric columns...")
    numeric_cols = ['Số phòng ngủ', 'Số phòng vệ sinh', 'Số tầng', 
                   'Chiều ngang (m)', 'Chiều dài (m)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    print(f"✅ Basic cleaning complete: {len(df):,} rows remaining")
    print(f"   Removed: {initial_count - len(df):,} rows ({(initial_count - len(df))/initial_count*100:.1f}%)\n")
    
    return df


# ============================================================================
# STEP 2: OUTLIER DETECTION & REMOVAL
# ============================================================================

def remove_domain_outliers(df):
    """
    Remove outliers based on domain knowledge
    """
    print("="*60)
    print("STEP 2: OUTLIER DETECTION & REMOVAL")
    print("="*60)
    
    initial_count = len(df)
    
    print("🔍 Removing domain knowledge outliers...")
    for col, (lower, upper) in config.OUTLIER_BOUNDS.items():
        if col in df.columns:
            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            removed = before - len(df)
            if removed > 0:
                print(f"   {col}: removed {removed} outliers")
    
    print(f"✅ Domain outliers removed: {initial_count - len(df):,} rows\n")
    return df


def remove_iqr_outliers(df):
    """
    Remove statistical outliers using IQR method
    """
    print(f"🔍 Removing IQR outliers (multiplier={config.IQR_MULTIPLIER})...")
    
    initial_count = len(df)
    
    for col in config.IQR_OUTLIER_COLS:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - config.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + config.IQR_MULTIPLIER * IQR
            
            before = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            removed = before - len(df)
            if removed > 0:
                print(f"   {col}: removed {removed} outliers")
    
    print(f"✅ IQR outliers removed: {initial_count - len(df):,} rows")
    print(f"   Total remaining: {len(df):,} rows\n")
    
    return df


# ============================================================================
# STEP 3: MISSING VALUE IMPUTATION
# ============================================================================

def handle_missing_values(df):
    """
    Handle missing values using various strategies
    """
    print("="*60)
    print("STEP 3: MISSING VALUE IMPUTATION")
    print("="*60)
    
    # Fill with specific values
    print("📝 Filling categorical missing with 'Không xác định'...")
    for col, value in config.FILL_WITH_VALUE.items():
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(value)
                print(f"   {col}: filled {missing_count} missing")
    
    # Fill with group median
    print("\n📊 Filling with group median (grouped by 'Loại hình')...")
    for col in config.FILL_WITH_GROUP_MEDIAN:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df.groupby('Loại hình')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill any remaining with global median
                df[col] = df[col].fillna(df[col].median())
                print(f"   {col}: filled {missing_count} missing")
    
    # Fill with global median
    print("\n📊 Filling with global median...")
    for col in config.FILL_WITH_MEDIAN:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].fillna(df[col].median())
                print(f"   {col}: filled {missing_count} missing")
    
    # Report remaining missing
    print("\n📋 Remaining missing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"   {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("   ✅ No missing values!")
    
    print()
    return df


# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    """
    Create new engineered features
    """
    print("="*60)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*60)
    
    print("🔧 Creating new features...")
    
    for feature_name, feature_func in config.FEATURE_ENGINEERING_CONFIG.items():
        try:
            df[feature_name] = feature_func(df)
            print(f"   ✅ Created: {feature_name}")
        except Exception as e:
            print(f"   ❌ Failed to create {feature_name}: {e}")
    
    print()
    return df


# ============================================================================
# STEP 5: ENCODING
# ============================================================================

def encode_features(df, is_training=True, encoders=None):
    """
    Encode categorical features
    
    Args:
        df: DataFrame to encode
        is_training: If True, fit encoders. If False, use provided encoders
        encoders: Dict of fitted encoders (for test set)
    
    Returns:
        df: Encoded DataFrame
        encoders: Dict of fitted encoders (if is_training=True)
    """
    print("="*60)
    print("STEP 5: ENCODING CATEGORICAL FEATURES")
    print("="*60)
    
    if encoders is None:
        encoders = {}
    
    # One-Hot Encoding for low cardinality
    print("🔤 One-Hot Encoding...")
    for col in config.ONEHOT_ENCODING_COLS:
        if col in df.columns:
            if is_training:
                # Get all possible values
                encoders[col] = df[col].unique()
            
            # Create dummy variables
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            print(f"   {col}: created {len(dummies.columns)} dummy columns")
    
    # Target Encoding for high cardinality (simplified - using mean)
    print("\n🎯 Target Encoding (using mean price)...")
    for col in config.TARGET_ENCODING_COLS:
        if col in df.columns:
            if is_training:
                # Calculate mean price per category
                encoding_map = df.groupby(col)['Giá bán_numeric'].mean().to_dict()
                encoders[f'{col}_target_map'] = encoding_map
            else:
                encoding_map = encoders[f'{col}_target_map']
            
            # Apply encoding
            df[f'{col}_encoded'] = df[col].map(encoding_map)
            
            # Fill unknown categories with global mean
            if df[f'{col}_encoded'].isna().any():
                global_mean = df['Giá bán_numeric'].mean()
                df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(global_mean)
            
            print(f"   {col}: encoded to {col}_encoded")
    
    print()
    
    if is_training:
        return df, encoders
    else:
        return df


# ============================================================================
# STEP 6: SCALING & TRANSFORMATION
# ============================================================================

def apply_transformations(df):
    """
    Apply log transformation and scaling
    """
    print("="*60)
    print("STEP 6: SCALING & TRANSFORMATION")
    print("="*60)
    
    # Log transformation
    print("📊 Applying log transformation...")
    for col in config.LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
            print(f"   {col} -> {col}_log")
    
    print()
    return df


# ============================================================================
# STEP 7: FINAL FEATURE SELECTION
# ============================================================================

def select_features(df):
    """
    Select final features for modeling
    """
    print("="*60)
    print("STEP 7: FEATURE SELECTION")
    print("="*60)
    
    # Drop original categorical columns (we have encoded versions)
    cols_to_drop = config.FEATURES_TO_EXCLUDE.copy()
    cols_to_drop.extend(config.ONEHOT_ENCODING_COLS)
    cols_to_drop.extend(config.TARGET_ENCODING_COLS)
    
    # Remove columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    print(f"🗑️  Dropping {len(cols_to_drop)} columns:")
    for col in cols_to_drop[:10]:  # Show first 10
        print(f"   - {col}")
    if len(cols_to_drop) > 10:
        print(f"   ... and {len(cols_to_drop) - 10} more")
    
    df = df.drop(columns=cols_to_drop)
    
    print(f"\n✅ Final dataset shape: {df.shape}")
    print(f"   Features: {df.shape[1] - 1} (excluding target)")
    print(f"   Samples: {df.shape[0]:,}\n")
    
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_preprocessing_pipeline(
    input_file=config.RAW_DATA_PATH,
    output_file=config.PROCESSED_DATA_PATH,
    create_splits=True
):
    """
    Run complete preprocessing pipeline
    
    Args:
        input_file: Path to raw CSV
        output_file: Path to save processed CSV
        create_splits: If True, create train/test splits
    
    Returns:
        df: Processed DataFrame
        encoders: Dict of encoders (for later use on new data)
    """
    print("\n" + "="*60)
    print("🏠 REAL ESTATE PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Load & Clean
    df = load_and_clean_data(input_file)
    
    # Step 2: Remove Outliers
    df = remove_domain_outliers(df)
    df = remove_iqr_outliers(df)
    
    # Step 3: Handle Missing Values
    df = handle_missing_values(df)
    
    # Step 4: Feature Engineering
    df = create_features(df)
    
    # Step 5: Encoding
    df, encoders = encode_features(df, is_training=True)
    
    # Step 6: Transformations
    df = apply_transformations(df)
    
    # Step 7: Feature Selection
    df = select_features(df)
    
    # Save processed data
    print("="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    print(f"💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ Saved: {len(df):,} rows, {len(df.columns)} columns\n")
    
    # Create train/test splits
    if create_splits:
        print("="*60)
        print("CREATING TRAIN/TEST SPLITS")
        print("="*60)
        
        # Ensure target exists
        if config.TARGET in df.columns:
            X = df.drop(columns=[config.TARGET])
            y = df[config.TARGET]
        else:
            print(f"⚠️  Warning: Target '{config.TARGET}' not found. Using all columns.")
            X = df
            y = None
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                shuffle=config.SHUFFLE
            )
            
            # Combine X and y for saving
            df_train = pd.concat([X_train, y_train], axis=1)
            df_test = pd.concat([X_test, y_test], axis=1)
        else:
            df_train, df_test = train_test_split(
                df,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                shuffle=config.SHUFFLE
            )
        
        print(f"📊 Train set: {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%)")
        print(f"📊 Test set:  {len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%)")
        
        # Save splits
        df_train.to_csv(config.TRAIN_DATA_PATH, index=False, encoding='utf-8-sig')
        df_test.to_csv(config.TEST_DATA_PATH, index=False, encoding='utf-8-sig')
        print(f"💾 Saved train: {config.TRAIN_DATA_PATH}")
        print(f"💾 Saved test:  {config.TEST_DATA_PATH}\n")
    
    print("="*60)
    print("✅ PREPROCESSING PIPELINE COMPLETE!")
    print("="*60)
    print(f"📈 Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"🎯 Target variable: {config.TARGET}")
    print(f"📁 Files created:")
    print(f"   - {output_file}")
    if create_splits:
        print(f"   - {config.TRAIN_DATA_PATH}")
        print(f"   - {config.TEST_DATA_PATH}")
    print("="*60 + "\n")
    
    return df, encoders


# ============================================================================
# RUN PIPELINE
# ============================================================================

if __name__ == "__main__":
    df_processed, encoders = run_preprocessing_pipeline()
    
    # Show sample
    print("📋 Sample of processed data:")
    print(df_processed.head())
    
    print("\n📊 Feature dtypes:")
    print(df_processed.dtypes.value_counts())
    
    print("\n📈 Target variable stats:")
    if config.TARGET in df_processed.columns:
        print(df_processed[config.TARGET].describe())
