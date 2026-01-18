"""
Configuration file for Real Estate Price Prediction Pipeline
"""

import numpy as np

# ============================================================================
# FILE PATHS
# ============================================================================

RAW_DATA_PATH = 'data/gia_nha.csv'
PROCESSED_DATA_PATH = 'data/gia_nha_processed_ml_ready.csv'
TRAIN_DATA_PATH = 'data/gia_nha_train.csv'
TEST_DATA_PATH = 'data/gia_nha_test.csv'

# ============================================================================
# DATA CLEANING
# ============================================================================

# Columns to drop completely (too many missing or not useful)
COLUMNS_TO_DROP = []

# Critical columns - drop row if missing
CRITICAL_COLUMNS = ['Giá bán', 'Diện tích (m2)']

# ============================================================================
# MISSING VALUE STRATEGIES
# ============================================================================

# Columns to fill with specific value
FILL_WITH_VALUE = {
    'Hướng': 'Không xác định',
    'Tình trạng nội thất': 'Không xác định'
}

# Columns to fill with group median (group by 'Loại hình')
FILL_WITH_GROUP_MEDIAN = [
    'Chiều ngang (m)',
    'Chiều dài (m)',
    'Số tầng'
]

# Columns to fill with global median
FILL_WITH_MEDIAN = [
    'Số phòng ngủ',
    'Số phòng vệ sinh'
]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# New features to create
FEATURE_ENGINEERING_CONFIG = {
    'Giá_per_m2': lambda df: df['Giá bán_numeric'] / df['Diện tích (m2)'],
    'Tổng_phòng': lambda df: df['Số phòng ngủ'].fillna(0) + df['Số phòng vệ sinh'].fillna(0),
    'Aspect_ratio': lambda df: df['Chiều ngang (m)'] / df['Chiều dài (m)'].replace(0, np.nan),
    'Diện_tích_per_phòng': lambda df: df['Diện tích (m2)'] / df['Tổng_phòng'].replace(0, np.nan),
}

# ============================================================================
# ENCODING
# ============================================================================

# High cardinality columns - use Target Encoding
TARGET_ENCODING_COLS = ['Phường/Xã', 'Thành phố']

# Low cardinality columns - use One-Hot Encoding
ONEHOT_ENCODING_COLS = ['Loại hình', 'Giấy tờ pháp lý', 'Hướng', 'Tình trạng nội thất']

# ============================================================================
# SCALING / TRANSFORMATION
# ============================================================================

# Columns to apply log transformation
LOG_TRANSFORM_COLS = ['Giá bán_numeric', 'Diện tích (m2)', 'Giá_per_m2']

# Columns to apply standard scaling (after encoding)
STANDARD_SCALE_COLS = []  # Will be filled after encoding

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

# Domain knowledge outlier bounds
OUTLIER_BOUNDS = {
    'Giá bán_numeric': (100_000_000, 500_000_000_000),  # 100M - 500B VND
    'Diện tích (m2)': (5, 10000),  # 5m² - 1 hectare
    'Giá_per_m2': (1_000_000, 1_000_000_000),  # 1M - 1B VND/m²
}

# IQR multiplier for statistical outlier detection
IQR_MULTIPLIER = 3.0  # Conservative (1.5 is standard, 3.0 keeps more data)

# Columns to apply IQR outlier detection
IQR_OUTLIER_COLS = ['Giá bán_numeric', 'Diện tích (m2)', 'Giá_per_m2']

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42
SHUFFLE = True

# ============================================================================
# TARGET VARIABLE
# ============================================================================

TARGET = 'Giá bán_numeric_log'  # Use log-transformed price as target

# ============================================================================
# FEATURES TO USE IN MODEL
# ============================================================================

# Will be determined after preprocessing, but exclude these:
FEATURES_TO_EXCLUDE = [
    'Giá bán',  # Original text
    'Giá bán_numeric',  # Original price (use log version)
    'Phường/Xã',  # Encoded version will be used
    'Thành phố',  # Encoded version will be used
]

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

CV_FOLDS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 42
