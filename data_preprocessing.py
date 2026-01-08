"""
Data Preprocessing for Nhatot Housing Dataset
Tiền xử lý dữ liệu cho bộ dữ liệu bất động sản Nhatot

This script handles:
- Loading data
- Cleaning empty rows and duplicates
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Feature engineering
- Saving processed data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class HousingDataPreprocessor:
    """
    Comprehensive preprocessor for housing data
    """
    
    def __init__(self, file_path):
        """
        Initialize preprocessor
        
        Args:
            file_path: Path to CSV file
        """
        self.file_path = file_path
        self.df = None
        self.df_original = None  # Lưu trữ dữ liệu gốc
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = None
        
    def load_data(self):
        """Load data from CSV"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.file_path)
        # Tạo bản sao của dữ liệu gốc để bảo toàn
        self.df_original = self.df.copy()
        print(f"✓ Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        print(f"✓ Created backup of original data")
        print(f"\nColumns: {list(self.df.columns)}")
        return self
    
    def clean_empty_rows(self):
        """Remove completely empty rows"""
        print("\n🧹 Cleaning empty rows...")
        initial_count = len(self.df)
        # Remove rows where all columns are NaN
        self.df = self.df.dropna(how='all')
        removed = initial_count - len(self.df)
        print(f"✓ Removed {removed} empty rows")
        print(f"  Remaining: {len(self.df)} rows")
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        print("\n🔍 Removing duplicates...")
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_count - len(self.df)
        print(f"✓ Removed {removed} duplicate rows")
        return self
    
    def analyze_missing_values(self):
        """Analyze missing values in dataset"""
        print("\n📊 Missing Values Analysis:")
        print("=" * 60)
        missing_stats = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        print(missing_stats.to_string(index=False))
        print("=" * 60)
        return self
    
    def parse_price(self, price_str):
        """
        Convert Vietnamese price format to numeric value
        
        Examples:
            "1,5 tỷ" -> 1500000000
            "500 triệu" -> 500000000
            "2,35 tỷ" -> 2350000000
        """
        if pd.isna(price_str):
            return np.nan
        
        price_str = str(price_str).strip()
        
        # Remove quotes if present
        price_str = price_str.replace('"', '')
        
        # Parse value
        try:
            if 'tỷ' in price_str:
                # Billion VND
                value = price_str.replace('tỷ', '').strip()
                value = value.replace(',', '.')
                return float(value) * 1_000_000_000
            elif 'triệu' in price_str:
                # Million VND
                value = price_str.replace('triệu', '').strip()
                value = value.replace(',', '.')
                return float(value) * 1_000_000
            else:
                # Try direct conversion
                value = price_str.replace(',', '.')
                return float(value)
        except:
            return np.nan
    
    def clean_price_column(self):
        """Clean and convert price column"""
        print("\n💰 Processing price column...")
        self.df['Giá bán (VND)'] = self.df['Giá bán'].apply(self.parse_price)
        # Remove rows with invalid prices
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Giá bán (VND)'])
        removed = initial_count - len(self.df)
        print(f"✓ Converted prices to numeric")
        print(f"  Removed {removed} rows with invalid prices")
        print(f"  Price range: {self.df['Giá bán (VND)'].min():,.0f} - {self.df['Giá bán (VND)'].max():,.0f} VND")
        return self
    
    def parse_numeric_column(self, col_name):
        """Parse numeric columns that might have special values"""
        def parse_value(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip().lower()
            
            # Handle special values
            if 'nhiều hơn' in val_str or 'hơn' in val_str:
                # Extract number if present
                import re
                numbers = re.findall(r'\d+', val_str)
                if numbers:
                    return float(numbers[0]) + 1  # Add 1 to represent "more than"
                return np.nan
            
            try:
                # Try direct conversion
                return float(val_str.replace(',', '.'))
            except:
                return np.nan
        
        self.df[col_name] = self.df[col_name].apply(parse_value)
    
    def clean_numeric_columns(self):
        """Clean all numeric columns"""
        print("\n🔢 Processing numeric columns...")
        
        numeric_cols = [
            'Diện tích (m2)',
            'Chiều ngang (m)',
            'Chiều dài (m)',
            'Số phòng ngủ',
            'Số phòng vệ sinh',
            'Số tầng'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.parse_numeric_column(col)
                print(f"  ✓ Cleaned {col}")
        
        return self
    
    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values with different strategies
        
        Args:
            strategy: 'auto', 'drop', or 'impute'
        """
        print(f"\n🔧 Handling missing values (strategy: {strategy})...")
        
        if strategy == 'drop':
            # Drop rows with any missing values
            initial_count = len(self.df)
            self.df = self.df.dropna()
            removed = initial_count - len(self.df)
            print(f"  Removed {removed} rows with missing values")
        
        elif strategy == 'auto' or strategy == 'impute':
            # Impute missing values intelligently
            
            # Numeric columns: fill with median
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  ✓ Filled {col} with median: {median_val:.2f}")
            
            # Categorical columns: fill with mode or 'Unknown'
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col == 'Giá bán':  # Skip original price column
                    continue
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                        print(f"  ✓ Filled {col} with mode: {mode_val[0]}")
                    else:
                        self.df[col].fillna('Không rõ', inplace=True)
                        print(f"  ✓ Filled {col} with 'Không rõ'")
        
        print(f"✓ Missing values handled. Remaining rows: {len(self.df)}")
        return self
    
    def encode_categorical_features(self, method='label'):
        """
        Encode categorical features
        
        Args:
            method: 'label' for LabelEncoder, 'onehot' for One-Hot Encoding
        """
        print(f"\n🏷️  Encoding categorical features (method: {method})...")
        
        categorical_cols = [
            'Thành phố',
            'Phường/Xã',
            'Loại hình',
            'Giấy tờ pháp lý',
            'Hướng',
            'Tình trạng nội thất'
        ]
        
        if method == 'label':
            for col in categorical_cols:
                if col in self.df.columns:
                    le = LabelEncoder()
                    # Handle NaN by treating as a separate category
                    self.df[col] = self.df[col].fillna('Không rõ')
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                    self.label_encoders[col] = le
                    n_categories = len(le.classes_)
                    print(f"  ✓ Encoded {col} ({n_categories} categories)")
        
        elif method == 'onehot':
            # One-hot encoding
            for col in categorical_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('Không rõ')
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    print(f"  ✓ One-hot encoded {col} ({len(dummies.columns)} features)")
        
        return self
    
    def feature_engineering(self):
        """Create new features from existing ones"""
        print("\n⚙️  Feature Engineering...")
        
        # Price per square meter
        if 'Diện tích (m2)' in self.df.columns and 'Giá bán (VND)' in self.df.columns:
            self.df['Giá/m2'] = self.df['Giá bán (VND)'] / self.df['Diện tích (m2)']
            print("  ✓ Created 'Giá/m2' (price per sqm)")
        
        # Total rooms
        if 'Số phòng ngủ' in self.df.columns and 'Số phòng vệ sinh' in self.df.columns:
            self.df['Tổng số phòng'] = self.df['Số phòng ngủ'] + self.df['Số phòng vệ sinh']
            print("  ✓ Created 'Tổng số phòng' (total rooms)")
        
        # Area from dimensions
        if 'Chiều ngang (m)' in self.df.columns and 'Chiều dài (m)' in self.df.columns:
            self.df['Diện tích ước tính'] = self.df['Chiều ngang (m)'] * self.df['Chiều dài (m)']
            print("  ✓ Created 'Diện tích ước tính' (estimated area)")
        
        # Property size category
        if 'Diện tích (m2)' in self.df.columns:
            def categorize_size(area):
                if pd.isna(area):
                    return 'Không rõ'
                if area < 30:
                    return 'Rất nhỏ'
                elif area < 50:
                    return 'Nhỏ'
                elif area < 80:
                    return 'Trung bình'
                elif area < 150:
                    return 'Lớn'
                else:
                    return 'Rất lớn'
            
            self.df['Kích thước'] = self.df['Diện tích (m2)'].apply(categorize_size)
            print("  ✓ Created 'Kích thước' (size category)")
        
        return self
    
    def scale_features(self, method='standard', columns=None):
        """
        Scale numerical features
        
        Args:
            method: 'standard' or 'minmax'
            columns: List of columns to scale. If None, scale all numeric columns
        """
        print(f"\n📏 Scaling features (method: {method})...")
        
        if columns is None:
            # Select numeric columns (excluding encoded categorical and target)
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target and ID columns
            columns = [col for col in columns if 'Giá bán' not in col and '_encoded' not in col and col != 'Giá/m2']
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        # Create scaled versions
        for col in columns:
            if col in self.df.columns:
                self.df[f'{col}_scaled'] = self.scaler.fit_transform(self.df[[col]])
                print(f"  ✓ Scaled {col}")
        
        return self
    
    def get_processed_data(self):
        """Get the processed dataframe"""
        return self.df
    
    def save_processed_data(self, output_path=None, save_original=True):
        """
        Save processed data to CSV
        
        Args:
            output_path: Path for processed data file
            save_original: Whether to also save original data backup
        """
        if output_path is None:
            output_path = self.file_path.replace('.csv', '_processed.csv')
        
        # Lưu dữ liệu đã xử lý
        print(f"\n💾 Saving processed data...")
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ Processed data: {output_path}")
        print(f"  → {len(self.df)} rows and {len(self.df.columns)} columns")
        
        # Lưu bản sao dữ liệu gốc (nếu cần)
        if save_original and self.df_original is not None:
            original_backup_path = self.file_path.replace('.csv', '_original_backup.csv')
            self.df_original.to_csv(original_backup_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ Original data backup: {original_backup_path}")
            print(f"  → {len(self.df_original)} rows and {len(self.df_original.columns)} columns")
        
        print(f"\n📁 Files saved:")
        print(f"  • Original file (unchanged): {self.file_path}")
        print(f"  • Processed file (new): {output_path}")
        if save_original and self.df_original is not None:
            print(f"  • Backup file (new): {original_backup_path}")
        
        return output_path
    
    def get_summary_statistics(self):
        """Print summary statistics"""
        print("\n📈 Summary Statistics:")
        print("=" * 80)
        
        # Numeric columns summary
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print("\nNumeric Features:")
        print(self.df[numeric_cols].describe())
        
        # Categorical columns summary
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\n\nCategorical Features:")
            for col in categorical_cols[:5]:  # Show first 5
                print(f"\n{col}:")
                print(self.df[col].value_counts().head())
        
        print("=" * 80)
        return self
    
    def prepare_for_modeling(self, target_col='Giá bán (VND)', test_size=0.2, random_state=42):
        """
        Prepare data for machine learning
        
        Args:
            target_col: Target column name
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\n🎯 Preparing data for modeling...")
        
        # Select feature columns (encoded and numeric)
        feature_cols = []
        for col in self.df.columns:
            if '_encoded' in col or '_scaled' in col:
                feature_cols.append(col)
            elif col in ['Diện tích (m2)', 'Chiều ngang (m)', 'Chiều dài (m)', 
                         'Số phòng ngủ', 'Số phòng vệ sinh', 'Số tầng', 'Tổng số phòng']:
                feature_cols.append(col)
        
        X = self.df[feature_cols].fillna(0)
        y = self.df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        print(f"✓ Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test


# Main execution
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = HousingDataPreprocessor('nhatot_crawl4ai.csv')
    
    # Step 1: Load and clean data
    preprocessor.load_data()
    preprocessor.clean_empty_rows()
    preprocessor.remove_duplicates()
    
    # Step 2: Analyze missing values
    preprocessor.analyze_missing_values()
    
    # Step 3: Clean and parse columns
    preprocessor.clean_price_column()
    preprocessor.clean_numeric_columns()
    
    # Step 4: Handle missing values
    preprocessor.handle_missing_values(strategy='auto')
    
    # Step 5: Feature engineering
    preprocessor.feature_engineering()
    
    # Step 6: Encode categorical features
    preprocessor.encode_categorical_features(method='label')
    
    # Step 7: Scale numerical features
    # preprocessor.scale_features(method='standard')
    
    # Step 8: Save processed data
    output_file = preprocessor.save_processed_data()
    
    # Step 9: Show summary
    preprocessor.get_summary_statistics()
    
    # Step 10: Prepare for modeling (optional)
    print("\n" + "=" * 80)
    print("🎓 Data is ready for modeling!")
    print("=" * 80)
    
    # Example: prepare train/test split
    try:
        X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling()
        print(f"\nFeature columns: {list(X_train.columns)}")
    except Exception as e:
        print(f"\nNote: {str(e)}")
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Output file: {output_file}")
