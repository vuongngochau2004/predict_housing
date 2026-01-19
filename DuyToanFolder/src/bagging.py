"""
Bagging Regressor Implementation
Bao gồm 2 cách tiếp cận:
- P1: Custom Bagging Regressor (code tay)
- P2: sklearn BaggingRegressor wrapper
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor as SKLearnDecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor as SKLearnBaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DecisionTreeRegressorWrapper:
    """Wrapper cho sklearn DecisionTreeRegressor để sử dụng trong custom Bagging"""
    
    def __init__(self, max_depth, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = SKLearnDecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class BaggingRegressor:
    """
    Custom Bagging Regressor implementation (P1)
    Bootstrap Aggregating với Decision Tree base estimators
    """
    
    def __init__(self, n_estimators, max_depth, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = []
        np.random.seed(random_state)
    
    def fit(self, X, y):
        """Train ensemble bằng bootstrap sampling"""
        self.models = []
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train base estimator
            tree = DecisionTreeRegressorWrapper(max_depth=self.max_depth, random_state=self.random_state+i)
            tree.fit(X_bootstrap, y_bootstrap)
            self.models.append(tree)
        
        return self
    
    def predict(self, X):
        """Dự đoán bằng trung bình các predictions từ tất cả base estimators"""
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.mean(predictions, axis=1)
    
    def score_rmse(self, X, y):
        """Tính RMSE"""
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    def score_r2(self, X, y):
        """Tính R² score"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class SKLearnBaggingRegressorWrapper:
    """
    Wrapper cho sklearn BaggingRegressor (P2)
    Để dễ so sánh với custom implementation
    """
    
    def __init__(self, n_estimators, max_depth, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        base_estimator = SKLearnDecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        
        # Try new parameter name first, fallback to old name
        try:
            self.model = SKLearnBaggingRegressor(
                estimator=base_estimator,
                n_estimators=n_estimators,
                bootstrap=True,
                random_state=random_state,
                n_jobs=n_jobs
            )
        except TypeError:
            self.model = SKLearnBaggingRegressor(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                bootstrap=True,
                random_state=random_state,
                n_jobs=n_jobs
            )
    
    def fit(self, X, y):
        """Train model"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict"""
        return self.model.predict(X)
    
    def score_rmse(self, X, y):
        """Tính RMSE"""
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    def score_r2(self, X, y):
        """Tính R² score"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def preprocess_data(df, target_col='Giá'):
    """
    Preprocess dữ liệu: xử lý missing values và encode categorical columns
    
    Args:
        df: DataFrame
        target_col: Tên cột target (mặc định 'Giá')
    
    Returns:
        X, y: Features và target đã được xử lý
    """
    # Copy để không thay đổi original
    df_processed = df.copy()
    
    # Xử lý missing values
    df_processed.dropna(inplace=True)
    df_processed.reset_index(drop=True, inplace=True)
    
    # Encode categorical columns
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' and col != target_col:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Tách features và target
    X = df_processed.drop(target_col, axis=1).values
    y = df_processed[target_col].values
    
    return X, y


def train_and_evaluate(X_train, y_train, X_test, y_test, 
                       n_estimators_range=None, max_depth_range=None,
                       use_custom=True):
    """
    Train và đánh giá Bagging models với grid search
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_estimators_range: List số lượng estimators để thử
        max_depth_range: List độ sâu cây để thử
        use_custom: True = dùng custom implementation, False = dùng sklearn
    
    Returns:
        best_model, results_df, best_config
    """
    if n_estimators_range is None:
        n_estimators_range = list(range(10, 101, 10))
    if max_depth_range is None:
        max_depth_range = list(range(3, 11))
    
    model_name = "P1 (Custom)" if use_custom else "P2 (Sklearn)"
    print("=" * 70)
    print(f"TRAINING {model_name}")
    print("=" * 70)
    
    results = []
    
    for n_est in n_estimators_range:
        for max_d in max_depth_range:
            # Tạo model
            if use_custom:
                model = BaggingRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
            else:
                model = SKLearnBaggingRegressorWrapper(n_estimators=n_est, max_depth=max_d, random_state=42)
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            rmse_test = model.score_rmse(X_test, y_test)
            r2_test = model.score_r2(X_test, y_test)
            
            results.append({
                'n_estimators': n_est,
                'max_depth': max_d,
                'rmse_test': rmse_test,
                'r2_test': r2_test
            })
            
            print(f"Trees: {n_est:3d}, Depth: {max_d:2d}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")
    
    # Tạo DataFrame kết quả
    df_results = pd.DataFrame(results)
    
    # Tìm best model
    best_result = df_results.loc[df_results['rmse_test'].idxmin()]
    best_n_est = int(best_result['n_estimators'])
    best_depth = int(best_result['max_depth'])
    
    # Train lại best model
    if use_custom:
        best_model = BaggingRegressor(n_estimators=best_n_est, max_depth=best_depth, random_state=42)
    else:
        best_model = SKLearnBaggingRegressorWrapper(n_estimators=best_n_est, max_depth=best_depth, random_state=42)
    
    best_model.fit(X_train, y_train)
    
    # Đánh giá best model
    rmse_train = best_model.score_rmse(X_train, y_train)
    rmse_test = best_model.score_rmse(X_test, y_test)
    r2_train = best_model.score_r2(X_train, y_train)
    r2_test = best_model.score_r2(X_test, y_test)
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL {model_name}")
    print("=" * 70)
    print(f"Best n_estimators: {best_n_est}")
    print(f"Best max_depth: {best_depth}")
    print(f"RMSE train: {rmse_train:.4f}, R²: {r2_train:.4f}")
    print(f"RMSE test:  {rmse_test:.4f}, R²: {r2_test:.4f}")
    
    best_config = {
        'n_estimators': best_n_est,
        'max_depth': best_depth,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'r2_train': r2_train,
        'r2_test': r2_test
    }
    
    return best_model, df_results, best_config


def main():
    """
    Main function để chạy toàn bộ pipeline:
    1. Load data từ data/test_data.csv
    2. Preprocess
    3. Train cả P1 (custom) và P2 (sklearn)
    4. So sánh kết quả
    5. Đánh giá trên test set
    """
    print("\n" + "=" * 70)
    print("BAGGING REGRESSOR - HOUSE PRICE PREDICTION")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading data...")
    data_path = 'data/test_data.csv'
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Preprocess
    print("\n2. Preprocessing data...")
    X, y = preprocess_data(df, target_col='Giá')
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # 3. Split train/test (80/20)
    print("\n3. Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Train P1 (Custom Bagging)
    print("\n4. Training P1 (Custom Bagging)...")
    model_p1, results_p1, config_p1 = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        n_estimators_range=list(range(10, 101, 10)),
        max_depth_range=list(range(3, 11)),
        use_custom=True
    )
    
    # 5. Train P2 (Sklearn Bagging)
    print("\n\n5. Training P2 (Sklearn Bagging)...")
    model_p2, results_p2, config_p2 = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        n_estimators_range=list(range(10, 101, 10)),
        max_depth_range=list(range(3, 11)),
        use_custom=False
    )
    
    # 6. So sánh P1 vs P2
    print("\n\n" + "=" * 70)
    print("SO SÁNH P1 vs P2")
    print("=" * 70)
    
    print("\nP1 (Custom Bagging):")
    print(f"  Config: n_estimators={config_p1['n_estimators']}, max_depth={config_p1['max_depth']}")
    print(f"  RMSE test: {config_p1['rmse_test']:.4f}")
    print(f"  R² test:   {config_p1['r2_test']:.4f}")
    
    print("\nP2 (Sklearn Bagging):")
    print(f"  Config: n_estimators={config_p2['n_estimators']}, max_depth={config_p2['max_depth']}")
    print(f"  RMSE test: {config_p2['rmse_test']:.4f}")
    print(f"  R² test:   {config_p2['r2_test']:.4f}")
    
    # Chọn model tốt hơn
    if config_p1['rmse_test'] < config_p2['rmse_test']:
        print(f"\n✅ P1 tốt hơn P2 với RMSE test thấp hơn {config_p2['rmse_test'] - config_p1['rmse_test']:.4f}")
        best_model = model_p1
        best_config = config_p1
        best_name = "P1 (Custom)"
    else:
        print(f"\n✅ P2 tốt hơn P1 với RMSE test thấp hơn {config_p1['rmse_test'] - config_p2['rmse_test']:.4f}")
        best_model = model_p2
        best_config = config_p2
        best_name = "P2 (Sklearn)"
    
    # 7. Đánh giá full test set
    print("\n\n" + "=" * 70)
    print("ĐÁNH GIÁ TRÊN FULL TEST SET")
    print("=" * 70)
    print(f"Sử dụng model: {best_name}")
    
    y_pred_test = best_model.predict(X_test)
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_final = r2_score(y_test, y_pred_test)
    
    print(f"\nKết quả cuối cùng:")
    print(f"  RMSE: {rmse_final:.4f}")
    print(f"  R²:   {r2_final:.4f}")
    
    # Lưu predictions
    df_predictions = pd.DataFrame({
        'Giá_Thực': y_test,
        'Giá_Dự_Đoán': y_pred_test,
        'Sai_Số': np.abs(y_test - y_pred_test)
    })
    
    output_file = 'predictions_output.csv'
    df_predictions.to_csv(output_file, index=False)
    print(f"\n✅ Đã lưu predictions vào: {output_file}")
    
    # Hiển thị mẫu
    print(f"\nMẫu 10 dự đoán đầu tiên:")
    print(df_predictions.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("HOÀN TẤT!")
    print("=" * 70)
    
    return best_model, best_config, df_predictions


if __name__ == "__main__":
    best_model, best_config, predictions = main()
