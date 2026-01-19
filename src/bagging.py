"""
BAGGING REGRESSOR - HOUSE PRICE PREDICTION
Tập hợp code từ notebook bagging_final.ipynb
Gồm 2 phương pháp:
- P1: Bagging code tay (custom implementation)
- P2: Bagging sklearn (wrapper)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SKLearnDecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor as SKLearnBaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# CLASS 1: DecisionTreeRegressorWrapper
# ============================================================================
class DecisionTreeRegressorWrapper:
    """Wrapper cho sklearn DecisionTreeRegressor"""
    def __init__(self, max_depth, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = SKLearnDecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)


# ============================================================================
# CLASS 2: BaggingRegressor (P1 - Code tay)
# ============================================================================
class BaggingRegressor:
    """Custom Bagging Regressor implementation with bootstrap sampling"""
    def __init__(self, n_estimators, max_depth, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = []
        np.random.seed(random_state)
    
    def fit(self, X, y):
        self.models = []
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            tree = DecisionTreeRegressorWrapper(max_depth=self.max_depth, random_state=self.random_state+i)
            tree.fit(X_bootstrap, y_bootstrap)
            self.models.append(tree)
        
        return self
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.mean(predictions, axis=1)
    
    def score_rmse(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    def score_r2(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# ============================================================================
# CLASS 3: SKLearnBaggingRegressorWrapper (P2 - Sklearn)
# ============================================================================
class SKLearnBaggingRegressorWrapper:
    """Wrapper cho sklearn BaggingRegressor để có interface giống P1"""
    def __init__(self, n_estimators, max_depth, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        base_est = SKLearnDecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        
        # Try newer parameter name first, fall back to older one
        try:
            self.model = SKLearnBaggingRegressor(
                estimator=base_est,
                n_estimators=n_estimators,
                bootstrap=True,
                random_state=random_state,
                n_jobs=-1
            )
        except TypeError:
            self.model = SKLearnBaggingRegressor(
                base_estimator=base_est,
                n_estimators=n_estimators,
                bootstrap=True,
                random_state=random_state,
                n_jobs=-1
            )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score_rmse(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y, y_pred))
    
    def score_r2(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_data(df):
    """Tiền xử lý dữ liệu: xóa NaN, encode categorical"""
    df = df.copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'Giá':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def train_and_evaluate_p1(X_train, y_train, X_test, y_test, n_estimators_range, max_depth_range):
    """Train và evaluate P1 (custom bagging)"""
    print("=" * 70)
    print("TRAINING P1 (Custom)")
    print("=" * 70)
    
    results_p1 = []
    
    for n_est in n_estimators_range:
        for max_d in max_depth_range:
            model = BaggingRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
            model.fit(X_train, y_train)
            
            rmse_test = model.score_rmse(X_test, y_test)
            r2_test = model.score_r2(X_test, y_test)
            
            results_p1.append({
                'n_estimators': n_est,
                'max_depth': max_d,
                'rmse_test': rmse_test,
                'r2_test': r2_test
            })
            
            print(f"Trees: {n_est:3d}, Depth: {max_d:2d}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")
    
    df_results_p1 = pd.DataFrame(results_p1)
    
    # Tìm model tốt nhất
    best_p1 = df_results_p1.loc[df_results_p1['rmse_test'].idxmin()]
    best_n_est_p1 = int(best_p1['n_estimators'])
    best_depth_p1 = int(best_p1['max_depth'])
    
    # Train lại model tốt nhất
    best_model_p1 = BaggingRegressor(n_estimators=best_n_est_p1, max_depth=best_depth_p1, random_state=42)
    best_model_p1.fit(X_train, y_train)
    
    best_rmse_train_p1 = best_model_p1.score_rmse(X_train, y_train)
    best_rmse_test_p1 = best_model_p1.score_rmse(X_test, y_test)
    best_r2_train_p1 = best_model_p1.score_r2(X_train, y_train)
    best_r2_test_p1 = best_model_p1.score_r2(X_test, y_test)
    
    print("\n" + "=" * 70)
    print("BEST MODEL P1")
    print("=" * 70)
    print(f"Best n_estimators: {best_n_est_p1}")
    print(f"Best max_depth: {best_depth_p1}")
    print(f"RMSE train: {best_rmse_train_p1:.4f}, R²: {best_r2_train_p1:.4f}")
    print(f"RMSE test:  {best_rmse_test_p1:.4f}, R²: {best_r2_test_p1:.4f}")
    
    return best_model_p1, df_results_p1, best_rmse_test_p1, best_r2_test_p1


def train_and_evaluate_p2(X_train, y_train, X_test, y_test, n_estimators_range, max_depth_range):
    """Train và evaluate P2 (sklearn bagging)"""
    print("\n" + "=" * 70)
    print("TRAINING P2 (SKLearn)")
    print("=" * 70)
    
    results_p2 = []
    
    for n_est in n_estimators_range:
        for max_d in max_depth_range:
            model = SKLearnBaggingRegressorWrapper(n_estimators=n_est, max_depth=max_d, random_state=42)
            model.fit(X_train, y_train)
            
            rmse_test = model.score_rmse(X_test, y_test)
            r2_test = model.score_r2(X_test, y_test)
            
            results_p2.append({
                'n_estimators': n_est,
                'max_depth': max_d,
                'rmse_test': rmse_test,
                'r2_test': r2_test
            })
            
            print(f"Trees: {n_est:3d}, Depth: {max_d:2d}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")
    
    df_results_p2 = pd.DataFrame(results_p2)
    
    # Tìm model tốt nhất
    best_p2 = df_results_p2.loc[df_results_p2['rmse_test'].idxmin()]
    best_n_est_p2 = int(best_p2['n_estimators'])
    best_depth_p2 = int(best_p2['max_depth'])
    
    # Train lại model tốt nhất
    best_model_p2 = SKLearnBaggingRegressorWrapper(n_estimators=best_n_est_p2, max_depth=best_depth_p2, random_state=42)
    best_model_p2.fit(X_train, y_train)
    
    best_rmse_train_p2 = best_model_p2.score_rmse(X_train, y_train)
    best_rmse_test_p2 = best_model_p2.score_rmse(X_test, y_test)
    best_r2_train_p2 = best_model_p2.score_r2(X_train, y_train)
    best_r2_test_p2 = best_model_p2.score_r2(X_test, y_test)
    
    print("\n" + "=" * 70)
    print("BEST MODEL P2")
    print("=" * 70)
    print(f"Best n_estimators: {best_n_est_p2}")
    print(f"Best max_depth: {best_depth_p2}")
    print(f"RMSE train: {best_rmse_train_p2:.4f}, R²: {best_r2_train_p2:.4f}")
    print(f"RMSE test:  {best_rmse_test_p2:.4f}, R²: {best_r2_test_p2:.4f}")
    
    return best_model_p2, df_results_p2, best_rmse_test_p2, best_r2_test_p2


def compare_models(best_model_p1, best_model_p2, best_rmse_p1, best_r2_p1, best_rmse_p2, best_r2_p2):
    """So sánh 2 models"""
    print("\n" + "=" * 70)
    print("SO SÁNH P1 vs P2")
    print("=" * 70)
    
    print("\nP1 (Bagging code tay):")
    print(f"  RMSE test: {best_rmse_p1:.4f}")
    print(f"  R² test:   {best_r2_p1:.4f}")
    
    print("\nP2 (Bagging sklearn):")
    print(f"  RMSE test: {best_rmse_p2:.4f}")
    print(f"  R² test:   {best_r2_p2:.4f}")
    
    print("\nKẾT LUẬN:")
    if best_rmse_p1 < best_rmse_p2:
        print(f"  P1 tốt hơn P2 với RMSE test thấp hơn {best_rmse_p2 - best_rmse_p1:.4f}")
        return best_model_p1
    else:
        print(f"  P2 tốt hơn P1 với RMSE test thấp hơn {best_rmse_p1 - best_rmse_p2:.4f}")
        return best_model_p2


def evaluate_private_test(best_model, test_csv_path, best_rmse_validation, best_r2_validation):
    """Đánh giá trên test data private"""
    print("\n" + "=" * 70)
    print("DỰ ĐOÁN TRÊN TEST DATA PRIVATE")
    print("=" * 70)
    
    # Load test data
    df_test = pd.read_csv(test_csv_path)
    print(f"\nTest data shape: {df_test.shape}")
    
    # Preprocess
    df_test_processed = preprocess_data(df_test)
    
    # Lưu giá thực để so sánh
    y_true_private = None
    if 'Giá' in df_test_processed.columns:
        print("\n⚠️  Test data có cột 'Giá' - sẽ bỏ để dự đoán và dùng để đánh giá")
        y_true_private = df_test_processed['Giá'].values
        X_test_private = df_test_processed.drop('Giá', axis=1).values
    else:
        print("\n✓ Test data không có cột 'Giá' - dùng toàn bộ làm features")
        X_test_private = df_test_processed.values
    
    # Dự đoán
    y_pred_private = best_model.predict(X_test_private)
    
    # Đánh giá nếu có giá thực
    if y_true_private is not None:
        rmse_private = np.sqrt(mean_squared_error(y_true_private, y_pred_private))
        r2_private = r2_score(y_true_private, y_pred_private)
        
        print("\n" + "=" * 70)
        print("ĐÁNH GIÁ TRÊN TEST DATA PRIVATE")
        print("=" * 70)
        print(f"RMSE: {rmse_private:.4f}")
        print(f"R²:   {r2_private:.4f}")
        
        # So sánh với kết quả trên validation set
        print("\n" + "=" * 70)
        print("SO SÁNH: VALIDATION SET vs PRIVATE TEST SET")
        print("=" * 70)
        print(f"Validation set:")
        print(f"  RMSE: {best_rmse_validation:.4f}")
        print(f"  R²:   {best_r2_validation:.4f}")
        print(f"\nPrivate test set:")
        print(f"  RMSE: {rmse_private:.4f}")
        print(f"  R²:   {r2_private:.4f}")
        
        diff_rmse = rmse_private - best_rmse_validation
        diff_r2 = r2_private - best_r2_validation
        print(f"\nChênh lệch:")
        print(f"  RMSE: {diff_rmse:+.4f} {'(tốt hơn)' if diff_rmse < 0 else '(kém hơn)'}")
        print(f"  R²:   {diff_r2:+.4f} {'(tốt hơn)' if diff_r2 > 0 else '(kém hơn)'}")
        
        # Tạo DataFrame kết quả
        df_result = pd.DataFrame({
            'Index': range(len(y_pred_private)),
            'Giá_Thực': y_true_private,
            'Giá_Dự_Đoán': y_pred_private,
            'Sai_Số': np.abs(y_true_private - y_pred_private)
        })
    else:
        df_result = pd.DataFrame({
            'Index': range(len(y_pred_private)),
            'Giá_Dự_Đoán': y_pred_private
        })
    
    # Lưu kết quả
    output_file = 'predictions_output.csv'
    df_result.to_csv(output_file, index=False)
    print(f"\n✅ Đã lưu kết quả dự đoán vào file: {output_file}")
    print(f"Số dòng dự đoán: {len(y_pred_private)}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main(train_csv_path='data/train_data.csv', test_csv_path='data/test_data.csv'):
    """
    Main function để chạy toàn bộ pipeline
    
    Parameters:
    -----------
    train_csv_path : str
        Đường dẫn đến file train data CSV
    test_csv_path : str
        Đường dẫn đến file test data CSV
    """
    print("=" * 70)
    print("BAGGING REGRESSOR - HOUSE PRICE PREDICTION")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = pd.read_csv(train_csv_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Preprocess
    print("\n2. Preprocessing data...")
    df = preprocess_data(df)
    
    X = df.drop('Giá', axis=1).values
    y = df['Giá'].values
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # 3. Split train/test
    print("\n3. Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Define grid search parameters
    n_estimators_range = list(range(10, 101, 10))
    max_depth_range = list(range(3, 11))
    
    # 5. Train P1
    print("\n4. Training P1 (Custom Bagging)...")
    best_model_p1, df_results_p1, best_rmse_p1, best_r2_p1 = train_and_evaluate_p1(
        X_train, y_train, X_test, y_test, n_estimators_range, max_depth_range
    )
    
    # 6. Train P2
    print("\n5. Training P2 (SKLearn Bagging)...")
    best_model_p2, df_results_p2, best_rmse_p2, best_r2_p2 = train_and_evaluate_p2(
        X_train, y_train, X_test, y_test, n_estimators_range, max_depth_range
    )
    
    # 7. Compare models
    print("\n6. Comparing models...")
    best_model = compare_models(best_model_p1, best_model_p2, best_rmse_p1, best_r2_p1, best_rmse_p2, best_r2_p2)
    
    # 8. Evaluate on private test
    print("\n7. Evaluating on private test data...")
    best_rmse_validation = min(best_rmse_p1, best_rmse_p2)
    best_r2_validation = best_r2_p1 if best_rmse_p1 < best_rmse_p2 else best_r2_p2
    evaluate_private_test(best_model, test_csv_path, best_rmse_validation, best_r2_validation)
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH!")
    print("=" * 70)


if __name__ == "__main__":
    # Bạn có thể truyền đường dẫn data tùy chỉnh ở đây
    # Ví dụ: main(train_csv_path='path/to/train.csv', test_csv_path='path/to/test.csv')
    main()

