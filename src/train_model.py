"""
Train and save ML model for house price prediction
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def load_data():
    """Load train and test data"""
    print("📁 Loading data...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    
    print(f"   Train: {len(df_train):,} rows")
    print(f"   Test: {len(df_test):,} rows")
    
    return df_train, df_test


def prepare_features(df_train, df_test):
    """Prepare features and target"""
    print("\n🔧 Preparing features...")
    
    # Target
    target = config.TARGET
    
    # Features (all columns except target)
    feature_cols = [col for col in df_train.columns if col != target]
    
    X_train = df_train[feature_cols]
    y_train = df_train[target]
    
    X_test = df_test[feature_cols]
    y_test = df_test[target]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target: {target}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def train_model(X_train, y_train):
    """Train RandomForest model"""
    print("\n🏋️ Training RandomForest model...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✅ Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n📊 Evaluating model...")
    
    # Predict (log scale)
    y_pred_log = model.predict(X_test)
    
    # Convert back to original scale
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    
    # Metrics in original scale (VND)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\n" + "="*50)
    print("📈 MODEL PERFORMANCE (Original Scale)")
    print("="*50)
    print(f"MAE:  {mae/1e9:.2f} tỷ VND")
    print(f"RMSE: {rmse/1e9:.2f} tỷ VND")
    print(f"R²:   {r2:.3f}")
    print(f"MAPE: {mape:.1f}%")
    print("="*50)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def save_model(model, feature_cols, metrics):
    """Save model and metadata"""
    print("\n💾 Saving model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/model.joblib')
    print("   ✅ Saved: models/model.joblib")
    
    # Save feature names
    with open('models/feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print("   ✅ Saved: models/feature_names.json")
    
    # Save metrics
    with open('models/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'mae_billion': metrics['mae'] / 1e9,
            'rmse_billion': metrics['rmse'] / 1e9,
            'r2': metrics['r2'],
            'mape': metrics['mape']
        }, f, indent=2)
    print("   ✅ Saved: models/metrics.json")


def main():
    print("="*50)
    print("🏠 TRAINING HOUSE PRICE PREDICTION MODEL")
    print("="*50)
    
    # Load data
    df_train, df_test = load_data()
    
    # Prepare features
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save
    save_model(model, feature_cols, metrics)
    
    print("\n" + "="*50)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
