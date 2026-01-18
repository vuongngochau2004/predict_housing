"""
Train and compare multiple ML models for house price prediction
Models: Random Forest, LightGBM, CatBoost
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================================
# DATA LOADING
# ============================================================================

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
    
    target = config.TARGET
    feature_cols = [col for col in df_train.columns if col != target]
    
    X_train = df_train[feature_cols]
    y_train = df_train[target]
    
    X_test = df_test[feature_cols]
    y_test = df_test[target]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target: {target}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def clean_feature_names(df):
    """Clean feature names for LightGBM compatibility (remove special chars)"""
    import re
    new_columns = {}
    for col in df.columns:
        # Replace Vietnamese chars and special chars
        new_col = col.replace('(', '_').replace(')', '_').replace(' ', '_')
        new_col = new_col.replace('/', '_').replace(',', '_').replace('.', '_')
        new_col = re.sub(r'[^a-zA-Z0-9_]', '', new_col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        new_columns[col] = new_col
    return df.rename(columns=new_columns), new_columns


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models():
    """Define all models to train"""
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=100,
            max_depth=15,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostRegressor(
            iterations=100,
            depth=10,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
    }
    return models


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    # Predict (log scale)
    y_pred_log = model.predict(X_test)
    
    # Convert to original scale
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def train_all_models(X_train, y_train, X_test, y_test):
    """Train all models and compare performance"""
    models = get_models()
    results = {}
    trained_models = {}
    
    print("\n" + "="*60)
    print("🏋️ TRAINING MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n📊 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        
        print(f"   ✅ MAE:  {metrics['mae']/1e9:.2f} tỷ VND")
        print(f"   ✅ RMSE: {metrics['rmse']/1e9:.2f} tỷ VND")
        print(f"   ✅ R²:   {metrics['r2']:.4f}")
        print(f"   ✅ MAPE: {metrics['mape']:.2f}%")
    
    return trained_models, results


def print_comparison(results):
    """Print comparison table"""
    print("\n" + "="*60)
    print("📈 MODEL COMPARISON")
    print("="*60)
    
    # Header
    print(f"{'Model':<15} {'MAE (tỷ)':<12} {'RMSE (tỷ)':<12} {'R²':<10} {'MAPE (%)':<10}")
    print("-"*60)
    
    # Sort by R² (best first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    for i, (name, metrics) in enumerate(sorted_results):
        prefix = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"{prefix} {name:<12} {metrics['mae']/1e9:<12.2f} {metrics['rmse']/1e9:<12.2f} {metrics['r2']:<10.4f} {metrics['mape']:<10.2f}")
    
    print("="*60)
    
    # Best model
    best_model_name = sorted_results[0][0]
    print(f"\n🏆 Best Model: {best_model_name}")
    
    return best_model_name


# ============================================================================
# SAVING
# ============================================================================

def save_models(trained_models, results, feature_cols, best_model_name):
    """Save all models and metadata"""
    print("\n💾 Saving models...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save all models
    for name, model in trained_models.items():
        filepath = f'models/{name.lower()}_model.joblib'
        joblib.dump(model, filepath)
        print(f"   ✅ Saved: {filepath}")
    
    # Save best model as default
    joblib.dump(trained_models[best_model_name], 'models/model.joblib')
    print(f"   ✅ Saved: models/model.joblib (best: {best_model_name})")
    
    # Save feature names
    with open('models/feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print("   ✅ Saved: models/feature_names.json")
    
    # Save all metrics
    all_metrics = {}
    for name, metrics in results.items():
        all_metrics[name] = {
            'mae_billion': metrics['mae'] / 1e9,
            'rmse_billion': metrics['rmse'] / 1e9,
            'r2': metrics['r2'],
            'mape': metrics['mape']
        }
    
    with open('models/all_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print("   ✅ Saved: models/all_metrics.json")
    
    # Save best model metrics (for backward compatibility)
    with open('models/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_model': best_model_name,
            'mae_billion': results[best_model_name]['mae'] / 1e9,
            'rmse_billion': results[best_model_name]['rmse'] / 1e9,
            'r2': results[best_model_name]['r2'],
            'mape': results[best_model_name]['mape']
        }, f, indent=2)
    print("   ✅ Saved: models/metrics.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🏠 HOUSE PRICE PREDICTION - MULTI-MODEL TRAINING")
    print("="*60)
    print("Models: Random Forest, LightGBM, CatBoost")
    
    # Load data
    df_train, df_test = load_data()
    
    # Prepare features
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # Clean feature names for LightGBM compatibility
    print("\n🧹 Cleaning feature names for LightGBM compatibility...")
    X_train, col_mapping = clean_feature_names(X_train)
    X_test = X_test.rename(columns=col_mapping)
    feature_cols = [col_mapping.get(c, c) for c in feature_cols]
    print(f"   ✅ Cleaned {len(col_mapping)} feature names")
    
    # Train all models
    trained_models, results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Compare and find best
    best_model_name = print_comparison(results)
    
    # Save (use original feature names for app compatibility)
    save_models(trained_models, results, list(col_mapping.keys()), best_model_name)
    
    # Also save the column mapping
    with open('models/column_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(col_mapping, f, ensure_ascii=False, indent=2)
    print("   ✅ Saved: models/column_mapping.json")
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED AND SAVED!")
    print("="*60)
    print("\nFiles created:")
    print("  - models/randomforest_model.joblib")
    print("  - models/lightgbm_model.joblib")
    print("  - models/catboost_model.joblib")
    print("  - models/model.joblib (best model)")
    print("  - models/all_metrics.json")
    print("  - models/column_mapping.json")
    print("="*60)


if __name__ == "__main__":
    main()
