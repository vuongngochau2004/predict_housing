"""
Train multiple ML models for house price prediction with:
- K-Fold Cross Validation
- Optuna Hyperparameter Optimization for ALL models
- Model comparison and selection
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
TRAIN_DATA_PATH = 'data/train_data.csv'
TEST_DATA_PATH = 'data/test_data.csv'

# Target variable
TARGET = 'Giá'  # Target column name

# Cross-validation settings
N_FOLDS = 5
N_OPTUNA_TRIALS = 30  # Trials per model
RANDOM_STATE = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load train and test data"""
    print("📁 Loading data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    print(f"   Train: {len(df_train):,} rows")
    print(f"   Test: {len(df_test):,} rows")
    
    return df_train, df_test


def prepare_features(df_train, df_test):
    """Prepare features and target"""
    print("\n🔧 Preparing features...")
    
    target = TARGET
    feature_cols = [col for col in df_train.columns if col != target]
    
    X_train = df_train[feature_cols]
    y_train = df_train[target]
    
    X_test = df_test[feature_cols]
    y_test = df_test[target]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target: {target}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def clean_feature_names(df):
    """Clean feature names for LightGBM compatibility"""
    import re
    new_columns = {}
    for col in df.columns:
        new_col = col.replace('(', '_').replace(')', '_').replace(' ', '_')
        new_col = new_col.replace('/', '_').replace(',', '_').replace('.', '_')
        new_col = re.sub(r'[^a-zA-Z0-9_]', '', new_col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        new_columns[col] = new_col
    return df.rename(columns=new_columns), new_columns


def convert_categorical_columns(X_train, X_test):
    """Convert object columns to category type for LightGBM native categorical support"""
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    
    return X_train, X_test, categorical_cols


def label_encode_for_sklearn(X_train, X_test, categorical_cols):
    """Create label-encoded copies for sklearn models (RandomForest)"""
    from sklearn.preprocessing import LabelEncoder
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on combined data to handle all categories
        all_values = pd.concat([X_train[col].astype(str), X_test[col].astype(str)]).unique()
        le.fit(all_values)
        
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    return X_train_encoded, X_test_encoded, label_encoders


# ============================================================================
# OPTUNA OBJECTIVE FUNCTIONS FOR EACH MODEL
# ============================================================================

def create_lightgbm_objective(X_train, y_train, n_folds=N_FOLDS):
    """Create Optuna objective for LightGBM"""
    
    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        }
        
        try:
            model = LGBMRegressor(**param)
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=n_folds, 
                scoring='neg_root_mean_squared_error'
            )
            rmse = -scores.mean()
            return rmse if not np.isnan(rmse) else 1e10
        except Exception:
            return 1e10
    
    return objective


def create_randomforest_objective(X_train, y_train, n_folds=N_FOLDS):
    """Create Optuna objective for RandomForest"""
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }
        
        try:
            model = RandomForestRegressor(**param)
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=n_folds, 
                scoring='neg_root_mean_squared_error'
            )
            rmse = -scores.mean()
            return rmse if not np.isnan(rmse) else 1e10
        except Exception:
            return 1e10
    
    return objective


def create_catboost_objective(X_train, y_train, cat_features, n_folds=N_FOLDS):
    """Create Optuna objective for CatBoost using native CV"""
    from catboost import cv as catboost_cv
    
    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': RANDOM_STATE,
            'verbose': 0,
            'loss_function': 'RMSE',
        }
        
        try:
            # Create CatBoost Pool with categorical features
            pool = Pool(
                data=X_train, 
                label=y_train, 
                cat_features=cat_features
            )
            
            # Use CatBoost native CV
            cv_results = catboost_cv(
                pool=pool,
                params=param,
                fold_count=n_folds,
                seed=RANDOM_STATE,
                verbose=False,
                early_stopping_rounds=50
            )
            
            # Get final RMSE from CV results
            rmse = cv_results['test-RMSE-mean'].iloc[-1]
            return rmse if not np.isnan(rmse) else 1e10
        except Exception as e:
            return 1e10
    
    return objective


# ============================================================================
# OPTUNA OPTIMIZATION FOR ALL MODELS
# ============================================================================

def optimize_all_models(X_train, y_train, X_train_encoded, cat_features, n_trials=N_OPTUNA_TRIALS):
    """Run Optuna optimization for all models"""
    print("\n" + "="*70)
    print("🔬 OPTUNA HYPERPARAMETER OPTIMIZATION FOR ALL MODELS")
    print("="*70)
    print(f"   K-Fold CV: {N_FOLDS} folds")
    print(f"   Trials per model: {n_trials}")
    
    sampler = TPESampler(seed=RANDOM_STATE)
    all_best_params = {}
    all_studies = {}
    
    # 1. LightGBM (uses category dtype)
    print("\n" + "-"*70)
    print("🌲 [1/3] Optimizing LightGBM...")
    study_lgbm = optuna.create_study(direction='minimize', sampler=sampler)
    study_lgbm.optimize(
        create_lightgbm_objective(X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    all_best_params['LightGBM'] = study_lgbm.best_params
    all_studies['LightGBM'] = study_lgbm
    print(f"   ✅ Best CV RMSE: {study_lgbm.best_value:.4f}")
    
    # 2. RandomForest (uses label-encoded data)
    print("\n" + "-"*70)
    print("🌳 [2/3] Optimizing RandomForest...")
    study_rf = optuna.create_study(direction='minimize', sampler=sampler)
    study_rf.optimize(
        create_randomforest_objective(X_train_encoded, y_train),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    all_best_params['RandomForest'] = study_rf.best_params
    all_studies['RandomForest'] = study_rf
    print(f"   ✅ Best CV RMSE: {study_rf.best_value:.4f}")
    
    # 3. CatBoost (uses category dtype with cat_features indices)
    print("\n" + "-"*70)
    print("🐱 [3/3] Optimizing CatBoost...")
    study_cb = optuna.create_study(direction='minimize', sampler=sampler)
    study_cb.optimize(
        create_catboost_objective(X_train, y_train, cat_features),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    all_best_params['CatBoost'] = study_cb.best_params
    all_studies['CatBoost'] = study_cb
    print(f"   ✅ Best CV RMSE: {study_cb.best_value:.4f}")
    
    print("\n" + "="*70)
    print("✅ Optimization completed for all models!")
    
    return all_best_params, all_studies


# ============================================================================
# TRAINING WITH K-FOLD CV
# ============================================================================

def train_with_kfold(X_train, y_train, X_test, y_test, 
                      X_train_encoded, X_test_encoded,
                      best_params, cat_features):
    """Train all models with K-Fold CV and optimized parameters"""
    print("\n" + "="*70)
    print("🏋️ TRAINING ALL MODELS WITH K-FOLD CROSS VALIDATION")
    print("="*70)
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    all_models = {}
    all_cv_scores = {}
    all_test_metrics = {}
    
    # Define model configs with correct data sources
    models_config = {
        'LightGBM': {
            'class': LGBMRegressor,
            'X_train': X_train,  # Uses category dtype
            'X_test': X_test,
            'extra_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbosity': -1
            }
        },
        'RandomForest': {
            'class': RandomForestRegressor,
            'X_train': X_train_encoded,  # Uses label-encoded data
            'X_test': X_test_encoded,
            'extra_params': {
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
        },
        'CatBoost': {
            'class': CatBoostRegressor,
            'X_train': X_train,  # Uses category dtype
            'X_test': X_test,
            'extra_params': {
                'random_state': RANDOM_STATE,
                'verbose': 0,
                'cat_features': cat_features
            }
        }
    }
    
    for model_name, model_config in models_config.items():
        print(f"\n📊 Training {model_name}...")
        print("-"*50)
        
        # Get correct data for this model
        X_tr = model_config['X_train']
        X_te = model_config['X_test']
        
        # Merge best params with extra params
        params = best_params[model_name].copy()
        params.update(model_config['extra_params'])
        
        cv_scores = {'rmse': [], 'mae': [], 'r2': []}
        
        # K-Fold CV
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_tr), 1):
            X_fold_train, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = model_config['class'](**params)
            model.fit(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            mae = mean_absolute_error(y_fold_val, y_pred)
            r2 = r2_score(y_fold_val, y_pred)
            
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            
            print(f"   Fold {fold}: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        print("-"*50)
        print(f"   Mean: RMSE={np.mean(cv_scores['rmse']):.4f} ± {np.std(cv_scores['rmse']):.4f}")
        print(f"         R²={np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
        
        all_cv_scores[model_name] = cv_scores
        
        # Train final model on full data
        print(f"\n   🎯 Training final {model_name} on full data...")
        final_model = model_config['class'](**params)
        final_model.fit(X_tr, y_train)
        all_models[model_name] = final_model
        
        # Evaluate on test set
        y_pred_test = final_model.predict(X_te)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        all_test_metrics[model_name] = {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2
        }
        print(f"   📊 Test: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}")
    
    return all_models, all_cv_scores, all_test_metrics


# ============================================================================
# COMPARISON & SAVING
# ============================================================================

def print_comparison(all_test_metrics, all_cv_scores):
    """Print comparison table"""
    print("\n" + "="*70)
    print("📈 FINAL MODEL COMPARISON")
    print("="*70)
    
    print(f"\n{'Model':<15} {'Test RMSE':<12} {'Test MAE':<12} {'Test R²':<10} {'CV R² (mean±std)':<20}")
    print("-"*70)
    
    sorted_results = sorted(all_test_metrics.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    for i, (name, metrics) in enumerate(sorted_results):
        prefix = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        cv_r2_mean = np.mean(all_cv_scores[name]['r2'])
        cv_r2_std = np.std(all_cv_scores[name]['r2'])
        print(f"{prefix} {name:<12} {metrics['rmse']:<12.4f} {metrics['mae']:<12.4f} {metrics['r2']:<10.4f} {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    
    print("="*70)
    
    best_model_name = sorted_results[0][0]
    print(f"\n🏆 Best Model: {best_model_name}")
    
    return best_model_name


def save_all_models(all_models, all_test_metrics, all_cv_scores, best_params, 
                    feature_cols, best_model_name):
    """Save all models and metadata"""
    print("\n💾 Saving models and metadata...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save all models
    for name, model in all_models.items():
        filepath = f'models/{name.lower()}_optuna_model.joblib'
        joblib.dump(model, filepath)
        print(f"   ✅ Saved: {filepath}")
    
    # Save best model as default
    joblib.dump(all_models[best_model_name], 'models/model.joblib')
    print(f"   ✅ Saved: models/model.joblib (best: {best_model_name})")
    
    # Save feature names
    with open('models/feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print("   ✅ Saved: models/feature_names.json")
    
    # Save best hyperparameters for all models
    params_to_save = {}
    for name, params in best_params.items():
        params_to_save[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else 
               int(v) if isinstance(v, (np.integer, int)) else v 
            for k, v in params.items() if k != 'cat_features'
        }
    
    with open('models/best_hyperparams.json', 'w', encoding='utf-8') as f:
        json.dump(params_to_save, f, indent=2)
    print("   ✅ Saved: models/best_hyperparams.json")
    
    # Save CV scores
    cv_scores_to_save = {}
    for name, scores in all_cv_scores.items():
        cv_scores_to_save[name] = {
            'rmse_mean': float(np.mean(scores['rmse'])),
            'rmse_std': float(np.std(scores['rmse'])),
            'mae_mean': float(np.mean(scores['mae'])),
            'mae_std': float(np.std(scores['mae'])),
            'r2_mean': float(np.mean(scores['r2'])),
            'r2_std': float(np.std(scores['r2'])),
            'fold_scores': {
                'rmse': [float(x) for x in scores['rmse']],
                'mae': [float(x) for x in scores['mae']],
                'r2': [float(x) for x in scores['r2']]
            }
        }
    
    with open('models/cv_scores.json', 'w', encoding='utf-8') as f:
        json.dump({'n_folds': N_FOLDS, 'models': cv_scores_to_save}, f, indent=2)
    print("   ✅ Saved: models/cv_scores.json")
    
    # Save all test metrics
    all_metrics = {}
    for name, metrics in all_test_metrics.items():
        all_metrics[name] = {k: float(v) for k, v in metrics.items()}
    
    with open('models/all_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print("   ✅ Saved: models/all_metrics.json")
    
    # Save best model metrics
    best_metrics = all_test_metrics[best_model_name]
    with open('models/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_model': best_model_name,
            'optuna_trials': N_OPTUNA_TRIALS,
            'cv_folds': N_FOLDS,
            'rmse': float(best_metrics['rmse']),
            'mae': float(best_metrics['mae']),
            'r2': float(best_metrics['r2'])
        }, f, indent=2)
    print("   ✅ Saved: models/metrics.json")


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_visualizations(all_studies, all_cv_scores, all_test_metrics, best_model_name):
    """Save training visualizations to outputs folder"""
    print("\n📊 Saving visualizations...")
    
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Optuna Optimization History
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {'LightGBM': '#3498db', 'RandomForest': '#2ecc71', 'CatBoost': '#e74c3c'}
    
    for idx, (name, study) in enumerate(all_studies.items()):
        ax = axes[idx]
        trials = [t.value for t in study.trials if t.value is not None]
        ax.plot(range(1, len(trials)+1), trials, 'o-', color=colors[name], alpha=0.7, markersize=4)
        ax.axhline(y=study.best_value, color='red', linestyle='--', alpha=0.5, label=f'Best: {study.best_value:.4f}')
        ax.set_xlabel('Trial')
        ax.set_ylabel('RMSE (CV)')
        ax.set_title(f'{name} Optimization')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/optuna_optimization_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/optuna_optimization_history.png")
    
    # 2. Model Comparison (Test Metrics)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    models = list(all_test_metrics.keys())
    
    # RMSE
    rmse_vals = [all_test_metrics[m]['rmse'] for m in models]
    bars = axes[0].bar(models, rmse_vals, color=[colors[m] for m in models], edgecolor='black')
    axes[0].set_ylabel('RMSE (tỷ VND)')
    axes[0].set_title('Test RMSE (lower is better)')
    for bar, val in zip(bars, rmse_vals):
        axes[0].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    # MAE
    mae_vals = [all_test_metrics[m]['mae'] for m in models]
    bars = axes[1].bar(models, mae_vals, color=[colors[m] for m in models], edgecolor='black')
    axes[1].set_ylabel('MAE (tỷ VND)')
    axes[1].set_title('Test MAE (lower is better)')
    for bar, val in zip(bars, mae_vals):
        axes[1].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    # R²
    r2_vals = [all_test_metrics[m]['r2'] for m in models]
    bars = axes[2].bar(models, r2_vals, color=[colors[m] for m in models], edgecolor='black')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('Test R² (higher is better)')
    axes[2].set_ylim(0, 1)
    for bar, val in zip(bars, r2_vals):
        axes[2].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'Model Comparison (Best: {best_model_name})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/model_comparison.png")
    
    # 3. K-Fold CV Scores
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_FOLDS)
    width = 0.25
    
    for i, (name, scores) in enumerate(all_cv_scores.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, scores['r2'], width, label=name, color=colors[name], alpha=0.8)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('R² Score')
    ax.set_title('K-Fold Cross Validation R² Scores')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(N_FOLDS)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('outputs/cv_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/cv_scores.png")
    
    # 4. Summary Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: Best RMSE per model from Optuna
    best_rmse = [all_studies[m].best_value for m in models]
    axes[0, 0].barh(models, best_rmse, color=[colors[m] for m in models])
    axes[0, 0].set_xlabel('Best CV RMSE')
    axes[0, 0].set_title('Optuna Best CV RMSE')
    
    # Top-right: Test R² comparison
    axes[0, 1].barh(models, r2_vals, color=[colors[m] for m in models])
    axes[0, 1].set_xlabel('R² Score')
    axes[0, 1].set_title('Test R² Score')
    axes[0, 1].set_xlim(0, 1)
    
    # Bottom-left: CV R² mean ± std
    cv_r2_means = [np.mean(all_cv_scores[m]['r2']) for m in models]
    cv_r2_stds = [np.std(all_cv_scores[m]['r2']) for m in models]
    axes[1, 0].barh(models, cv_r2_means, xerr=cv_r2_stds, color=[colors[m] for m in models], capsize=5)
    axes[1, 0].set_xlabel('R² Score (mean ± std)')
    axes[1, 0].set_title('Cross-Validation R² (5-Fold)')
    axes[1, 0].set_xlim(0, 1)
    
    # Bottom-right: Text summary
    axes[1, 1].axis('off')
    summary_text = f"""
    🏆 Training Summary
    
    Best Model: {best_model_name}
    
    Test Metrics:
    • RMSE: {all_test_metrics[best_model_name]['rmse']:.4f} tỷ VND
    • MAE:  {all_test_metrics[best_model_name]['mae']:.4f} tỷ VND  
    • R²:   {all_test_metrics[best_model_name]['r2']:.4f}
    
    CV Metrics ({N_FOLDS}-Fold):
    • R²: {np.mean(all_cv_scores[best_model_name]['r2']):.4f} ± {np.std(all_cv_scores[best_model_name]['r2']):.4f}
    
    Training Config:
    • Optuna Trials: {N_OPTUNA_TRIALS} per model
    • K-Fold CV: {N_FOLDS} folds
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('House Price Prediction - Training Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/training_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/training_summary.png")


def save_model_comparison_only(all_cv_scores, all_test_metrics, best_model_name):
    """Save only model comparison visualizations (when using cached hyperparams)"""
    print("\n📊 Saving model comparison visualizations...")
    
    colors = {'LightGBM': '#3498db', 'RandomForest': '#2ecc71', 'CatBoost': '#e74c3c'}
    models = list(all_test_metrics.keys())
    
    # Model Comparison (Test Metrics)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    rmse_vals = [all_test_metrics[m]['rmse'] for m in models]
    bars = axes[0].bar(models, rmse_vals, color=[colors[m] for m in models], edgecolor='black')
    axes[0].set_ylabel('RMSE (tỷ VND)')
    axes[0].set_title('Test RMSE (lower is better)')
    for bar, val in zip(bars, rmse_vals):
        axes[0].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    mae_vals = [all_test_metrics[m]['mae'] for m in models]
    bars = axes[1].bar(models, mae_vals, color=[colors[m] for m in models], edgecolor='black')
    axes[1].set_ylabel('MAE (tỷ VND)')
    axes[1].set_title('Test MAE (lower is better)')
    for bar, val in zip(bars, mae_vals):
        axes[1].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    r2_vals = [all_test_metrics[m]['r2'] for m in models]
    bars = axes[2].bar(models, r2_vals, color=[colors[m] for m in models], edgecolor='black')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('Test R² (higher is better)')
    axes[2].set_ylim(0, 1)
    for bar, val in zip(bars, r2_vals):
        axes[2].annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'Model Comparison (Best: {best_model_name})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/model_comparison.png")
    
    # K-Fold CV Scores
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(N_FOLDS)
    width = 0.25
    
    for i, (name, scores) in enumerate(all_cv_scores.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, scores['r2'], width, label=name, color=colors[name], alpha=0.8)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('R² Score')
    ax.set_title('K-Fold Cross Validation R² Scores')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(N_FOLDS)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('outputs/cv_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: outputs/cv_scores.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🏠 HOUSE PRICE PREDICTION - OPTUNA + K-FOLD CV FOR ALL MODELS")
    print("="*70)
    print(f"Configuration:")
    print(f"   - K-Fold CV: {N_FOLDS} folds")
    print(f"   - Optuna trials per model: {N_OPTUNA_TRIALS}")
    print(f"   - Models: LightGBM, RandomForest, CatBoost")
    print(f"   - Random state: {RANDOM_STATE}")
    
    # Load data
    df_train, df_test = load_data()
    
    # Prepare features
    X_train, y_train, X_test, y_test, feature_cols = prepare_features(df_train, df_test)
    
    # Clean feature names
    print("\n🧹 Cleaning feature names...")
    X_train, col_mapping = clean_feature_names(X_train)
    X_test = X_test.rename(columns=col_mapping)
    print(f"   ✅ Cleaned {len(col_mapping)} feature names")
    
    # Convert categorical columns (for LightGBM/CatBoost)
    print("\n📝 Converting categorical columns...")
    X_train, X_test, cat_cols = convert_categorical_columns(X_train.copy(), X_test.copy())
    
    # Get indices for CatBoost
    cat_features_indices = [X_train.columns.get_loc(col) for col in cat_cols] if cat_cols else []
    print(f"   ✅ Found {len(cat_cols)} categorical columns: {cat_cols}")
    
    # Create label-encoded data for sklearn (RandomForest)
    print("\n🔢 Creating label-encoded data for sklearn models...")
    X_train_encoded, X_test_encoded, _ = label_encode_for_sklearn(X_train, X_test, cat_cols)
    print(f"   ✅ Label encoding complete")
    
    # Step 1: Check for existing hyperparams or run Optuna optimization
    hyperparams_path = 'models/best_hyperparams.json'
    studies = None
    
    if os.path.exists(hyperparams_path):
        print("\n" + "="*70)
        print("📂 FOUND EXISTING HYPERPARAMETERS")
        print("="*70)
        
        with open(hyperparams_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        
        print("   ✅ Loaded from: models/best_hyperparams.json")
        print("   💡 To re-run Optuna, delete this file and run again")
        
        for model_name, params in best_params.items():
            print(f"\n   {model_name}:")
            for k, v in list(params.items())[:3]:
                print(f"      {k}: {v}")
            if len(params) > 3:
                print(f"      ... and {len(params)-3} more")
    else:
        print("\n   ⚠️ No existing hyperparams found, running Optuna optimization...")
        best_params, studies = optimize_all_models(
            X_train, y_train, X_train_encoded, cat_features_indices, N_OPTUNA_TRIALS
        )
    
    # Step 2: Train all models with K-Fold CV
    all_models, all_cv_scores, all_test_metrics = train_with_kfold(
        X_train, y_train, X_test, y_test, 
        X_train_encoded, X_test_encoded,
        best_params, cat_features_indices
    )
    
    # Step 3: Compare models
    best_model_name = print_comparison(all_test_metrics, all_cv_scores)
    
    # Step 4: Save everything
    save_all_models(all_models, all_test_metrics, all_cv_scores, best_params, 
                   list(col_mapping.keys()), best_model_name)
    
    # Save column mapping
    with open('models/column_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(col_mapping, f, ensure_ascii=False, indent=2)
    print("   ✅ Saved: models/column_mapping.json")
    
    # Step 5: Save visualizations (only if we ran Optuna)
    if studies is not None:
        save_visualizations(studies, all_cv_scores, all_test_metrics, best_model_name)
    else:
        print("\n📊 Skipping Optuna visualizations (used cached hyperparams)")
        # Still save model comparison visualizations
        os.makedirs('outputs', exist_ok=True)
        save_model_comparison_only(all_cv_scores, all_test_metrics, best_model_name)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("  📦 Models:")
    print("     - models/model.joblib (best model)")
    print("     - models/lightgbm_optuna_model.joblib")
    print("     - models/randomforest_optuna_model.joblib")
    print("     - models/catboost_optuna_model.joblib")
    print("  📊 Metadata:")
    print("     - models/best_hyperparams.json")
    print("     - models/cv_scores.json")
    print("     - models/all_metrics.json")
    print("     - models/column_mapping.json")
    print("  📈 Visualizations:")
    print("     - outputs/optuna_optimization_history.png")
    print("     - outputs/model_comparison.png")
    print("     - outputs/cv_scores.png")
    print("     - outputs/training_summary.png")
    print("="*70)


if __name__ == "__main__":
    main()
