# 📊 Training Strategy Documentation

## Overview

This document describes the machine learning training strategy used for house price prediction in `src/train_model.py`.

---

## 1. Models Used

| Model | Library | Categorical Handling |
|-------|---------|---------------------|
| **LightGBM** | `lightgbm` | Native categorical support (`category` dtype) |
| **RandomForest** | `sklearn` | Label Encoding (numeric conversion) |
| **CatBoost** | `catboost` | Native categorical support with `cat_features` |

---

## 2. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Load train/test data                                         │
│  2. Clean feature names (remove special chars for LightGBM)      │
│  3. Convert categorical columns to `category` dtype              │
│  4. Create label-encoded copy for sklearn (RandomForest)         │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OPTUNA HYPERPARAMETER TUNING                   │
├─────────────────────────────────────────────────────────────────┤
│  For each model:                                                 │
│  • Run 30 trials with TPE Sampler                                │
│  • 5-Fold Cross Validation per trial                             │
│  • Optimize for minimum RMSE                                     │
│  • Save best hyperparameters                                     │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    K-FOLD CROSS VALIDATION                       │
├─────────────────────────────────────────────────────────────────┤
│  For each model with optimized params:                           │
│  • Train on 5 folds                                              │
│  • Compute CV metrics (RMSE, MAE, R²)                            │
│  • Train final model on full training data                       │
│  • Evaluate on held-out test set                                 │
└─────────────────────────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SELECTION & SAVING                      │
├─────────────────────────────────────────────────────────────────┤
│  • Compare all models on test metrics                            │
│  • Select best model based on Test RMSE                          │
│  • Save all models and metadata to `models/`                     │
│  • Generate visualizations to `outputs/`                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Optuna Hyperparameter Search Spaces

### LightGBM

| Parameter | Range | Type |
|-----------|-------|------|
| `n_estimators` | 100 - 2000 | int |
| `learning_rate` | 0.001 - 0.3 | log float |
| `num_leaves` | 20 - 300 | int |
| `max_depth` | 3 - 15 | int |
| `min_child_samples` | 5 - 100 | int |
| `colsample_bytree` | 0.5 - 1.0 | float |
| `subsample` | 0.5 - 1.0 | float |
| `reg_alpha` | 1e-8 - 10 | log float |
| `reg_lambda` | 1e-8 - 10 | log float |

### RandomForest

| Parameter | Range | Type |
|-----------|-------|------|
| `n_estimators` | 100 - 1000 | int |
| `max_depth` | 5 - 30 | int |
| `min_samples_split` | 2 - 20 | int |
| `min_samples_leaf` | 1 - 10 | int |
| `max_features` | sqrt, log2, None | categorical |

### CatBoost

| Parameter | Range | Type |
|-----------|-------|------|
| `iterations` | 200 - 2000 | int |
| `learning_rate` | 0.001 - 0.3 | log float |
| `depth` | 4 - 12 | int |
| `l2_leaf_reg` | 1e-8 - 100 | log float |
| `border_count` | 32 - 255 | int |

---

## 4. Cross-Validation Strategy

- **Method:** K-Fold Cross Validation
- **Folds:** 5
- **Shuffle:** Yes
- **Random State:** 42

### Why K-Fold CV?

1. **Robust evaluation:** Performance averaged across 5 different train/val splits
2. **Variance estimation:** Standard deviation shows model stability
3. **Better generalization:** Reduces overfitting risk

---

## 5. Evaluation Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **RMSE** | Root Mean Squared Error | tỷ VND |
| **MAE** | Mean Absolute Error | tỷ VND |
| **R²** | Coefficient of Determination | 0-1 |

### Model Selection Criterion

**Best model = Lowest Test RMSE**

---

## 6. Categorical Feature Handling

### Challenge

Different models require different data formats:
- **sklearn (RandomForest):** Numeric only, no categorical support
- **LightGBM:** Native `category` dtype support
- **CatBoost:** Native categorical with indices

### Solution

```python
# Original data with categorical columns
X_train_category = convert_categorical_columns(X_train)

# Label-encoded copy for sklearn
X_train_encoded = label_encode_for_sklearn(X_train)

# Use correct format for each model:
# - LightGBM/CatBoost → X_train_category
# - RandomForest → X_train_encoded
```

---

## 7. Output Files

### Models (`models/`)

| File | Description |
|------|-------------|
| `model.joblib` | Best model (production-ready) |
| `lightgbm_optuna_model.joblib` | Optimized LightGBM |
| `randomforest_optuna_model.joblib` | Optimized RandomForest |
| `catboost_optuna_model.joblib` | Optimized CatBoost |

### Metadata (`models/`)

| File | Description |
|------|-------------|
| `best_hyperparams.json` | Optimized hyperparameters for all models |
| `cv_scores.json` | K-Fold CV scores per fold |
| `all_metrics.json` | Test metrics for all models |
| `metrics.json` | Best model metrics (for Streamlit) |
| `feature_names.json` | List of feature names |
| `column_mapping.json` | Original → cleaned column name mapping |

### Visualizations (`outputs/`)

| File | Description |
|------|-------------|
| `optuna_optimization_history.png` | Hyperparam search progress |
| `model_comparison.png` | RMSE/MAE/R² comparison bars |
| `cv_scores.png` | R² per fold for all models |
| `training_summary.png` | Overall summary figure |

---

## 8. Configuration

```python
# Data paths
TRAIN_DATA_PATH = 'data/train_data.csv'
TEST_DATA_PATH = 'data/test_data.csv'

# Target variable
TARGET = 'Giá'  # Price in tỷ VND

# Training config
N_FOLDS = 5              # K-Fold splits
N_OPTUNA_TRIALS = 30     # Trials per model
RANDOM_STATE = 42        # Reproducibility
```

---

## 9. Time Estimates

| Phase | Estimated Time |
|-------|---------------|
| Optuna LightGBM | ~5-10 min |
| Optuna RandomForest | ~3-5 min |
| Optuna CatBoost | ~10-20 min |
| K-Fold Training | ~5-10 min |
| **Total** | **~25-45 min** |

---

## 10. Usage

```bash
# Activate environment
source .venv/bin/activate

# Run training
uv run python src/train_model.py

# Run Streamlit app (after training)
streamlit run app.py
```
