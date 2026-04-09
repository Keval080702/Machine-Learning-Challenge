# Feature selection module
# Uses CV based permutation importance to select top features
# All scaling is done inside CV folds to prevent data leakage

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
import catboost as cb
from config import RANDOM_STATE, N_FOLDS, N_REPEATS, BASELINE_PARAMS


def compute_cv_permutation_importance(X_train, y_train):
    """
    Compute permutation importance using k fold cross validation.
    
    This is the correct way to compute importance without data leakage.
    For each fold we train on fold train and compute importance on fold val.
    Then we average importances across all folds for stable rankings.
    
    Returns a dataframe with feature names and importance scores.
    """
    print("Computing CV based permutation importance...")
    print(f"Using {N_FOLDS} folds with {N_REPEATS} repeats per feature")
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    feature_names = X_train.columns.tolist()
    all_importances = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"Processing fold {fold_idx + 1}/{N_FOLDS}...")
        
        # Get fold data
        X_fold_train = X_train.iloc[train_idx].values
        X_fold_val = X_train.iloc[val_idx].values
        y_fold_train = y_train.iloc[train_idx].values
        y_fold_val = y_train.iloc[val_idx].values
        
        # Scale inside fold to prevent leakage
        fold_scaler = RobustScaler()
        X_fold_train = fold_scaler.fit_transform(X_fold_train)
        X_fold_val = fold_scaler.transform(X_fold_val)
        
        # Train baseline model on this fold
        model = cb.CatBoostRegressor(**BASELINE_PARAMS)
        model.fit(X_fold_train, y_fold_train, verbose=False)
        
        # Compute permutation importance on validation set
        result = permutation_importance(
            model, X_fold_val, y_fold_val,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        all_importances.append(result.importances_mean)
    
    # Average importance across all folds
    avg_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_importance,
        "std": std_importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    print("Permutation importance computation complete")
    return importance_df


def select_top_features(importance_df, X_train, y_train, k_values=None):
    """
    Test different numbers of top features using CV.
    Returns the optimal number of features and the selected feature list.
    
    If k_values is None, uses default values.
    """
    if k_values is None:
        k_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 50, 100, 200, 273]
    
    print("Testing different feature counts with CV...")
    
    results = []
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for k in k_values:
        if k > len(importance_df):
            continue
            
        # Get top k features
        top_features = importance_df.head(k)["feature"].tolist()
        feature_indices = [X_train.columns.get_loc(f) for f in top_features]
        X_train_k = X_train.iloc[:, feature_indices].values
        y_train_arr = y_train.values
        
        fold_val_r2 = []
        fold_train_r2 = []
        fold_val_rmse = []
        fold_val_mae = []
        
        for train_idx, val_idx in kfold.split(X_train_k):
            X_fold_train = X_train_k[train_idx]
            X_fold_val = X_train_k[val_idx]
            y_fold_train = y_train_arr[train_idx]
            y_fold_val = y_train_arr[val_idx]
            
            # Scale inside fold
            fold_scaler = RobustScaler()
            X_fold_train = fold_scaler.fit_transform(X_fold_train)
            X_fold_val = fold_scaler.transform(X_fold_val)
            
            # Train and evaluate
            model = cb.CatBoostRegressor(**BASELINE_PARAMS)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            train_pred = model.predict(X_fold_train)
            val_pred = model.predict(X_fold_val)
            
            fold_train_r2.append(r2_score(y_fold_train, train_pred))
            fold_val_r2.append(r2_score(y_fold_val, val_pred))
            fold_val_rmse.append(np.sqrt(mean_squared_error(y_fold_val, val_pred)))
            fold_val_mae.append(mean_absolute_error(y_fold_val, val_pred))
        
        train_r2 = np.mean(fold_train_r2)
        val_r2 = np.mean(fold_val_r2)
        val_rmse = np.mean(fold_val_rmse)
        val_mae = np.mean(fold_val_mae)
        overfitting = train_r2 - val_r2
        
        results.append({
            "k": k,
            "train_r2": train_r2,
            "val_r2_mean": val_r2,
            "val_r2_std": np.std(fold_val_r2),
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "overfitting": overfitting
        })
        print(f"K={k}: Train R2={train_r2:.4f}, Val R2={val_r2:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, Overfit={overfitting*100:.2f}%")
    
    # Find best k based on validation R2
    results_df = pd.DataFrame(results)
    best_idx = results_df["val_r2_mean"].idxmax()
    best_k = results_df.loc[best_idx, "k"]
    
    # Get selected features
    selected_features = importance_df.head(best_k)["feature"].tolist()
    
    print(f"Best K = {best_k} features with Val R2 = {results_df.loc[best_idx, 'val_r2_mean']:.4f}")
    
    # Print top 5 k values with full metrics
    print("\nTop 5 Feature Counts by Validation R2:")
    top_5_df = results_df.nlargest(5, "val_r2_mean")
    for _, row in top_5_df.iterrows():
        print(f"  K={int(row['k'])}: Train R2={row['train_r2']:.4f}, Val R2={row['val_r2_mean']:.4f}, RMSE={row['val_rmse']:.4f}, MAE={row['val_mae']:.4f}, Overfit={row['overfitting']*100:.2f}%")
    
    # Print selected features with importance
    print(f"\nTop {best_k} Selected Features:")
    for i, feat in enumerate(selected_features):
        imp = importance_df[importance_df["feature"] == feat]["importance"].values[0]
        print(f"  {i+1}. {feat}: importance = {imp:.6f}")
    
    return best_k, selected_features, results_df
# Model training module
# Handles baseline model training and Optuna hyperparameter tuning
# No scaling needed for tree-based models (CatBoost)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import catboost as cb
import optuna
from config import RANDOM_STATE, N_FOLDS, BASELINE_PARAMS, N_TRIALS


def train_baseline_model_cv(X_train_selected, y_train):
    """
    Train baseline model using k fold cross validation.
    No scaling needed for CatBoost (tree-based model).
    Returns average validation R2 and individual fold scores.
    """
    print("Training baseline model with CV...")
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_r2_scores = []
    val_r2_scores = []
    train_rmse_scores = []
    val_rmse_scores = []
    train_mae_scores = []
    val_mae_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_selected)):
        X_fold_train = X_train_selected[train_idx]
        X_fold_val = X_train_selected[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train model (no scaling needed for CatBoost)
        model = cb.CatBoostRegressor(**BASELINE_PARAMS)
        model.fit(X_fold_train, y_fold_train, verbose=False)
        
        # Evaluate
        train_pred = model.predict(X_fold_train)
        val_pred = model.predict(X_fold_val)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        train_r2_scores.append(r2_score(y_fold_train, train_pred))
        val_r2_scores.append(r2_score(y_fold_val, val_pred))
        train_rmse_scores.append(np.sqrt(mean_squared_error(y_fold_train, train_pred)))
        val_rmse_scores.append(np.sqrt(mean_squared_error(y_fold_val, val_pred)))
        train_mae_scores.append(mean_absolute_error(y_fold_train, train_pred))
        val_mae_scores.append(mean_absolute_error(y_fold_val, val_pred))
    
    avg_train_r2 = np.mean(train_r2_scores)
    avg_val_r2 = np.mean(val_r2_scores)
    avg_train_rmse = np.mean(train_rmse_scores)
    avg_val_rmse = np.mean(val_rmse_scores)
    avg_train_mae = np.mean(train_mae_scores)
    avg_val_mae = np.mean(val_mae_scores)
    overfitting = avg_train_r2 - avg_val_r2
    
    print(f"\nBaseline CV Results:")
    print(f"  Train R2:   {avg_train_r2:.4f}")
    print(f"  Train RMSE: {avg_train_rmse:.4f}")
    print(f"  Train MAE:  {avg_train_mae:.4f}")
    print(f"  Val R2:     {avg_val_r2:.4f}")
    print(f"  Val RMSE:   {avg_val_rmse:.4f}")
    print(f"  Val MAE:    {avg_val_mae:.4f}")
    print(f"  Overfitting: {overfitting*100:.2f}%")
    
    return avg_val_r2, train_r2_scores, val_r2_scores


def optuna_objective(trial, X_train_selected, y_train):
    """
    Optuna objective function for hyperparameter tuning.
    Uses k fold CV without scaling (CatBoost doesn't need it).
    """
    # User-specified hyperparameters for better balance
    params = {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 6, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "eval_metric": "R2",
        "random_state": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False
    }
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    val_r2_scores = []
    
    for train_idx, val_idx in kfold.split(X_train_selected):
        X_fold_train = X_train_selected[train_idx]
        X_fold_val = X_train_selected[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train and evaluate (no scaling)
        model = cb.CatBoostRegressor(**params)
        model.fit(X_fold_train, y_fold_train, verbose=False)
        val_pred = model.predict(X_fold_val)
        val_r2_scores.append(r2_score(y_fold_val, val_pred))
    
    return np.mean(val_r2_scores)


def tune_hyperparameters(X_train_selected, y_train):
    """
    Run Optuna hyperparameter tuning.
    Returns best parameters, best CV R2, RMSE, and MAE.
    """
    print(f"Starting Optuna hyperparameter tuning with {N_TRIALS} trials...")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, X_train_selected, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_params.update({
        "eval_metric": "R2",
        "random_state": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False
    })
    
    # Calculate CV metrics with best params
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    val_r2_scores = []
    val_rmse_scores = []
    val_mae_scores = []
    
    for train_idx, val_idx in kfold.split(X_train_selected):
        X_fold_train = X_train_selected[train_idx]
        X_fold_val = X_train_selected[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        model = cb.CatBoostRegressor(**best_params)
        model.fit(X_fold_train, y_fold_train, verbose=False)
        val_pred = model.predict(X_fold_val)
        
        val_r2_scores.append(r2_score(y_fold_val, val_pred))
        val_rmse_scores.append(np.sqrt(mean_squared_error(y_fold_val, val_pred)))
        val_mae_scores.append(mean_absolute_error(y_fold_val, val_pred))
    
    cv_r2 = np.mean(val_r2_scores)
    cv_rmse = np.mean(val_rmse_scores)
    cv_mae = np.mean(val_mae_scores)
    
    print(f"Best CV R2: {cv_r2:.4f}")
    print(f"Best CV RMSE: {cv_rmse:.4f}")
    print(f"Best CV MAE: {cv_mae:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params, cv_r2, cv_rmse, cv_mae


def train_final_model(X_train_selected, y_train, params):
    """
    Train final model on full training data with given parameters.
    Returns trained model only (no scaler needed for CatBoost).
    """
    print("Training final model on full training data...")
    
    # Train model directly (no scaling needed)
    final_model = cb.CatBoostRegressor(**params)
    final_model.fit(X_train_selected, y_train, verbose=False)
    
    print("Final model trained successfully")
    
    return final_model
# Evaluation module
# Calculates model performance metrics

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate R2, RMSE, and MAE for given predictions.
    Returns a dictionary with all metrics.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model on both train and test sets.
    Test set is evaluated only once at the end.
    No scaling needed for CatBoost (tree-based model).
    Prints results and returns metrics dictionary.
    """
    # Get predictions directly (no scaling)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    
    # Print results
    print("\nModel Evaluation Results")
    print("-" * 40)
    print(f"Train R2:   {train_metrics['r2']:.4f}")
    print(f"Train RMSE: {train_metrics['rmse']:.4f}")
    print(f"Train MAE:  {train_metrics['mae']:.4f}")
    print("-" * 40)
    print(f"Test R2:    {test_metrics['r2']:.4f}")
    print(f"Test RMSE:  {test_metrics['rmse']:.4f}")
    print(f"Test MAE:   {test_metrics['mae']:.4f}")
    print("-" * 40)
    
    overfitting = train_metrics["r2"] - test_metrics["r2"]
    print(f"Overfitting: {overfitting*100:.2f} percent")
    
    return {
        "train": train_metrics,
        "test": test_metrics,
        "overfitting": overfitting
    }
