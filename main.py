# Main entry point for Part 1 ML Challenge
# Runs the complete pipeline from data loading to prediction generation

# This pipeline follows these steps:
# 1. Load training and evaluation data
# 2. Split data into train and test sets (80/20)
# 3. Compute CV based permutation importance for feature selection
# 4. Select optimal number of features based on CV validation R2
# 5. Train baseline model with selected features
# 6. Tune hyperparameters using Optuna
# 7. Train final model with best parameters
# 8. Evaluate on test set (evaluated only once)
# 9. Generate predictions for evaluation dataset

# Data leakage prevention:
# - Feature selection uses CV based importance
# - Test set is never used during training or tuning
# Note: No scaling needed for CatBoost (tree-based model)

import numpy as np
from pathlib import Path
from data_utils import load_training_data, load_eval_data, split_data, generate_predictions
from modeling import compute_cv_permutation_importance, select_top_features, train_baseline_model_cv, tune_hyperparameters, train_final_model, evaluate_model
from visualization import (
    plot_feature_importance, 
    plot_feature_selection_results,
    plot_predictions_vs_actual_train_test,
    plot_residuals_train_test,
    plot_test_predictions_with_residuals_and_errors,
    plot_selected_features_distribution,
    plot_selected_features_importance_boxplot,
    plot_impurity_vs_permutation
)
from config import BASELINE_PARAMS, VIS_DIR


def main():
    """
    Main function that runs the complete ML pipeline.
    """
    print("=" * 60)
    print("Part 1: ML Challenge - Target01 Prediction")
    print("=" * 60)
    
    # Create visualization directory if it doesn't exist
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Visualizations will be saved to: {VIS_DIR}")
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    X, y = load_training_data()
    X_eval = load_eval_data()
    
    # Step 2: Split data
    print("\nStep 2: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 3: Compute feature importance
    print("\nStep 3: Computing feature importance...")
    importance_df = compute_cv_permutation_importance(X_train, y_train)
    
    # Save importance plot
    plot_feature_importance(importance_df, top_n=20, save_path=str(VIS_DIR / "feature_importance.png"))
    
    # Step 4: Select optimal features
    print("\nStep 4: Selecting optimal features...")
    best_k, selected_features, results_df = select_top_features(importance_df, X_train, y_train)
    
    # Save feature selection plot
    plot_feature_selection_results(results_df, save_path=str(VIS_DIR / "feature_selection.png"))
    
    # Plot selected features distribution
    print("\nGenerating additional visualizations...")
    plot_selected_features_distribution(X_train, selected_features, save_path=str(VIS_DIR / "selected_features_distribution.png"))
    
    # Plot selected features importance boxplot
    plot_selected_features_importance_boxplot(importance_df, selected_features, save_path=str(VIS_DIR / "selected_importance_boxplot.png"))
    
    # Get selected feature data
    X_train_selected = X_train[selected_features].values
    X_test_selected = X_test[selected_features].values
    y_train_arr = y_train.values
    y_test_arr = y_test.values
    
    # Step 5: Train baseline model
    print("\nStep 5: Training baseline model...")
    baseline_cv_r2, _, _ = train_baseline_model_cv(X_train_selected, y_train_arr)
    
    # Step 6: Hyperparameter tuning
    print("\nStep 6: Hyperparameter tuning with Optuna...")
    best_params, tuned_cv_r2, tuned_cv_rmse, tuned_cv_mae = tune_hyperparameters(X_train_selected, y_train_arr)
    
    print(f"\nImprovement from tuning: {(tuned_cv_r2 - baseline_cv_r2)*100:.2f} percent")
    
    # Step 7: Train final model
    print("\nStep 7: Training final model...")
    final_model = train_final_model(X_train_selected, y_train_arr, best_params)
    
    # Step 8: Evaluate on test set (only once)
    print("\nStep 8: Evaluating on test set...")
    metrics = evaluate_model(
        final_model,
        X_train_selected, y_train_arr, 
        X_test_selected, y_test_arr
    )
    
    # Save prediction plots - Test set with three graphs in a row
    train_pred = final_model.predict(X_train_selected)
    test_pred = final_model.predict(X_test_selected)
    
    plot_test_predictions_with_residuals_and_errors(
        y_test_arr, test_pred,
        save_path=str(VIS_DIR / "predictions_vs_actual.png")
    )
    
    # Plot impurity vs permutation importance comparison
    plot_impurity_vs_permutation(final_model, importance_df, selected_features, save_path=str(VIS_DIR / "impurity_vs_permutation.png"))
    
    # Step 9: Generate predictions
    print("\nStep 9: Generating predictions...")
    predictions = generate_predictions(final_model, X_eval, selected_features)
    
    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - FINAL RESULTS")
    print("=" * 60)
    
    print(f"\nSelected Features: {len(selected_features)}")
    print(f"  Features: {', '.join(selected_features[:5])}...")
    
    print(f"\nModel Comparison:")
    print(f"  Baseline CV R2:      {baseline_cv_r2:.4f}")
    print(f"  Tuned CV R2:         {tuned_cv_r2:.4f}")
    print(f"  Improvement:         +{(tuned_cv_r2 - baseline_cv_r2)*100:.2f}%")
    
    print(f"\nFinal Model Performance:")
    print(f"  Train R2:   {metrics['train']['r2']:.4f}")
    print(f"  Train RMSE: {metrics['train']['rmse']:.4f}")
    print(f"  Train MAE:  {metrics['train']['mae']:.4f}")
    print(f"  -" * 20)
    print(f"  Val R2 (CV):   {tuned_cv_r2:.4f}")
    print(f"  Val RMSE (CV): {tuned_cv_rmse:.4f}")
    print(f"  Val MAE (CV):  {tuned_cv_mae:.4f}")
    print(f"  -" * 20)
    print(f"  Test R2:    {metrics['test']['r2']:.4f}")
    print(f"  Test RMSE:  {metrics['test']['rmse']:.4f}")
    print(f"  Test MAE:   {metrics['test']['mae']:.4f}")
    print(f"  -" * 20)
    print(f"  Overfitting (Train-Test): {metrics['overfitting']*100:.2f}%")
    
    print(f"\nPredictions:")
    print(f"  File: EVAL_target01_8.csv")
    print(f"  Samples: {len(predictions)}")
    print(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
