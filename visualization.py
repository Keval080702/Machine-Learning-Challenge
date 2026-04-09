# Visualization module
# Creates plots for feature importance, training loss, and predictions

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(importance_df, top_n=20, selected_features=None, X_train=None, save_path=None):
    """
    Plot top N features by importance score with histogram and distribution.
    Green bars indicate high importance, orange medium, gray low.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    top_features = importance_df.head(top_n)
    
    # Plot 1: Bar chart of importance
    ax1 = axes[0, 0]
    colors = []
    for imp in top_features["importance"]:
        if imp > 0.01:
            colors.append("green")
        elif imp > 0.001:
            colors.append("orange")
        else:
            colors.append("gray")
    
    y_pos = range(len(top_features))
    ax1.barh(y_pos, top_features["importance"], color=colors, alpha=0.7, edgecolor="black")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features["feature"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Permutation Importance")
    ax1.set_title(f"Top {top_n} Features by CV Based Permutation Importance")
    ax1.grid(axis="x", alpha=0.3)
    
    # Plot 2: Histogram of all importance scores
    ax2 = axes[0, 1]
    ax2.hist(importance_df["importance"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax2.axvline(x=importance_df["importance"].median(), color="red", linestyle="--", label=f"Median: {importance_df['importance'].median():.4f}")
    ax2.set_xlabel("Importance Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of All Feature Importances")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Top 10 with error bars (box plot style)
    ax3 = axes[1, 0]
    top_10 = importance_df.head(10)
    ax3.barh(range(10), top_10["importance"], xerr=top_10["std"], color="steelblue", alpha=0.7, capsize=3)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels(top_10["feature"])
    ax3.invert_yaxis()
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("Decrease in Accuracy Score")
    ax3.set_title("Permutation Importance with Std (Top 10)")
    ax3.grid(axis="x", alpha=0.3)
    
    # Plot 4: Cumulative importance
    ax4 = axes[1, 1]
    sorted_imp = importance_df["importance"].sort_values(ascending=False).values
    cumsum = np.cumsum(sorted_imp) / np.sum(sorted_imp) * 100
    ax4.plot(range(1, len(cumsum)+1), cumsum, "o-", markersize=2, color="green")
    ax4.axhline(y=90, color="red", linestyle="--", label="90% threshold")
    ax4.set_xlabel("Number of Features")
    ax4.set_ylabel("Cumulative Importance (%)")
    ax4.set_title("Cumulative Feature Importance")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_permutation_importance_boxplot(importance_data, feature_names, save_path=None):
    """
    Plot permutation importance as box plots showing distribution across CV folds.
    importance_data: list of arrays, one per fold
    feature_names: list of feature names
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Transpose to get importance per feature across folds
    importance_array = np.array(importance_data).T
    
    # Sort by median importance
    medians = np.median(importance_array, axis=1)
    sorted_idx = np.argsort(medians)[::-1][:15]  # Top 15
    
    sorted_data = importance_array[sorted_idx].T
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    bp = ax.boxplot(sorted_data, vert=False, patch_artist=True)
    
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Decrease in Accuracy Score")
    ax.set_title("Permutation Importances (CV Folds)")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_impurity_vs_permutation(model, importance_df, feature_names, save_path=None):
    """
    Compare impurity-based (Gini) importance vs permutation importance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get impurity importance from model (CatBoost feature importance)
    try:
        impurity_importance = model.get_feature_importance()
    except:
        impurity_importance = np.zeros(len(feature_names))
    
    # Create dataframe for comparison
    import pandas as pd
    compare_df = pd.DataFrame({
        "feature": feature_names,
        "impurity": impurity_importance,
        "permutation": importance_df.set_index("feature").loc[feature_names]["importance"].values
    }).sort_values("impurity", ascending=False)
    
    top_30 = compare_df.head(30)
    
    # Plot 1: Impurity importance
    ax1 = axes[0]
    ax1.barh(range(len(top_30)), top_30["impurity"], color="steelblue", alpha=0.7)
    ax1.set_yticks(range(len(top_30)))
    ax1.set_yticklabels(top_30["feature"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Gini Importance")
    ax1.set_title("Impurity-based Feature Importance")
    ax1.grid(axis="x", alpha=0.3)
    
    # Plot 2: Permutation importance with boxplot style
    ax2 = axes[1]
    perm_sorted = top_30.sort_values("permutation", ascending=False)
    
    # Use error bars to simulate box plot effect
    perm_vals = perm_sorted["permutation"].values
    perm_std = importance_df.set_index("feature").loc[perm_sorted["feature"]]["std"].values
    
    ax2.barh(range(len(perm_sorted)), perm_vals, xerr=perm_std, color="steelblue", alpha=0.7, capsize=2)
    ax2.set_yticks(range(len(perm_sorted)))
    ax2.set_yticklabels(perm_sorted["feature"])
    ax2.invert_yaxis()
    ax2.axvline(x=0, color="orange", linestyle="--", linewidth=2)
    ax2.set_xlabel("Decrease in Accuracy Score")
    ax2.set_title("Permutation Importance")
    ax2.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_selected_features_importance_boxplot(importance_df, selected_features, fold_importances=None, save_path=None):
    """
    Plot permutation importance box plots for selected subset of features.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for selected features
    selected_df = importance_df[importance_df["feature"].isin(selected_features)]
    selected_df = selected_df.sort_values("importance", ascending=False)
    
    # Create box plot data
    n_features = len(selected_df)
    positions = range(n_features)
    
    # Plot horizontal bars with error bars (simulating box plots)
    ax.barh(positions, selected_df["importance"], xerr=selected_df["std"], 
            color="white", edgecolor="black", linewidth=1.5, capsize=5, error_kw={"capthick": 1.5})
    
    # Add median line (orange)
    for i, (_, row) in enumerate(selected_df.iterrows()):
        ax.plot([row["importance"], row["importance"]], [i-0.3, i+0.3], color="orange", linewidth=2)
    
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(selected_df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Decrease in Accuracy Score")
    ax.set_title(f"Permutation Importances on Selected Subset ({len(selected_features)} features)")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_selected_features_distribution(X_train, selected_features, save_path=None):
    """
    Plot distribution (histogram and scatter) of top selected features.
    """
    n_features = min(len(selected_features), 9)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feat in enumerate(selected_features[:n_features]):
        ax = axes[i]
        ax.hist(X_train[feat], bins=50, edgecolor="black", alpha=0.7, color="teal")
        ax.axvline(x=X_train[feat].mean(), color="red", linestyle="--", label=f"Mean: {X_train[feat].mean():.3f}")
        ax.axvline(x=X_train[feat].median(), color="orange", linestyle="--", label=f"Median: {X_train[feat].median():.3f}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {feat}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_scaling_effect(X_train, selected_features, scaler, save_path=None):
    """
    Plot before and after scaling effect for selected features.
    Shows scatter plots and distributions.
    """
    n_features = min(len(selected_features), 4)
    
    fig, axes = plt.subplots(n_features, 2, figsize=(14, 4*n_features))
    
    # Get scaled data
    X_selected = X_train[selected_features[:n_features]]
    X_scaled = scaler.fit_transform(X_selected)
    
    for i, feat in enumerate(selected_features[:n_features]):
        # Before scaling
        ax1 = axes[i, 0] if n_features > 1 else axes[0]
        ax1.hist(X_selected[feat], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
        ax1.axvline(x=X_selected[feat].mean(), color="red", linestyle="--", linewidth=2)
        ax1.set_xlabel(feat)
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Before Scaling: {feat}")
        ax1.grid(True, alpha=0.3)
        
        # After scaling
        ax2 = axes[i, 1] if n_features > 1 else axes[1]
        ax2.hist(X_scaled[:, i], bins=50, edgecolor="black", alpha=0.7, color="teal")
        ax2.axvline(x=np.mean(X_scaled[:, i]), color="red", linestyle="--", linewidth=2)
        ax2.set_xlabel(f"{feat} (scaled)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"After Scaling: {feat}")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_feature_selection_results(results_df, save_path=None):
    """
    Plot validation R2 and overfitting for different feature counts.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Validation R2
    ax1 = axes[0, 0]
    ax1.plot(results_df["k"], results_df["val_r2_mean"], "o-", linewidth=2, markersize=8, color="blue", label="Val R2")
    ax1.plot(results_df["k"], results_df["train_r2"], "s-", linewidth=2, markersize=6, color="green", alpha=0.7, label="Train R2")
    ax1.set_xlabel("Number of Features")
    ax1.set_ylabel("R2 Score")
    ax1.set_title("Train vs Val R2 vs Number of Features")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overfitting
    ax2 = axes[0, 1]
    ax2.plot(results_df["k"], results_df["overfitting"] * 100, "o-", linewidth=2, markersize=8, color="red")
    ax2.set_xlabel("Number of Features")
    ax2.set_ylabel("Overfitting Percent")
    ax2.set_title("Overfitting vs Number of Features")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RMSE
    ax3 = axes[1, 0]
    ax3.plot(results_df["k"], results_df["val_rmse"], "o-", linewidth=2, markersize=8, color="purple")
    ax3.set_xlabel("Number of Features")
    ax3.set_ylabel("Validation RMSE")
    ax3.set_title("Validation RMSE vs Number of Features")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: MAE
    ax4 = axes[1, 1]
    ax4.plot(results_df["k"], results_df["val_mae"], "o-", linewidth=2, markersize=8, color="orange")
    ax4.set_xlabel("Number of Features")
    ax4.set_ylabel("Validation MAE")
    ax4.set_title("Validation MAE vs Number of Features")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_predictions_vs_actual_train_test(y_train, y_train_pred, y_test, y_test_pred, save_path=None):
    """
    Side by side scatter plots of predictions versus actual values for train and test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set
    ax1 = axes[0]
    ax1.scatter(y_train, y_train_pred, alpha=0.5, edgecolors="none", color="blue")
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title("Training Set: Predictions vs Actual")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test set
    ax2 = axes[1]
    ax2.scatter(y_test, y_test_pred, alpha=0.5, edgecolors="none", color="green")
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title("Test Set: Predictions vs Actual")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_test_predictions_with_residuals_and_errors(y_test, y_test_pred, save_path=None):
    """
    Plot test set predictions with three graphs in a row:
    1. Predictions vs Actual (scatter plot)
    2. Residual Plot (residuals vs predictions)
    3. Error Distribution (histogram of residuals)
    
    Args:
        y_test: True test values
        y_test_pred: Predicted test values
        save_path: Path to save the figure
    """
    # Calculate residuals
    test_residuals = y_test - y_test_pred
    
    # Create figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Graph 1: Predictions vs Actual
    ax1 = axes[0]
    ax1.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="none", color="green", s=30)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax1.set_xlabel("Actual Values", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Values", fontsize=11, fontweight="bold")
    ax1.set_title("Test Set: Predictions vs Actual", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Graph 2: Residual Plot
    ax2 = axes[1]
    ax2.scatter(y_test_pred, test_residuals, alpha=0.6, edgecolors="none", color="green", s=30)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Zero Line")
    ax2.set_xlabel("Predicted Values", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Residuals", fontsize=11, fontweight="bold")
    ax2.set_title("Test Set: Residual Plot", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    # Graph 3: Error Distribution
    ax3 = axes[2]
    ax3.hist(test_residuals, bins=50, edgecolor="black", alpha=0.7, color="green")
    ax3.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
    ax3.set_xlabel("Residual", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax3.set_title("Test Set: Error Distribution", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_residuals_train_test(y_train, y_train_pred, y_test, y_test_pred, save_path=None):
    """
    Plot residuals distribution and residuals vs predictions for both train and test.
    """
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Train: Histogram of residuals
    ax1 = axes[0, 0]
    ax1.hist(train_residuals, bins=50, edgecolor="black", alpha=0.7, color="blue")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Training Set: Residuals Distribution")
    
    # Test: Histogram of residuals
    ax2 = axes[0, 1]
    ax2.hist(test_residuals, bins=50, edgecolor="black", alpha=0.7, color="green")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Test Set: Residuals Distribution")
    
    # Train: Residuals vs predictions
    ax3 = axes[1, 0]
    ax3.scatter(y_train_pred, train_residuals, alpha=0.5, edgecolors="none", color="blue")
    ax3.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax3.set_xlabel("Predicted Values")
    ax3.set_ylabel("Residuals")
    ax3.set_title("Training Set: Residuals vs Predictions")
    
    # Test: Residuals vs predictions
    ax4 = axes[1, 1]
    ax4.scatter(y_test_pred, test_residuals, alpha=0.5, edgecolors="none", color="green")
    ax4.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax4.set_xlabel("Predicted Values")
    ax4.set_ylabel("Residuals")
    ax4.set_title("Test Set: Residuals vs Predictions")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_training_loss_curves(train_rmse_history, val_rmse_history, train_mae_history=None, val_mae_history=None, save_path=None):
    """
    Plot training loss curves for RMSE and MAE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE curve
    ax1 = axes[0]
    ax1.plot(train_rmse_history, label="Train RMSE", color="blue", linewidth=2)
    ax1.plot(val_rmse_history, label="Val RMSE", color="orange", linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Training Loss Curve - RMSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE curve
    ax2 = axes[1]
    if train_mae_history is not None and val_mae_history is not None:
        ax2.plot(train_mae_history, label="Train MAE", color="blue", linewidth=2)
        ax2.plot(val_mae_history, label="Val MAE", color="orange", linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("MAE")
        ax2.set_title("Training Loss Curve - MAE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.plot(1 - np.array(train_rmse_history)**2, label="Train R2 proxy", color="blue", linewidth=2)
        ax2.plot(1 - np.array(val_rmse_history)**2, label="Val R2 proxy", color="orange", linewidth=2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("R2 (proxy)")
        ax2.set_title("Training Progress - R2 Proxy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


# Keep old functions for backward compatibility
def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """
    Scatter plot of predictions versus actual values.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plot residuals distribution and residuals vs predictions.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Residual")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Residuals Distribution")
    
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors="none")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predictions")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.close()
