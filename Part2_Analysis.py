#!/usr/bin/env python
# coding: utf-8

# # Part 2: Rule Discovery for target02
# 

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

# Set random state
RANDOM_STATE = 42


# ---
# ## 1. Data Loading and Exploration

# In[5]:


# Load data
X = pd.read_csv('problem_8/dataset_8.csv')
y = pd.read_csv('problem_8/target_8.csv')['target02']

print(f"Features: {X.shape[1]} columns, {X.shape[0]} samples")
print(f"Target02 range: [{y.min():.4f}, {y.max():.4f}]")
print(f"Target02 mean: {y.mean():.4f}, std: {y.std():.4f}")


# In[6]:


# Target02 distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.3f}')
ax.set_xlabel('target02')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of target02')
ax.legend()
plt.tight_layout()
plt.savefig('target02_distribution.png')
plt.show()



# ## 2. Feature Selection: Identifying Key Features


# In[7]:


# Pearson correlation with target02
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("Top 10 Features by Correlation:")
print(correlations.head(10))


# In[8]:


# Mutual Information
mi_scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
mi_df = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Top 10 Features by Mutual Information:")
print(mi_df.head(10))


# In[9]:


# Decision Tree Importance
dt = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
dt.fit(X, y)
dt_imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 Features by Decision Tree Importance:")
print(dt_imp.head(10))


# In[10]:


# Combine top features from each method
top_corr = set(correlations.head(10).index)
top_mi = set(mi_df.head(10).index)
top_dt = set(dt_imp.head(10).index)

# Consensus Analysis
# Features appearing in correlation, MI, and DT
consensus_features = top_corr & top_mi & top_dt
print(f"Consensus Features (appear in all 3 methods): {consensus_features}")

# Key features manually selected based on this consensus
key_features = ['feat_31', 'feat_35', 'feat_45', 'feat_181']
print(f"\nKey Features Selected for Model: {key_features}")

# Save Feature Comparison Chart
comparison_df = pd.DataFrame({
    'Correlation': correlations[key_features],
    'Mutual_Info': mi_df[key_features],
    'Tree_Imp': dt_imp[key_features]
})
comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('Feature Selection Comparison')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('mutual_features_comparison.png')
plt.close()




# ### 2.5 Feature Visualization (Scatter Plots)
# 
# Visualizing the relationship between the 4 selected features and target02.

# In[11]:


# Visualization of Consensus Features
# Explicitly define key features for plotting
key_features = ['feat_31', 'feat_35', 'feat_45', 'feat_181']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    axes[i].scatter(X[feat], y, alpha=0.3, s=10)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('target02')
    axes[i].set_title(f'{feat} vs target02')

plt.tight_layout()
plt.savefig('key_features_scatter.png')
plt.show()


# ---
# ## 3. Initial Model Testing
# 
# Test different models with all features vs. selected features.

# In[12]:


# Strict Cross-Validation for Model Comparison
print("\n### Model Performance Comparison (5-Fold CV) ###")
print("| Model | Depth | Mean R² | Std Dev R² | Mean MSE | Std Dev MSE |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")

# Define models to test
models_to_test = [
    ("Linear Regression", LinearRegression(), "N/A"),
    ("Decision Tree", DecisionTreeRegressor(max_depth=2, random_state=RANDOM_STATE), 2),
    ("Decision Tree", DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE), 4),
    ("Decision Tree", DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE), 6),
    ("Decision Tree", DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE), 8),
    ("Decision Tree", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE), 10),
]

# Use the 4 key features
X_key = X[key_features]

for name, model, depth in models_to_test:
    # Cross-validation for R2
    r2_scores = cross_val_score(model, X_key, y, cv=5, scoring='r2')
    
    # Cross-validation for MSE (negative mse returned by sklearn)
    mse_scores = -cross_val_score(model, X_key, y, cv=5, scoring='neg_mean_squared_error')
    
    print(f"| {name} | {depth} | {r2_scores.mean():.4f} | {r2_scores.std():.4f} | {mse_scores.mean():.4f} | {mse_scores.std():.4f} |")


# In[13]:


# Simple Linear Regression Comparison
lr = LinearRegression()
lr.fit(X_key, y)
print(f"Linear Regression R²: {lr.score(X_key, y):.4f}")
print(f"Coefficients: {dict(zip(key_features, lr.coef_.round(4)))}")


# ---
# ## 4. Rule Extraction via Decision Tree
# 
# 

# In[14]:


# Visualize Simplified Decision Tree (Key Threshold Discovery)
from sklearn.tree import plot_tree

# Train a simplified tree (depth 3) to clearly visualize the top-level splits (feat_35)
model_simple = DecisionTreeRegressor(max_depth=3, random_state=42)
model_simple.fit(X[key_features], y)

plt.figure(figsize=(20, 10))
plot_tree(model_simple, 
          feature_names=key_features, 
          filled=True, 
          rounded=True, 
          fontsize=12,
          precision=2)
plt.title("Simplified Decision Tree Structure (Key Threshold Discovery)", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree_part2.png')
plt.show()


# ---
# ## 5. Piecewise Linear Regression Discovery
# 

# In[15]:


# Define regions
regions = {
    'R1': X['feat_35'] <= 0.2,
    'R2': (X['feat_35'] > 0.2) & (X['feat_35'] <= 0.5),
    'R3': (X['feat_35'] > 0.5) & (X['feat_35'] <= 0.7),
    'R4': X['feat_35'] > 0.7
}

print("Piecewise Linear Models:")
for name, mask in regions.items():
    if mask.sum() > 0:
        # Features for regression in this region (excluding feat_35 itself if it's just a switch, 
        # but we include all 3 others. We can include feat_35 too to check coefficient)
        X_reg = X.loc[mask, ['feat_31', 'feat_45', 'feat_181']]
        y_reg = y.loc[mask]
        
        reg = LinearRegression()
        reg.fit(X_reg, y_reg)
        r2 = reg.score(X_reg, y_reg)
        
        print(f"\nRegion {name} (N={mask.sum()}): R² = {r2:.6f}")
        coeffs = {k: float(f"{v:.2f}") for k, v in zip(X_reg.columns, reg.coef_)}
        intercept = float(f"{reg.intercept_:.2f}")
        print(f"Formula: {coeffs} + {intercept}")

