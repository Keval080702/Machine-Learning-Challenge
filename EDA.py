

# # Exploratory Data Analysis (EDA)

# 
# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

RANDOM_STATE = 42


# ---
# # 1. Data Loading and Overview

# In[2]:


# Load datasets
X = pd.read_csv('problem_8/dataset_8.csv')
y = pd.read_csv('problem_8/target_8.csv')
X_eval = pd.read_csv('problem_8/EVAL_8.csv')

print('DATASET OVERVIEW')
print('='*60)
print(f'Training Features: {X.shape[0]} samples, {X.shape[1]} features')
print(f'Targets: {y.shape[0]} samples, {y.shape[1]} targets')
print(f'Evaluation Features: {X_eval.shape[0]} samples, {X_eval.shape[1]} features')
print(f'\nTarget columns: {y.columns.tolist()}')


# In[3]:


# First few rows
print('Feature Data Sample:')
print(X.head())

print('\nTarget Data Sample:')
print(y.head())


# ---
# # 2. Data Quality Analysis

# In[4]:


# Missing values analysis
print('MISSING VALUES ANALYSIS')
print('='*60)

missing_X = X.isnull().sum().sum()
missing_y = y.isnull().sum().sum()
missing_eval = X_eval.isnull().sum().sum()

print(f'Missing values in features (X): {missing_X}')
print(f'Missing values in targets (y): {missing_y}')
print(f'Missing values in eval (X_eval): {missing_eval}')
print(f'\n No missing values in any dataset!' if missing_X + missing_y + missing_eval == 0 else ' Missing values detected!')


# In[5]:


# Data types
print('DATA TYPES')
print('='*60)
print(f'Feature data types: {X.dtypes.value_counts().to_dict()}')
print(f'Target data types: {y.dtypes.to_dict()}')

# Check for duplicates
print(f'\nDuplicate rows in features: {X.duplicated().sum()}')
print(f'Duplicate rows in targets: {y.duplicated().sum()}')


# ---
# # 3. Feature Statistics Summary

# In[6]:


# Feature statistics
print('FEATURE STATISTICS')
print('='*60)

feature_stats = X.describe().T
feature_stats['range'] = feature_stats['max'] - feature_stats['min']
feature_stats['cv'] = feature_stats['std'] / feature_stats['mean'] # Coefficient of variation

print(f'\nFeature value ranges:')
print(f' Min of all features: {X.min().min():.4f}')
print(f' Max of all features: {X.max().max():.4f}')
print(f' Mean of means: {X.mean().mean():.4f}')
print(f' Mean of stds: {X.std().mean():.4f}')

# Show summary table
print(feature_stats.head(10))


# In[7]:


# Feature distribution overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of feature means
axes[0, 0].hist(X.mean(), bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Mean Value')
axes[0, 0].set_ylabel('Number of Features')
axes[0, 0].set_title('Distribution of Feature Means')

# Distribution of feature stds
axes[0, 1].hist(X.std(), bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Standard Deviation')
axes[0, 1].set_ylabel('Number of Features')
axes[0, 1].set_title('Distribution of Feature Std Devs')

# Distribution of feature mins
axes[1, 0].hist(X.min(), bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('Min Value')
axes[1, 0].set_ylabel('Number of Features')
axes[1, 0].set_title('Distribution of Feature Minimums')

# Distribution of feature maxs
axes[1, 1].hist(X.max(), bins=30, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_xlabel('Max Value')
axes[1, 1].set_ylabel('Number of Features')
axes[1, 1].set_title('Distribution of Feature Maximums')

plt.tight_layout()
plt.savefig('eda_feature_overview.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 4. Target Variables Analysis

# In[8]:


# Target statistics
print('TARGET STATISTICS')
print('='*60)
print(y.describe())


# In[24]:


# Target distributions (side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target01 histogram
axes[0].hist(y['target01'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(y['target01'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y["target01"].mean():.3f}')
axes[0].axvline(y['target01'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {y["target01"].median():.3f}')
axes[0].set_xlabel('target01')
axes[0].set_ylabel('Frequency')
axes[0].set_title('target01 Distribution')
axes[0].legend()

# Target02 histogram
axes[1].hist(y['target02'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(y['target02'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y["target02"].mean():.3f}')
axes[1].axvline(y['target02'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {y["target02"].median():.3f}')
axes[1].set_xlabel('target02')
axes[1].set_ylabel('Frequency')
axes[1].set_title('target02 Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_target_distributions.png', dpi=150, bbox_inches='tight')
plt.show()


# In[10]:


# Relationship between targets
print('TARGET RELATIONSHIP')
print('='*60)
print(f'Correlation between target01 and target02: {y["target01"].corr(y["target02"]):.4f}')

plt.figure(figsize=(8, 6))
plt.scatter(y['target01'], y['target02'], alpha=0.3, s=5)
plt.xlabel('target01')
plt.ylabel('target02')
plt.title('target01 vs target02')
plt.grid(True, alpha=0.3)
plt.savefig('eda_target_relationship.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 5. Feature Correlation Analysis

# In[11]:


# Correlation with target01
corr_target01 = X.corrwith(y['target01']).abs().sort_values(ascending=False)

print('TOP FEATURES CORRELATED WITH target01')
print('='*60)
print(corr_target01.head(15))


# In[12]:


# Correlation with target02
corr_target02 = X.corrwith(y['target02']).abs().sort_values(ascending=False)

print('TOP FEATURES CORRELATED WITH target02')
print('='*60)
print(corr_target02.head(15))


# In[13]:


# Visualize top correlations for both targets
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Target01 correlations
top_t01 = corr_target01.head(20)
axes[0].barh(range(len(top_t01)), top_t01.values, color='steelblue', alpha=0.7)
axes[0].set_yticks(range(len(top_t01)))
axes[0].set_yticklabels(top_t01.index)
axes[0].invert_yaxis()
axes[0].set_xlabel('Absolute Correlation')
axes[0].set_title('Top 20 Features Correlated with target01')

# Target02 correlations
top_t02 = corr_target02.head(20)
axes[1].barh(range(len(top_t02)), top_t02.values, color='coral', alpha=0.7)
axes[1].set_yticks(range(len(top_t02)))
axes[1].set_yticklabels(top_t02.index)
axes[1].invert_yaxis()
axes[1].set_xlabel('Absolute Correlation')
axes[1].set_title('Top 20 Features Correlated with target02')

plt.tight_layout()
plt.savefig('eda_feature_correlations.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 6. Top Features Distribution Analysis

# In[14]:


# Get top features for each target
top_features_t01 = corr_target01.head(9).index.tolist()
top_features_t02 = corr_target02.head(9).index.tolist()

print('Top 9 features for target01:', top_features_t01)
print('Top 9 features for target02:', top_features_t02)


# In[15]:


# Distribution of top features for target01
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, feat in enumerate(top_features_t01):
 ax = axes[i//3, i%3]
 ax.hist(X[feat], bins=40, edgecolor='black', alpha=0.7)
 ax.set_xlabel(feat)
 ax.set_ylabel('Frequency')
 ax.set_title(f'{feat} (corr: {corr_target01[feat]:.3f})')

plt.suptitle('Top 9 Features for target01', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_top_features_target01.png', dpi=150, bbox_inches='tight')
plt.show()


# In[16]:


# Distribution of top features for target02
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, feat in enumerate(top_features_t02):
 ax = axes[i//3, i%3]
 ax.hist(X[feat], bins=40, edgecolor='black', alpha=0.7, color='orange')
 ax.set_xlabel(feat)
 ax.set_ylabel('Frequency')
 ax.set_title(f'{feat} (corr: {corr_target02[feat]:.3f})')

plt.suptitle('Top 9 Features for target02', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_top_features_target02.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 7. Feature-Target Scatter Plots

# In[17]:


# Top 4 features vs target01
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, feat in enumerate(top_features_t01[:4]):
 ax = axes[i//2, i%2]
 ax.scatter(X[feat], y['target01'], alpha=0.3, s=5)
 ax.set_xlabel(feat)
 ax.set_ylabel('target01')
 ax.set_title(f'{feat} vs target01 (corr: {corr_target01[feat]:.3f})')
 ax.grid(True, alpha=0.3)

plt.suptitle('Top Features vs target01', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_scatter_target01.png', dpi=150, bbox_inches='tight')
plt.show()


# In[18]:


# Top 4 features vs target02
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, feat in enumerate(top_features_t02[:4]):
 ax = axes[i//2, i%2]
 ax.scatter(X[feat], y['target02'], alpha=0.3, s=5, color='coral')
 ax.set_xlabel(feat)
 ax.set_ylabel('target02')
 ax.set_title(f'{feat} vs target02 (corr: {corr_target02[feat]:.3f})')
 ax.grid(True, alpha=0.3)

plt.suptitle('Top Features vs target02', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_scatter_target02.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 8. Feature Correlation Heatmap

# In[19]:


# Correlation matrix for top features
all_top_features = list(set(top_features_t01[:6] + top_features_t02[:6]))

plt.figure(figsize=(12, 10))
corr_matrix = X[all_top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
 square=True, linewidths=0.5, center=0)
plt.title('Correlation Heatmap of Top Features')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 9. Outlier Analysis

# In[20]:


# Outlier detection using IQR method
def count_outliers(series):
 Q1 = series.quantile(0.25)
 Q3 = series.quantile(0.75)
 IQR = Q3 - Q1
 lower = Q1 - 1.5 * IQR
 upper = Q3 + 1.5 * IQR
 return ((series < lower) | (series > upper)).sum()

# Count outliers per feature
outlier_counts = X.apply(count_outliers)
outlier_pct = (outlier_counts / len(X) * 100).sort_values(ascending=False)

print('OUTLIER ANALYSIS (IQR Method)')
print('='*60)
print(f'Features with most outliers (% of data):')
print(outlier_pct.head(10))

# Visualize
plt.figure(figsize=(12, 6))
plt.bar(range(len(outlier_pct.head(20))), outlier_pct.head(20).values, color='coral', alpha=0.7)
plt.xticks(range(len(outlier_pct.head(20))), outlier_pct.head(20).index, rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Outlier Percentage (%)')
plt.title('Top 20 Features by Outlier Percentage')
plt.tight_layout()
plt.savefig('eda_outlier_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 10. Skewness and Kurtosis Analysis

# In[21]:


# Calculate skewness and kurtosis
skewness = X.skew().sort_values()
kurtosis = X.kurtosis().sort_values()

print('SKEWNESS AND KURTOSIS')
print('='*60)
print(f'\nMost negatively skewed features:')
print(skewness.head())
print(f'\nMost positively skewed features:')
print(skewness.tail())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(skewness, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Skewness')
axes[0].set_ylabel('Number of Features')
axes[0].set_title('Distribution of Feature Skewness')

axes[1].hist(kurtosis, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Kurtosis')
axes[1].set_ylabel('Number of Features')
axes[1].set_title('Distribution of Feature Kurtosis')

plt.tight_layout()
plt.savefig('eda_skewness_kurtosis.png', dpi=150, bbox_inches='tight')
plt.show()


# ---
# # 11. Train vs Evaluation Data Comparison

# In[22]:


# Compare training and evaluation data distributions
print('TRAIN vs EVALUATION DATA')
print('='*60)

train_means = X.mean()
eval_means = X_eval.mean()

train_stds = X.std()
eval_stds = X_eval.std()

mean_diff = (train_means - eval_means).abs()
std_diff = (train_stds - eval_stds).abs()

print(f'\nFeatures with largest mean differences:')
print(mean_diff.sort_values(ascending=False).head(10))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(train_means, eval_means, alpha=0.5, s=10)
axes[0].plot([train_means.min(), train_means.max()], [train_means.min(), train_means.max()], 'r--')
axes[0].set_xlabel('Train Mean')
axes[0].set_ylabel('Eval Mean')
axes[0].set_title('Feature Means: Train vs Eval')

axes[1].scatter(train_stds, eval_stds, alpha=0.5, s=10, color='orange')
axes[1].plot([train_stds.min(), train_stds.max()], [train_stds.min(), train_stds.max()], 'r--')
axes[1].set_xlabel('Train Std')
axes[1].set_ylabel('Eval Std')
axes[1].set_title('Feature Stds: Train vs Eval')

plt.tight_layout()
plt.savefig('eda_train_vs_eval.png', dpi=150, bbox_inches='tight')
plt.show()


