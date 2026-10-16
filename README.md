<div align="center">

# 🧠 Machine Learning Challenge — Problem 8

**High-dimensional regression and exact rule discovery on a 273-feature dataset**

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-FFCC00)
![Optuna](https://img.shields.io/badge/Optuna-500%20trials-2C5BB4)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Test R²](https://img.shields.io/badge/Part%201%20Test%20R²-0.952-brightgreen)
![Rules R²](https://img.shields.io/badge/Part%202%20R²-1.000-brightgreen)

*Ostbayerische Technische Hochschule Amberg-Weiden · Supervised by Prof. Dr. Patrick Levi*

</div>

---

## ✨ Highlights

| | Part 1 — `target01` | Part 2 — `target02` |
|---|---|---|
| **Task** | Predict a noisy continuous target | Recover the hidden rule that generates the target |
| **Approach** | CatBoost + CV permutation importance + Optuna | Consensus feature selection → decision-tree thresholds → piecewise linear fit |
| **Features used** | **9** of 273 | **4** of 273 |
| **Result** | **Test R² = 0.9519** (from 0.70 with all features) | **R² = 1.0000, MSE = 0.0000** |
| **Deployment** | Gradient-boosted model | 3 if-checks + 3 multiplications, **zero ML dependencies** |

---

## 📂 Repository Structure

```
Machine-Learning-Challenge/
├── main.py               # Part 1 — end-to-end pipeline entry point
├── config.py             # Paths, seeds, CV settings, hyperparameters
├── data_utils.py         # Data loading, train/test split, prediction export
├── modeling.py           # Feature selection, training, Optuna tuning, evaluation
├── visualization.py      # All Part 1 plots
├── EDA.py                # Exploratory data analysis (both targets)
├── Part2_Analysis.py     # Part 2 — rule discovery analysis
├── framework_8.py        # Part 2 — rule-based predictor (edge-device ready)
├── EVAL_target01_8.csv   # Final Part 1 predictions (10,000 rows)
├── Report.pdf            # Full project report
└── requirements.txt
```

The data is not included. Place it next to the code as:

```
problem_8/
├── dataset_8.csv   # Training features (10,000 × 273)
├── target_8.csv    # Training targets (target01, target02)
└── EVAL_8.csv      # Evaluation features (10,000 × 273)
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Part 1: full regression pipeline → EVAL_target01_8.csv + visualizations/
python main.py

# Part 2: predict target02 with the discovered rules
python framework_8.py --eval_file_path problem_8/EVAL_8.csv

# Optional analyses
python EDA.py
python Part2_Analysis.py
```

> All scripts are run from the repository root.

---

## 📊 Data Insights

- **Clean data:** no missing values, duplicates or infinite values.
- **Mostly noise:** many features have very low variance, and no single feature correlates strongly with `target01`.
- **Low redundancy:** features are weakly correlated with each other, so there is little multicollinearity.
- **Conclusion:** the signal lives in a *small subset* of features, which makes feature selection the key step.

---

## 🎯 Part 1 — `target01` Regression

### Pipeline

```mermaid
flowchart LR
    A[Load data] --> B[80/20 split]
    B --> C[5-fold CV<br/>permutation importance]
    C --> D[Pick best K<br/>by CV R²]
    D --> E[Baseline<br/>CatBoost CV]
    E --> F[Optuna<br/>500 trials]
    F --> G[Final model]
    G --> H[Test set<br/>evaluated once]
    H --> I[EVAL_target01_8.csv]
```

### 1. Why feature selection matters

Using all 273 features, every model overfits badly:

| Model | Train R² | Val R² | Overfit |
|---|---|---|---|
| HistGradientBoosting | 0.9150 | **0.6983** | 21.67% |
| CatBoost | 0.9042 | 0.6582 | 24.60% |
| XGBoost | 0.9858 | 0.5898 | 39.60% |
| Random Forest | 0.8756 | 0.0933 | 78.23% |

### 2. Permutation vs. impurity importance

Permutation importance computed with CatBoost gave the most stable rankings and the best validation scores:

| Method | Features | Val R² (CV) | Val RMSE | Overfit |
|---|---|---|---|---|
| **CatBoost + Permutation** | **9** | **89.93%** | **0.0755** | **5.09%** |
| CatBoost + MDI | 9 | 89.41% | 0.0774 | 5.16% |
| HistGB + Permutation | 8 | 85.27% | 0.0914 | 10.11% |
| Random Forest + Permutation | 6 | 53.01% | 0.1631 | 35.62% |
| Random Forest + MDI | 30 | 15.84% | 0.2184 | 66.79% |

### 3. Optimal feature count

Subsets from 5 to 273 features were compared with cross-validation. **9 features** gave the best result: fewer underfit, more added noise.

`feat_167` · `feat_225` · `feat_115` · `feat_31` · `feat_116` · `feat_90` · `feat_0` · `feat_83` · `feat_220`

### 4. Final results

After 500 Optuna trials, CV R² improved by **4.70%** over the baseline:

| Split | R² | RMSE | MAE |
|---|---|---|---|
| Training | 0.9912 | 0.0224 | 0.0169 |
| Validation (CV) | 0.9483 | 0.0541 | 0.0412 |
| **Test** | **0.9519** | **0.0532** | **0.0404** |

**Overfitting gap: 3.93%**

### 🛡️ Data leakage prevention

- ✅ The test set is never used for feature selection, tuning or training.
- ✅ Feature importance is computed only on validation folds of the training data.
- ✅ Any scaling is fitted inside each CV fold.
- ✅ The test set is evaluated exactly **once**, after the configuration is fixed.

---

## 🔍 Part 2 — `target02` Rule Discovery

### The clue

`target02` did not look like real-world data:

- Multi-modal, tightly clustered values with sharp gaps between them
- 269 of 273 features had near-zero correlation with it
- A global linear regression scored only **R² ≈ 0.10**

All of this pointed to a **deterministic, rule-based generator**.

### Finding the rule

1. **Consensus feature selection** with Pearson correlation, mutual information and decision-tree importance. All three agreed on `feat_31`, `feat_35`, `feat_45` and `feat_181`.
2. **Decision trees** converged slowly towards a perfect fit, which suggests sloped linear segments rather than steps:

   | Model | Depth | Mean R² (5-fold) |
   |---|---|---|
   | Linear Regression | — | 0.0994 |
   | Decision Tree | 2 | 0.7217 |
   | Decision Tree | 6 | 0.9505 |
   | Decision Tree | 10 | 0.9890 |

3. **The tree splits** revealed that `feat_35` is a master switch with thresholds at **0.2, 0.5 and 0.7**.
4. **A linear regression inside each region** fits perfectly.

### The discovered rules

```python
if   feat_35 <= 0.2:  target02 = -1.25*feat_31 + 0.65*feat_45 - 1.05*feat_181
elif feat_35 <= 0.5:  target02 =  1.65*feat_31 + 1.75*feat_45 + 1.65*feat_181
elif feat_35 <= 0.7:  target02 =  0.95*feat_31 + 1.75*feat_45 + 0.45*feat_181
else:                 target02 =  1.65*feat_31 - 1.25*feat_45 + 1.15*feat_181
```

| Regime | `feat_35` | Behaviour |
|---|---|---|
| 🔵 Low | ≤ 0.2 | Damped: `feat_31` and `feat_181` push the target down |
| 🟢 Lower-mid | 0.2 – 0.5 | Strong response: every feature contributes positively |
| 🟡 Upper-mid | 0.5 – 0.7 | Transition: `feat_31` weakens and `feat_181` barely matters |
| 🔴 High | > 0.7 | Regulated: `feat_45` turns negative and `feat_31` dominates |

### ⚡ Built for edge devices

| | Decision Tree (depth 10) | **Discovered rules** |
|---|---|---|
| R² | 0.9890 | **1.0000** |
| Ops per prediction | up to 10 node traversals | **≤ 3 comparisons + 5 arithmetic ops** |
| Memory | full tree structure | **12 coefficients** |
| Dependencies | scikit-learn | **none** |

---

## 📈 Generated Outputs

| Script | Outputs |
|---|---|
| `main.py` | `EVAL_target01_8.csv` and, in `visualizations/`: `feature_importance.png`, `feature_selection.png`, `selected_features_distribution.png`, `selected_importance_boxplot.png`, `predictions_vs_actual.png`, `impurity_vs_permutation.png` |
| `Part2_Analysis.py` | `target02_distribution.png`, `mutual_features_comparison.png`, `key_features_scatter.png`, `decision_tree_part2.png` |
| `EDA.py` | `eda_*.png` (distributions, correlations, outliers, skewness, train vs. eval) |

---

## 🧰 Tech Stack

| Purpose | Libraries |
|---|---|
| Data | `numpy`, `pandas`, `scipy` |
| Modelling | `catboost`, `scikit-learn` |
| Tuning | `optuna` |
| Visualisation | `matplotlib`, `seaborn` |

---

## 📄 Report

The complete methodology, figures and discussion are in **[Report.pdf](Report.pdf)**.

<div align="center">

**Keval Dineshbhai Viradiya** · OTH Amberg-Weiden · Department of Electrical Engineering, Media and Computer Science

</div>
