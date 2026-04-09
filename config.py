# Configuration file for Part 1 ML Challenge
# Contains all constants, paths, and hyperparameters

from pathlib import Path

# Random seed for reproducibility
RANDOM_STATE = 42

# Data paths
DATA_DIR = Path(__file__).parent / "problem_8"
TRAIN_FEATURES_PATH = DATA_DIR / "dataset_8.csv"
TRAIN_TARGETS_PATH = DATA_DIR / "target_8.csv"
EVAL_FEATURES_PATH = DATA_DIR / "EVAL_8.csv"
OUTPUT_PATH = Path(__file__).parent / "EVAL_target01_8.csv"

# Visualization output directory
VIS_DIR = Path(__file__).parent / "visualizations"

# Data split ratio
TEST_SIZE = 0.20

# Cross validation settings
N_FOLDS = 5
N_REPEATS = 10

# Baseline model parameters
BASELINE_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "eval_metric": "R2",
    "random_state": RANDOM_STATE,
    "verbose": 0,
    "allow_writing_files": False
}

# Optuna tuning settings
N_TRIALS = 500

# Best hyperparameters found by Optuna
# These are updated after tuning completes
BEST_PARAMS = {
    "iterations": 444,
    "learning_rate": 0.116062,
    "depth": 6,
    "l2_leaf_reg": 9.240444,
    "random_strength": 0.586404,
    "bagging_temperature": 0.065583,
    "border_count": 47,
    "eval_metric": "R2",
    "random_state": RANDOM_STATE,
    "verbose": 0,
    "allow_writing_files": False
}
