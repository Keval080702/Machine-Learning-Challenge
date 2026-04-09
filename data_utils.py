# Data loading and splitting module
# Handles loading CSV files and creating train test split

import pandas as pd
from sklearn.model_selection import train_test_split
from config import TRAIN_FEATURES_PATH, TRAIN_TARGETS_PATH, EVAL_FEATURES_PATH
from config import TEST_SIZE, RANDOM_STATE


def load_training_data():
    """
    Load training features and targets from CSV files.
    Returns X dataframe and y series.
    """
    X = pd.read_csv(TRAIN_FEATURES_PATH)
    target_df = pd.read_csv(TRAIN_TARGETS_PATH)
    y = target_df["target01"]
    
    print(f"Loaded training data: {len(X)} samples, {X.shape[1]} features")
    return X, y


def load_eval_data():
    """
    Load evaluation features from CSV file.
    Returns X_eval dataframe.
    """
    X_eval = pd.read_csv(EVAL_FEATURES_PATH)
    print(f"Loaded evaluation data: {len(X_eval)} samples")
    return X_eval


def split_data(X, y):
    """
    Split data into train and test sets.
    Uses 80 percent for training and 20 percent for testing.
    Random state is fixed for reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        shuffle=True
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


# Prediction module
# Generates predictions on evaluation dataset

import numpy as np
from config import OUTPUT_PATH


def generate_predictions(model, X_eval, selected_features):
    """
    Generate predictions for evaluation dataset.
    Uses the trained model directly (no scaling needed for CatBoost).
    Saves predictions to CSV file.
    """
    print("Generating predictions for evaluation dataset...")
    
    # Select same features as training
    X_eval_selected = X_eval[selected_features].values
    
    # Generate predictions directly (no scaling)
    predictions = model.predict(X_eval_selected)
    
    # Ensure predictions are positive
    predictions = np.clip(predictions, 0, None)
    
    # Save to CSV
    output_df = pd.DataFrame({"target01": predictions})
    output_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Predictions saved to: {OUTPUT_PATH}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    return predictions
