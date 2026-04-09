import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition
        - condition: must be a tupe of (int, string, float). The second entry must be a string from the list below, describing the operator. Third entry of the tuple must be a float). If condition is None, it is always evaluated to true.
        - arr: array on which the condition is evaluated

    The python operator package is used. Second entry in condition must be one of those:
       ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    """
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


def main(args):
    """
    Predict target02 using discovered piecewise linear formula.
    
    The formula depends on feat_35 threshold and uses linear combinations
    of feat_31, feat_45, and feat_181.
    
    Feature indices:
        feat_31 -> index 31
        feat_35 -> index 35
        feat_45 -> index 45
        feat_181 -> index 181
    
    Rules (discovered via analysis, achieves R² = 1.0):
        if feat_35 <= 0.2:
            target02 = -1.25*feat_31 + 0.65*feat_45 - 1.05*feat_181
        elif feat_35 <= 0.5:
            target02 = 1.65*feat_31 + 1.75*feat_45 + 1.65*feat_181
        elif feat_35 <= 0.7:
            target02 = 0.95*feat_31 + 1.75*feat_45 + 0.45*feat_181
        else:
            target02 = 1.65*feat_31 - 1.25*feat_45 + 1.15*feat_181
    """
    
    # Define calculation functions for each region
    def calc_region1(arr):
        """Region 1: feat_35 <= 0.2"""
        return -1.25 * arr[31] + 0.65 * arr[45] - 1.05 * arr[181]
    
    def calc_region2(arr):
        """Region 2: 0.2 < feat_35 <= 0.5"""
        return 1.65 * arr[31] + 1.75 * arr[45] + 1.65 * arr[181]
    
    def calc_region3(arr):
        """Region 3: 0.5 < feat_35 <= 0.7"""
        return 0.95 * arr[31] + 1.75 * arr[45] + 0.45 * arr[181]
    
    def calc_region4(arr):
        """Region 4: feat_35 > 0.7"""
        return 1.65 * arr[31] - 1.25 * arr[45] + 1.15 * arr[181]
    
    # Define conditions based on feat_35 (index 35)
    # Order matters - first matching condition is executed
    condition1 = (35, "<=", 0.2)    # Region 1: feat_35 <= 0.2
    condition2 = (35, "<=", 0.5)    # Region 2: 0.2 < feat_35 <= 0.5
    condition3 = (35, "<=", 0.7)    # Region 3: 0.5 < feat_35 <= 0.7
    condition4 = None               # Region 4: else (feat_35 > 0.7)
    
    # Create pair list - order matters!
    pair_list = [
        (condition1, calc_region1),
        (condition2, calc_region2),
        (condition3, calc_region3),
        (condition4, calc_region4)
    ]
    
    # Load evaluation data
    data_array = pd.read_csv(args.eval_file_path).values
    
    # Generate predictions
    return framework(pair_list, data_array)

    
def main_example(args):

    # Example: 
    test_arr = np.ones((10,10))

    def calc1(arr):
        """square first array column"""
        return arr[0]**2

    def calc2(arr):
        """add columns 3 and 4"""
        return arr[2] + arr[3]

    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)

    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print (predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)
    print(f"Generated {len(target02)} predictions")
