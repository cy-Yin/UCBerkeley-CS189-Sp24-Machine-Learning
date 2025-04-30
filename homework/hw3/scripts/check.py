"""
    Checks the validity of Kaggle submissions
    for homework 1.
    
    Usage:
        python3 check.py spam <my_spam_submission.csv>
        or 
        python check.py mnist <my_mnist_submission.csv>
"""

import argparse
import pandas as pd


def create_parser():
    parser = argparse.ArgumentParser(description='Check the validity of Kaggle submissions for homework 1.')
    parser.add_argument('dataset', type=str, help='The dataset to check. Options: "spam" or "mnist".')
    parser.add_argument('submission_file', type=str, help='The path to the submission file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    submission_data = pd.read_csv(args.submission_file, sep=',')
    
    has_right_columns = submission_data.columns.tolist() == ['Id', 'Category']
    if not has_right_columns:
        raise ValueError('Submission file must have columns "Id" and "Category"')
    
    if args.dataset == "spam":
        has_right_ids = submission_data.Id.tolist() == list(range(1, 1001))
        has_right_categories = submission_data.Category.isin([0, 1]).all()
    elif args.dataset == "mnist":
        has_right_ids = submission_data.Id.tolist() == list(range(1, 10001))
        has_right_categories = submission_data.Category.isin(list(range(10))).all()
    else:
        raise ValueError('Dataset must be "spam" or "mnist"')
    
    if not has_right_ids:
        raise ValueError('Submission file has incorrect Ids')
    if not has_right_categories:
        raise ValueError('Submission file has incorrect Categories')
    print('Submission file is valid!')
    
