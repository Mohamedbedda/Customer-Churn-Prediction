# -*- coding: utf-8 -*-
"""
Usage: python train.py <model> <method>
"""

import argparse
from src.logistic_regression import LogisticRegressionTrainer
from src.random_forest import RandomForestTrainer
from src.xgboost import XGBoostTrainer
from src.data_prep import load_data

def main():
    X_train, X_test, y_train, y_test = load_data()
    scale_pos_weight = None # default value for XGBoost trainer

    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('model', choices=['lr', 'rf', 'xgb'])
    parser.add_argument(
        'method', 
        choices=['balanced', 'smote', 'adasyn'],
        nargs='?',
        default=None
        )
    args = parser.parse_args()
    if args.model == 'xgboost' and args.method == 'balanced':
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos

    # Select model
    trainers = {
        'lr': LogisticRegressionTrainer(method=args.method),
        'rf': RandomForestTrainer(method=args.method),
        'xgb': XGBoostTrainer(method=args.method, scale_pos_weight=scale_pos_weight)
    }
    

    trainer = trainers[args.model]    
    trainer.train(X_train, X_test, y_train, y_test, args.method)

if __name__ == "__main__":
    main()
