# -*- coding: utf-8 -*-
"""
Usage: python main.py <model> <method>
"""

import argparse
import sys
from src.logistic_regression import LogisticRegressionTrainer
from src.random_forest import RandomForestTrainer
from src.xgboost import XGBoostTrainer
from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET, TEST_SIZE, RANDOM_STATE
from src.data_prep import preprocess_raw_data, impute_total_charges
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    preprocess_raw_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col="customerID")
    
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_test = impute_total_charges(X_train, y_train, X_test, y_test)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    scale_pos_weight = None # default value for XGBoost trainer
    
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('model', choices=['logistic_regression', 'random_forest', 'xgboost'])
    parser.add_argument(
        'method', 
        choices=['', 'balanced', 'smote', 'adasyn'],
        nargs='?',
        default=None
        )
    args = parser.parse_args()
    if args.model == 'xgboost' and args.method == 'balanced':
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos
        
    trainers = {
        'logistic_regression': LogisticRegressionTrainer(method=args.method),
        'random_forest': RandomForestTrainer(method=args.method),
        'xgboost': XGBoostTrainer(method=args.method, scale_pos_weight=scale_pos_weight)
    }
    

    trainer = trainers[args.model]    
    trainer.train(X_train, X_test, y_train, y_test, args.method)

if __name__ == "__main__":
    main()
