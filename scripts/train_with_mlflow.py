# -*- coding: utf-8 -*-
"""
Usage: python main.py <model> <method>
MLflow-integrated training script for customer churn prediction.
"""

import argparse
import sys
import mlflow
import mlflow.sklearn
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.logistic_regression import LogisticRegressionTrainer
from src.random_forest import RandomForestTrainer
from src.xgboost import XGBoostTrainer
from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET, TEST_SIZE, RANDOM_STATE
from src.data_prep import preprocess_raw_data, impute_total_charges

def load_data():
    """Load and prepare the dataset."""
    preprocess_raw_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col="customerID")
    
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_test = impute_total_charges(X_train, y_train, X_test, y_test)
    return X_train, X_test, y_train, y_test

def calculate_metrics(model, X_train, y_train, X_test, y_test):
    """Calculate comprehensive metrics for train and test sets."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Get probabilities for AUC calculation
    if hasattr(model, 'predict_proba'):
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        test_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba
        train_pred_proba = train_pred
        test_pred_proba = test_pred
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'train_precision': precision_score(y_train, train_pred),
        'train_recall': recall_score(y_train, train_pred),
        'train_f1': f1_score(y_train, train_pred),
        'train_auc': roc_auc_score(y_train, train_pred_proba),
        
        'test_accuracy': accuracy_score(y_test, test_pred),
        'test_precision': precision_score(y_test, test_pred),
        'test_recall': recall_score(y_test, test_pred),
        'test_f1': f1_score(y_test, test_pred),
        'test_auc': roc_auc_score(y_test, test_pred_proba)
    }
    
    return metrics

def train_with_mlflow(trainer, X_train, X_test, y_train, y_test, method, model_name):
    """Train model with MLflow tracking."""
    
    # Set experiment name
    experiment_name = "customer_churn_prediction"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model_name}_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("method", method or "default")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("cv_scoring", "f1")
        
        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("class_imbalance", f"{(y_train == 0).sum()}:{(y_train == 1).sum()}")
        
        # Special parameters for XGBoost balanced
        if isinstance(trainer, XGBoostTrainer) and method == 'balanced':
            mlflow.log_param("scale_pos_weight", trainer.scale_pos_weight)
        
        print(f"Training {trainer.get_name()} with MLflow tracking...")
        
        # Train the model (this calls the original train method)
        model = trainer.get_model(method)
        params = trainer.get_params()
        
        # Log hyperparameter grid
        for param, values in params.items():
            mlflow.log_param(f"grid_{param}", str(values))
        
        # Perform grid search
        from sklearn.model_selection import GridSearchCV
        gs = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        # Log best parameters
        for param, value in gs.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_metric("best_f1score", gs.best_score_)
        
        # Get best model
        best_model = gs.best_estimator_
        
        # Calculate and log comprehensive metrics
        metrics = calculate_metrics(best_model, X_train, y_train, X_test, y_test)
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model, 
            "model",
            registered_model_name=f"{model_name.replace(' ', '_')}_{method or ''}"
        )
        
        # Log additional artifacts
        import tempfile
        import os
        
        # Create temporary files for additional logging
        with tempfile.TemporaryDirectory() as tmp_dir:
            
            # Log feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                feature_importance_path = os.path.join(tmp_dir, "feature_importance.txt")
                with open(feature_importance_path, 'w') as f:
                    for i, importance in enumerate(best_model.feature_importances_):
                        f.write(f"feature_{i}: {importance}\n")
                mlflow.log_artifact(feature_importance_path)
            
            # Log model summary
            summary_path = os.path.join(tmp_dir, "model_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Model: {trainer.get_name()}\n")
                f.write(f"Method: {method or 'default'}\n")
                f.write(f"Best CV Score: {gs.best_score_:.4f}\n")
                f.write(f"Best Parameters: {gs.best_params_}\n")
                f.write(f"\nTest Metrics:\n")
                for metric, value in metrics.items():
                    if metric.startswith('test_'):
                        f.write(f"{metric}: {value:.4f}\n")
            mlflow.log_artifact(summary_path)
        
        # Print results
        print(f"Best parameters for {trainer.get_name()}: {gs.best_params_}")
        print(f"Test F1 Score: {metrics['test_f1']:.4f}")
        print(f"Test AUC Score: {metrics['test_auc']:.4f}")
        
        return best_model, metrics

def main():
    """Main function with MLflow integration."""
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Model training with MLflow')
    parser.add_argument('model', choices=['logistic_regression', 'random_forest', 'xgboost'])
    parser.add_argument(
        'method', 
        choices=['', 'balanced', 'smote', 'adasyn'],
        nargs='?',
        default=None
    )
    parser.add_argument('--experiment-name', default='customer_churn_prediction',
                       help='MLflow experiment name')
    parser.add_argument('--tracking-uri', default=None,
                       help='MLflow tracking URI (default: local)')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    # Calculate scale_pos_weight for XGBoost balanced
    scale_pos_weight = None
    if args.model == 'xgboost' and args.method == 'balanced':
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos
    
    # Initialize trainers
    trainers = {
        'logistic_regression': LogisticRegressionTrainer(method=args.method),
        'random_forest': RandomForestTrainer(method=args.method),
        'xgboost': XGBoostTrainer(method=args.method, scale_pos_weight=scale_pos_weight)
    }
    
    trainer = trainers[args.model]
    
    # Train with MLflow tracking
    try:
        model, metrics = train_with_mlflow(
            trainer, X_train, X_test, y_train, y_test, 
            args.method, trainer.get_name()
        )
        
        print(f"\Training completed successfully!")
        print(f"Results logged to MLflow experiment: {args.experiment_name}")

        
    except Exception as e:
        print(f"Training failed with error: {e}")
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        raise

if __name__ == "__main__":
    main()