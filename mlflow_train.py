# -*- coding: utf-8 -*-
"""
Usage: python mlflow_train.py <model> <method>
"""

import argparse
import mlflow
import mlflow.sklearn

from src.evaluation import plot_confusion_matrix
from src.logistic_regression import LogisticRegressionTrainer
from src.random_forest import RandomForestTrainer
from src.xgboost import XGBoostTrainer
from src.data_prep import load_data


def main():
    X_train, X_test, y_train, y_test = load_data()
    scale_pos_weight = None # default value for XGBoost trainer

    parser = argparse.ArgumentParser(description='model training with MLflow')
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
    model_name = trainer.get_name()

    # Start MLflow run
    mlflow.set_experiment("customer_churn_prediction")
    with mlflow.start_run(run_name=model_name):

        # Train
        model, param, results = trainer.train(X_train, X_test, y_train, y_test, args.method)
        mlflow.log_param("best_param", param)

        # Evaluate
        y_pred = model.predict(X_test)

        train_acc = round(results["train_accuracy"], 2)
        test_acc = round(results["test_accuracy"], 2)
        f1 = round(results["f1_score"], 2)
        roc_auc = round(results["roc_auc"], 2)
        pr_auc = round(results["pr_auc"], 2)

        # Log metrics
        mlflow.log_metric("train_acc", train_acc)
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        filename = f"{model_name.lower().replace(' ', '_')}"
        conf_path = plot_confusion_matrix(
            y_test, y_pred, 
            title=f"Confusion Matrix ({model_name})", 
            save_as=f"{filename}.png"
            )   
        mlflow.log_artifact(conf_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")
        print(f"Model {args.model} trained and logged to MLflow.")

if __name__ == "__main__":
    main()
