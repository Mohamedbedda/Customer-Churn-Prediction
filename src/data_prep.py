import pandas as pd
import numpy as np

def preprocess_raw_data(csv_input_path, csv_output_path):
    
    df = pd.read_csv(csv_input_path, index_col="customerID")
    
    # Binary columns mapping
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    df[binary_cols] = df[binary_cols].replace({'No': 0, 'Yes': 1})
    
    df["gender"] = df["gender"].replace({'Male': 0, 'Female': 1})
    
    # One-hot encode multi-categorical columns
    multi_categ_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_categ_cols, dtype=int)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    df.to_csv(csv_output_path)
    print(f"Preprocessed data saved to {csv_output_path}")

def impute_total_charges(X_train, y_train, X_test, y_test):
    """Impute missing TotalCharges using median per class."""
    train_df = X_train.copy()
    train_df["Churn"] = y_train
    medians = train_df.groupby("Churn")["TotalCharges"].median()

    def impute(row, medians_map, target_series=None):
        if pd.isna(row["TotalCharges"]):
            if target_series is not None:
                return medians_map[target_series.loc[row.name]]
            else:
                return medians_map[row["Churn"]]
        return row["TotalCharges"]

    X_train["TotalCharges"] = train_df.apply(impute, axis=1, medians_map=medians)
    X_test["TotalCharges"] = X_test.apply(impute, axis=1, medians_map=medians, target_series=y_test)

    return X_train, X_test