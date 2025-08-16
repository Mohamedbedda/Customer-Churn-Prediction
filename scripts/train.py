import numpy as np
from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET, NUM_FEATURES, RANDOM_STATE, TEST_SIZE
from src.data_prep import impute_total_charges, preprocess_raw_data
from src.evaluation import evaluate_model, plot_confusion_matrix
from src.utils import save_model

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier

# load and preprocess the data
preprocess_raw_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
df = pd.read_csv(PROCESSED_DATA_PATH, index_col="customerID")
X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_test = impute_total_charges(X_train, y_train, X_test, y_test)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUM_FEATURES)
])

#############################################
# Logistic Regression
#############################################

print("\nTraining Logistic Regression...")
lr = Pipeline([
    ('preprocessor', preprocessor),
    ('lr', LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, solver='liblinear'))
])
param_grid_lr = [
    {
        'lr__penalty': ['l1', 'l2'], 
        'lr__C': np.logspace(-2, 2, num=5),
    }
]
gs_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
gs_lr.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", gs_lr.best_params_)
evaluate_model(gs_lr.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_lr.best_estimator_.predict(X_test),
                      "Logistic Regression",
                      save_as="logistic_regression.png")
save_model(gs_lr.best_estimator_, "logistic_regression.pkl")

# #############################################

# # Logistic Regression with SMOTE
print("\nTraining Logistic Regression with SMOTE sampling...")
smote = SMOTE(random_state=RANDOM_STATE)
lr_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('lr', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'))
])
gs_lr_smote = GridSearchCV(lr_smote, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
gs_lr_smote.fit(X_train, y_train)
print("Best parameters for Logistic Regression with SMOTE:", gs_lr_smote.best_params_)
evaluate_model(gs_lr_smote.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_lr_smote.best_estimator_.predict(X_test),
                      "Logistic Regression (SMOTE)",
                      save_as="logistic_regression_smote.png")
save_model(gs_lr_smote.best_estimator_, "logistic_regression_smote.pkl")

#############################################

# Logistic Regression with ADASYN sampling
print("\nTraining Logistic Regression with ADASYN sampling...")
adasyn = ADASYN(random_state=RANDOM_STATE, n_neighbors=5)
lr_adasyn = ImbPipeline([
    ('preprocessor', preprocessor),
    ('adasyn', adasyn),
    ('lr', LogisticRegression(random_state=RANDOM_STATE, solver='liblinear'))
])
gs_lr_adasyn = GridSearchCV(lr_adasyn, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
gs_lr_adasyn.fit(X_train, y_train)
print("Best parameters for Logistic Regression with ADASYN:", gs_lr_adasyn.best_params_)
evaluate_model(gs_lr_adasyn.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_lr_adasyn.best_estimator_.predict(X_test),
                      "Logistic Regression (ADASYN)",
                      save_as="logistic_regression_adasyn.png")
save_model(gs_lr_adasyn.best_estimator_, "logistic_regression_adasyn.pkl")

#############################################
# Random Forest
#############################################

print("\nTraining Random Forest...")
rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
param_grid_rf = {
    'n_estimators': np.arange(100, 501, 100),
    'max_depth': np.arange(5, 21, 5),
    'max_features': ['sqrt', 'log2']
    }
gs_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
gs_rf.fit(X_train, y_train)
print("Best parameters for Random Forest:", gs_rf.best_params_)
evaluate_model(gs_rf.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_rf.best_estimator_.predict(X_test),
                      "Random Forest",
                      save_as="random_forest.png")
save_model(gs_rf.best_estimator_, "random_forest.pkl")

#############################################

print("\nTraining Balanced Random Forest...")
brf = BalancedRandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)
gs_brf = GridSearchCV(
    brf,
    param_grid_rf,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
gs_brf.fit(X_train, y_train)
print("Best parameters for Balanced Random Forest:", gs_brf.best_params_)
evaluate_model(gs_brf.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_brf.best_estimator_.predict(X_test),
                      "Balanced Random Forest",
                      save_as="balanced_random_forest.png")
save_model(gs_brf.best_estimator_, "balanced_random_forest.pkl")

# #############################################

print("\nTraining Random Forest with SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE)

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
)

pipe_rf_smote = ImbPipeline([
    ('smote', smote),
    ('rf', rf)
])
param_grid_rf_smote = {
    'rf__n_estimators': np.arange(100, 501, 100),
    'rf__max_depth': np.arange(5, 21, 5),
    'rf__max_features': ['sqrt', 'log2']
}

gs_rf_smote = GridSearchCV(
    pipe_rf_smote,
    param_grid_rf_smote,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

gs_rf_smote.fit(X_train, y_train)

print("Best parameters for Random Forest with SMOTE:", gs_rf_smote.best_params_)

evaluate_model(gs_rf_smote.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_rf_smote.best_estimator_.predict(X_test),
                      "Random Forest (SMOTE)",
                      save_as="random_forest_smote.png")
save_model(gs_rf_smote.best_estimator_, "random_forest_smote.pkl")

#############################################
# XGBoost
#############################################
print("\nTraining XGBoost...")
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
xgb = XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight)

param_grid_xgb = {
    'n_estimators': np.arange(100, 501, 100),
    'max_depth': np.arange(5, 21, 5),
    'learning_rate': np.logspace(-2, 0, num=5),
}
gs_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='f1', n_jobs=-1)
gs_xgb.fit(X_train, y_train)
print("Best parameters for XGBoost:", gs_xgb.best_params_)
evaluate_model(gs_xgb.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_xgb.best_estimator_.predict(X_test),
                      "XGBoost",
                      save_as="xgboost.png")
save_model(gs_xgb.best_estimator_, "xgboost.pkl")



