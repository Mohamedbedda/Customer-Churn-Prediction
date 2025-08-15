from src.config import PROCESSED_DATA_PATH, TARGET, NUM_FEATURES, RANDOM_STATE, TEST_SIZE
from src.data_prep import impute_total_charges
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

# Charger données
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

# Logistic Regression
print("\nTraining Logistic Regression...")
lr = Pipeline([
    ('preprocessor', preprocessor),
    ('lr', LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE))
])
param_grid_lr = [{'lr__penalty': ['l1', 'l2'], 'lr__C': [0.01, 0.1, 1], 'lr__solver': ['liblinear']}]
gs_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
gs_lr.fit(X_train, y_train)
evaluate_model(gs_lr.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_lr.best_estimator_.predict(X_test),
                      "Logistic Regression",
                      save_as="logistic_regression.png")
save_model(gs_lr.best_estimator_, "logistic_regression.pkl")

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
param_grid_rf = {'n_estimators': [100, 300], 'max_depth': [None, 20]}
gs_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
gs_rf.fit(X_train, y_train)
evaluate_model(gs_rf.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_lr.best_estimator_.predict(X_test),
                      "Random Forest",
                      save_as="random_forest.png")
save_model(gs_rf.best_estimator_, "random_forest.pkl")

# XGBoost
print("\nTraining XGBoost...")
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
xgb = XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight)
param_grid_xgb = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.1]}
gs_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='f1', n_jobs=-1)
gs_xgb.fit(X_train, y_train)
evaluate_model(gs_xgb.best_estimator_, X_train, y_train, X_test, y_test)
plot_confusion_matrix(y_test, gs_lr.best_estimator_.predict(X_test),
                      "XGBoost",
                      save_as="xgboost.png")
save_model(gs_xgb.best_estimator_, "xgboost.pkl")
