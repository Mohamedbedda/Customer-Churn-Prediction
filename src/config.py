RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/telco_churn_processed.csv"
MODEL_DIR = "models/"
CONF_MAT_DIR = "reports/confusion_matrices/"
FEAT_IMP_DIR = "reports/feature_importances/"

TARGET = "Churn"
NUM_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
RANDOM_STATE = 42
TEST_SIZE = 0.25
