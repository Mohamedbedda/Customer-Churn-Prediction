from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_prep import preprocess_raw_data

if __name__ == "__main__":
    preprocess_raw_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)