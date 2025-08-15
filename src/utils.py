import os
import joblib
from .config import MODEL_DIR

def save_model(model, filename):
    """Save a trained model to the models directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(filename):
    """Load a saved model from the models directory."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
