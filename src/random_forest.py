import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from src.config import RANDOM_STATE
from src.base_trainer import BaseTrainer

class RandomForestTrainer(BaseTrainer):
    
    def __init__(self, method=None):
        super().__init__()
        self.method = method
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': np.arange(5, 21, 5),
            'max_features': ['sqrt', 'log2']
        }
        
    def get_name(self):
        if self.method is None:
            return "Random Forest"
        return "Random Forest " + self.method
    
    def get_model(self, method):
        if method is None:
            return RandomForestClassifier(random_state=RANDOM_STATE)
            
        elif method == 'balanced':
            return RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
            
        elif method == 'smote':
            rf = RandomForestClassifier(random_state=RANDOM_STATE)
            return ImbPipeline([
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('model', rf)
            ])
            
        elif method == 'adasyn':
            rf = RandomForestClassifier(random_state=RANDOM_STATE)
            return ImbPipeline([
                ('adasyn', ADASYN(random_state=RANDOM_STATE)),
                ('model', rf)
            ])
