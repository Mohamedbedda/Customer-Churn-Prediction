import numpy as np
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from src.config import RANDOM_STATE
from src.base_trainer import BaseTrainer

class XGBoostTrainer(BaseTrainer):
    
    def __init__(self, method=None, scale_pos_weight=None):
        super().__init__()
        self.method = method
        self.scale_pos_weight = scale_pos_weight
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': np.arange(5, 21, 5),
            'learning_rate': [0.001, 0.01, 0.1]
        }
        
    def get_name(self):
        if self.method is None:
            return "XGBoost"
        return "XGBoost " + self.method
    
    
    def get_model(self, method):
        if method is None:
            return XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
            
        elif method == 'balanced':
            return XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', 
                               scale_pos_weight=self.scale_pos_weight)
            
        elif method == 'smote':
            xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
            return ImbPipeline([
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('model', xgb)
            ])
            
        elif method == 'adasyn':
            xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
            return ImbPipeline([
                ('adasyn', ADASYN(random_state=RANDOM_STATE)),
                ('model', xgb)
            ])
