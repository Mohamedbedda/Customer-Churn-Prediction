import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from src.config import NUM_FEATURES, RANDOM_STATE
from src.base_trainer import BaseTrainer

class LogisticRegressionTrainer(BaseTrainer):
    
    def __init__(self, method=None):
        super().__init__()
        self.method = method
        self.param_grid = {
                'C': np.logspace(-2, 2, 5),
                'penalty': ['l1', 'l2']
        }
        
    def get_name(self):
        if self.method is None:
            return "Logistic Regression"
        return "Logistic Regression " + self.method
    
    def get_params(self):
        if self.method in [None, "balanced"]:
            return {f'model__{k}': v for k, v in self.param_grid.items()}
        else:
            return {f'pipeline__model__{k}': v for k, v in self.param_grid.items()}

    
    def get_model(self, method):
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), NUM_FEATURES)
        ])
        
        lr = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear')
        
        if method in [None, 'balanced']:
            if method == "balanced":
                lr.set_params(class_weight='balanced')
            return Pipeline([
                ('preprocessor', preprocessor),
                ('model', lr)
            ])
            
        elif method == 'smote':
            base_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', lr)
            ])
            return ImbPipeline([
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('pipeline', base_pipeline)
            ])
            
        elif method == 'adasyn':
            base_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', lr)
            ])
            return ImbPipeline([
                ('adasyn', ADASYN(random_state=RANDOM_STATE)),
                ('pipeline', base_pipeline)
            ])
