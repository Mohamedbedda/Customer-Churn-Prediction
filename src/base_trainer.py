from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from src.evaluation import evaluate_model, plot_confusion_matrix

class BaseTrainer(ABC):
    
    def __init__(self):
        self.param_grid = {}
        
    @abstractmethod
    def get_model(self, method):
        pass
    
    @abstractmethod
    def get_name(self):
        pass
    
    def get_params(self):
        if self.method in ['smote', 'adasyn']:
            return {f'model__{k}': v for k, v in self.param_grid.items()}
        else:
            return self.param_grid
    
    def train(self, X_train, X_test, y_train, y_test, method):

        print(f"Training {self.get_name()}...")
        model = self.get_model(method)

        params = self.get_params()
        
        # GridSearch
        gs = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        
        print(f"Best parameters for {self.get_name()}: {gs.best_params_}")
        
        # Evaluation and saving the confusion matrix
        best_model = gs.best_estimator_
        results = evaluate_model(best_model, X_train, y_train, X_test, y_test)
        
        filename = f"{self.get_name().lower().replace(' ', '_')}"
        plot_confusion_matrix(y_test, best_model.predict(X_test), 
                            f"{self.get_name()}", 
                            save_as=f"{filename}.png")
                
        return best_model, gs.best_params_, results
