import os
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from.config import CONF_MAT_DIR

def plot_confusion_matrix(y_true, y_pred, title, save_as=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_as:
        os.makedirs(CONF_MAT_DIR, exist_ok=True)
        path = os.path.join(CONF_MAT_DIR, save_as)
        plt.savefig(path, bbox_inches="tight")
    
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
        
    print(f"Train Accuracy: {accuracy_score(y_train, train_pred) * 100:.2f} %")
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred) * 100:.2f} %") 
    print('-' * 50)
    print("Test Classification Report:\n")
    print(classification_report(y_test, test_pred))
    
    # Area Under the Receiver Operating Characteristic Curve
    print(f"roc_auc: {roc_auc_score(y_test, y_prob):.2f}")
    # Area Under the Precision-Recall Curve
    print(f"pr_auc:  {average_precision_score(y_test, y_prob):.2f}") # useful for imbalanced dataset especially when positive class is rare