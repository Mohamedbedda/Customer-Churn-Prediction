# 📊 Customer Churn Prediction

This repository contains a **machine learning project** for predicting customer churn in the telecom sector, using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
It demonstrates an **end-to-end ML workflow**, from data preparation to model evaluation, with integrated **experiment tracking using MLflow**.



---

## 🚀 Features

* 🛠 **Data preparation** (cleaning, encoding, normalization)
* 📓 **Exploratory Data Analysis (EDA)** in Jupyter notebook
* 🤖 **Multiple models implemented**: Logistic Regression, Random Forest, XGBoost
* 🎛 **Hyperparameter fine-tuning** with grid search / cross-validation
* ⚖️ **Handling class imbalance** using weighted loss and resampling methods (SMOTE, ADASYN)
* 📈 **Evaluation helpers**: metrics, confusion matrix
* 🔬 **Experiment tracking with MLflow**

---

## 📂 Project Structure

```
.
├── data/                     # Raw and processed datasets
├── notebooks/
│   └── EDA.ipynb             # Exploratory data analysis
├── reports/                  # Evaluation outputs (metrics, plots)
├── src/
│   ├── base_trainer.py       # Abstract class for models training
│   ├── config.py             # Paths, hyperparameters, configuration
│   ├── data_prep.py          # Data loading & preprocessing
│   ├── evaluation.py         # Metrics and reporting helpers
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── xgboost.py
├── train.py                  # Main training entry point
├── mlflow_train.py           # Training with MLflow experiment tracking
└── requirements.txt          # Dependencies
```

---

## ⚡ Quickstart

### 1. Create and activate a virtual environment

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train a model
Run a model directly (`<method>` choices: `lr` = Logistic Regression, `rf` = Random Forest, `xgb` = XGBoost).

You can handle class imbalance with different methods (optional argument `<method>`):  
- `balanced` → applies **class weights** (weighted loss).  
- `smote` → applies **Synthetic Minority Oversampling Technique**.  
- `adasyn` → applies **Adaptive Synthetic Sampling**.  

If `<method>` is omitted, the model is trained without class balancing.

**Default training (no tracking):**
```bash
python train.py <model> <method>
```

**With MLflow experiment tracking:**
For full experiment logging (parameters, metrics, artifacts, confusion matrix, saved model):

```bash
python mlflow_train.py <model> <method>
```

### 4. Evaluate results

* The evaluation results are printed on the terminal, and the confusion matrix plots are saved under `reports/`.
* You can also explore the results interactively in the **MLflow UI**:

```bash
mlflow ui
```

👉 Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to compare runs.

---
