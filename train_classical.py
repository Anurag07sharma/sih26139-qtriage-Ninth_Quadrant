import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
import os

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else preds
    print(f"--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1-Score:  {f1_score(y_test, preds):.4f}\n")

def main():
    os.makedirs('models', exist_ok=True)
    
    # 1. Heart Disease
    print("Evaluating Heart Disease Track...")
    heart_train = pd.read_csv('data/processed/heart_train.csv')
    heart_test = pd.read_csv('data/processed/heart_test.csv')
    
    # Assuming 'target' is the dependent variable (as per context.md)
    # If the column is named differently, we'll catch it.
    target_col = 'target'
    if 'target' not in heart_train.columns:
        target_col = [c for c in heart_train.columns if c.lower() in ['target', 'tenyearchd', 'heartdisease']][0]

    X_train_h, y_train_h = heart_train.drop(target_col, axis=1), heart_train[target_col]
    X_test_h, y_test_h = heart_test.drop(target_col, axis=1), heart_test[target_col]
    
    rf_h = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_h, y_train_h)
    xgb_h = XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train_h, y_train_h)
    lr_h = LogisticRegression(max_iter=1000).fit(X_train_h, y_train_h)
    
    evaluate_model("Random Forest (Heart)", rf_h, X_test_h, y_test_h)
    evaluate_model("XGBoost (Heart)", xgb_h, X_test_h, y_test_h)
    evaluate_model("Logistic Regression (Heart)", lr_h, X_test_h, y_test_h)
    
    joblib.dump(rf_h, 'models/heart_rf.pkl')
    
    # 2. Diabetes
    print("Evaluating Diabetes Track...")
    diabetes_train = pd.read_csv('data/processed/diabetes_train.csv')
    diabetes_test = pd.read_csv('data/processed/diabetes_test.csv')
    
    target_col_d = 'target'
    if 'target' not in diabetes_train.columns:
        target_col_d = [c for c in diabetes_train.columns if c.lower() in ['target', 'diabetes', 'outcome']][0]

    X_train_d, y_train_d = diabetes_train.drop(target_col_d, axis=1), diabetes_train[target_col_d]
    X_test_d, y_test_d = diabetes_test.drop(target_col_d, axis=1), diabetes_test[target_col_d]
    
    rf_d = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_d, y_train_d)
    xgb_d = XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train_d, y_train_d)
    lr_d = LogisticRegression(max_iter=1000).fit(X_train_d, y_train_d)
    
    evaluate_model("Random Forest (Diabetes)", rf_d, X_test_d, y_test_d)
    evaluate_model("XGBoost (Diabetes)", xgb_d, X_test_d, y_test_d)
    evaluate_model("Logistic Regression (Diabetes)", lr_d, X_test_d, y_test_d)
    
    joblib.dump(xgb_d, 'models/diabetes_xgb.pkl')
    print("Classical training complete. Models saved.")

if __name__ == "__main__":
    main()
