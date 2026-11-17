import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loader import load_data
from feature_engineering import add_features
from preprocessing import clean_and_preprocess_data

def train_model():
    df = load_data()

    df_feat = add_features(df)

    X, Y, preprocessor = clean_and_preprocess_data(df_feat)

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=Y
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=3, random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
    }

    best_model_name = ""
    best_model_pipeline = None
    best_pr_auc = -1.0

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, Y_train)

        Y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(Y_val, Y_val_prob)
        roc_auc = roc_auc_score(Y_val, Y_val_prob)

        print(f"-> Model: {name} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_model_pipeline = pipeline

        print(f"Best Model so far: {best_model_name} with PR-AUC: {best_pr_auc:.4f}")

        os.makedirs("models", exist_ok=True)
        model_path = "models/churn_model.pkl"
        joblib.dump(best_model_pipeline, model_path)

        test_data = X_test.copy()
        test_data['Churn'] = Y_test
        test_data.to_csv("data/test_data.csv", index=False)

if __name__ == "__main__":
    train_model()