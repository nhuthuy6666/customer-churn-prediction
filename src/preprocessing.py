import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_and_preprocess_data(df: pd.DataFrame):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)

    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['Churn'])
    Y = df['Churn']

    numerical_col = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_col = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_col),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_col)
        ]
    )

    return X, Y, preprocessor

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    X, Y, preprocessor = clean_and_preprocess_data(df)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
