import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame):
    df_feat = df.copy()

    if 'TotalCharges' in df_feat.columns:
        df_feat['TotalCharges'] = pd.to_numeric(df_feat['TotalCharges'], errors='coerce')
        df_feat['TotalCharges'] = df_feat['TotalCharges'].fillna(df_feat['TotalCharges'].median())

    if 'tenure' in df_feat.columns:
        safe_tenure = np.maximum(df_feat['tenure'], 1)

        df_feat['AverageChargesPerMonth'] = df_feat['TotalCharges'] / safe_tenure

    if 'Contract' in df_feat.columns:
        df_feat['IsLongTermContract'] = df_feat['Contract'].apply(
            lambda x: 0 if str(x).strip().lower() == 'month-to-month' else 1
        )

    return df_feat

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    df_engineered = add_features(df)
    print(df_engineered[['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageChargesPerMonth', 'Contract', 'IsLongTermContract']].head(3))