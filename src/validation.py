import pandas as pd
import numpy as np

def validate_raw_data(df: pd.DataFrame):
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df['customerID'].isnull().sum() > 0:
        print("Warning: Missing values found in 'customerID' column.")

    total_charges_converted = pd.to_numeric(df['TotalCharges'], errors='coerce')
    invalid_count = total_charges_converted.isna().sum()

    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid 'TotalCharges' values found that could not be converted to numeric.")

    print("Raw data validation completed successfully.")

    return True

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    validate_raw_data(df)