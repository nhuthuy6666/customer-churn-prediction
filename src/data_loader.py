import pandas as pd
import os

def load_data(file_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    df = pd.read_csv(file_path)
    print("File read")

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head(2))