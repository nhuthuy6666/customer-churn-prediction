import joblib
import pandas as pd

from feature_engineering import add_features

class ChurnPredictor:
    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")

    def predict(self, input_data: dict):
        if isinstance(input_data, dict):
            df_raw = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df_raw = input_data.copy()
        else:
            raise ValueError("Input data must be a dictionary or a pandas DataFrame.")

        df_feat = add_features(df_raw)

        if 'customerID' in df_feat.columns:
            df_feat.drop(columns=['customerID'], inplace=True)

        if 'TotalCharges' in df_feat.columns:
            df_feat['TotalCharges'] = pd.to_numeric(df_feat['TotalCharges'], errors='coerce').fillna(0.0)

        predictions = self.model.predict(df_feat)
        probabilities = self.model.predict_proba(df_feat)[:, 1]

        results = []

        for pred, prob in zip(predictions, probabilities):
            results.append({
                "churn_prediction": int(pred),
                "churn_label": "yes" if pred == 1 else "no",
                "churn_probability": round(float(prob), 4)
            })

        return results[0] if isinstance(input_data, dict) else results

if __name__ == "__main__":
    sample_customer = {
        "customerID": "9999-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": "85.5"
    }

    predictor = ChurnPredictor("models/churn_model.pkl")
    result = predictor.predict(sample_customer)

    print(result)