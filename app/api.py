import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import ChurnPredictor

app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API for predicting customer churn using a pre-trained machine learning model.",
    version="1.0.0",
)

try:
    predictor = ChurnPredictor(model_path="models/churn_model.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChurnPredictor: {e}")
    predictor = None

class CustomerData(BaseModel):
    customerID: str = Field(..., example="7590-VHVEG")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=1)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: str = Field(..., example="29.85")

@app.get("/")
def home():
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if predictor is None:
        raise HTTPException(status_code=500, detail="ChurnPredictor is not initialized. Please check the server logs for more details.")
    try:
        input_dict = data.dict()
        result = predictor.predict(input_dict)
        return {
            "status": "success",
            "prediction_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")