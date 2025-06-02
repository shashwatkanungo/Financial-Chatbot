from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.model import load_model 
import pandas as pd
import numpy as np
import os
import pickle
import datetime
#from app.services.predict_service import predict  # adjust to your actual path

router = APIRouter()

# Define expected input schema
class PredictionInput(BaseModel):
    Amount: float
    Days_Until_Due: int
    Invoice_day: int
    Invoice_month: int
    Invoice_year: int

@router.post("/predict")
def predict_days_to_pay(input_data: PredictionInput):
    try:
        model = load_model()

        # Convert dict to DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])

        columns_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "services", "models", "stacking_columns.pkl")
        )

        with open(columns_path, "rb") as f:
            expected_columns = pickle.load(f)
        
        # Ensure all expected columns exist in the input (missing -> fill with 0)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0.0

        df = df[expected_columns]

        # Predict
        prediction = model.predict(df)[0]
        predicted_days = int(round(prediction))

        try:
            invoice_date = datetime.date(input_data.Invoice_year, input_data.Invoice_month, input_data.Invoice_day)
            expected_payment_date = invoice_date + datetime.timedelta(days=predicted_days)
        except ValueError as date_error:
            raise HTTPException(status_code=400, detail=f"Invalid invoice date: {date_error}")

        return {
            "Predicted Days to Pay": predicted_days,
            "Expected Payment Date": expected_payment_date.strftime("%Y-%m-%d")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
