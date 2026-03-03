from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Initialize the FastAPI app
app = FastAPI(title="FinAgent-360 API Engine")

# 2. Path to your trained brain
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fin_brain.pkl')

# 3. Load the Multi-Model Ensemble
# This is like waking up the brain we just saved
model = joblib.load(MODEL_PATH)

# 4. Define what an "Application" looks like (The Schema)
class LoanApplication(BaseModel):
    income: float
    score: int
    debt: float

@app.get("/")
def home():
    return {"message": "FinAgent-360 AI Engine is Online"}

# 5. The Prediction Endpoint
@app.post("/predict")
def predict_loan(data: LoanApplication):
    # Convert input data into a format the AI understands
    features = np.array([[data.income, data.score, data.debt]])
    
    # Get the decision (0 or 1)
    prediction = model.predict(features)[0]
    
    # Get the confidence (How sure is the AI?)
    probability = model.predict_proba(features)[0]
    confidence = max(probability) * 100
    
    return {
        "decision": "Approved" if prediction == 1 else "Rejected",
        "confidence": f"{confidence:.2f}%",
        "status": "Success"
    }