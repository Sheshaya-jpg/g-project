# main file for the API
import sys
import os
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.Format import DataRequest, PredictionResponse, PermutationImportanceResponse
from api.ModelUtils import feature_engineering
import api.CostumModel

sys.modules['__main__'] = api.CostumModel  # Ensure the custom model is available in the main namespace for joblib loading
model_component = {} # hold model in memory

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Move up to d:\sen-proj
    model_path = os.path.join(base_directory, "model", "insurance_gbmodel.pkl")
    preprocessor_path = os.path.join(base_directory, "model", "insurance_preprocessor.pkl")

    try:
        model_component["InsuranceModel"] = joblib.load(model_path)
        model_component["InsurancePreprocessor"] = joblib.load(preprocessor_path)
        print("Model and Preprocessor loaded successfully.")
    except FileNotFoundError as err:
        print(f"Error loading files: {err}")
        raise

    yield  # running the API
    model_component.clear()  # clear the model from memory when API is shutdown

app = FastAPI(
    title="Medical Insurance Prediction",
    lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/") # Health Check endpoint
async def check():
    return {"Status": "Active",
            "Message": "API is running and ready to accept requests.",
            "ModelStatus": "InsuranceModel" in model_component and "InsurancePreprocessor" in model_component
            }
   
@app.post("/predict", response_model=PredictionResponse) # prediction endpoint
async def predict(patient_data: DataRequest):
    if "InsuranceModel" not in model_component or "InsurancePreprocessor" not in model_component:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded")

    try:
        df = feature_engineering(patient_data)
        X = model_component["InsurancePreprocessor"].transform(df)
        predicted_cost = model_component["InsuranceModel"].predict(X)[0]
        return PredictionResponse(Predicted_Cost=predicted_cost)
    except ValueError as verr:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {verr}")
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Prediction error: {err}")
    
@app.get("/permutation_importance", response_model=PermutationImportanceResponse) # permutation importance endpoint
async def PermutationImportance():    # Avarage Permutation importance scores from jupyter
    return PermutationImportanceResponse(
        Feature_Importance={
            "smoker": 1.4149,
            "bmi": 0.2875,
            "age": 0.1994,
            "children": 0.0054,
            "sex": 0.0002,
            "region": -0.0003,
        },
        Engineering_Features={
            "bmi_smoker": 0.8739,
            "bmi_age": 0.0251,
            "age_smoker": 0.0244,
            "smoker_yes": 0.0112,
            "no_children": 0.0004,
        })