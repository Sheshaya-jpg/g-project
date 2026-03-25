# Data validation and formatting for the insurance model API

from pydantic import BaseModel, Field
from typing import Literal, Dict

class DataRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the Patient")
    sex: Literal["male", "female"]
    bmi: float = Field(..., ge=0, le=100, description="Body Mass Index")
    smoker: Literal["yes", "no"]
    children: int = Field(..., ge=0, le=20, description="Number of Children")
    region: Literal["northeast", "northwest", "southeast", "southwest"]

class PredictionResponse(BaseModel):
    Predicted_Cost: float = Field(..., ge=0, description="Predicted Insurance Cost")

class PermutationImportanceResponse(BaseModel):
    Feature_Importance: Dict[str, float] = Field(..., description="Permutation importance scores for each feature")
    Engineering_Features: Dict[str, float] = Field(..., description="Permutation importance scores for engineered features")