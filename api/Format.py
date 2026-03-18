# Data validation and formatting for the insurance model API

from pydantic import BaseModel, Field
from typing import Literal

class DataInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the Patient")
    sex: Literal["male", "female"]
    bmi: float = Field(..., ge=0, le=100, description="Body Mass Index")
    smoker: Literal["yes", "no"]
    children: int = Field(..., ge=0, le=20, description="Number of Children")
    region: Literal["northeast", "northwest", "southeast", "southwest"]

class PredictionOutput(BaseModel):
    PredictedCost: float = Field(..., ge=0, description="Predicted Insurance Cost")
    