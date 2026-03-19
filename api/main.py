# main file for the API

import sys
import os
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.Format import DataRequest, PredictionResponse
from api.ModelUtils import FeatureEngineering
from api import CostumModel

# map the model to the main module for pickling
sys.modules["__main__"] = CostumModel

# hold model in memory
ModelComponent = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    BaseDirectory = os.path.dirname(os.path.abspath(__file__))
    ModelPath = os.path.join(BaseDirectory, "model.joblib")
    PreprocessorPath = os.path.join(BaseDirectory, "preprocessor.joblib")

    try:
        ModelComponent["model"] = joblib.load(ModelPath)
        ModelComponent["preprocessor"] = joblib.load(PreprocessorPath)
        print("Model and Preprocessor loaded successfully.")
    except FileNotFoundError as err:
        print(f"Error loading files: {err}")

    yield                                               #runnig the API
    ModelComponent.clear()                              #clear the model from memory when API is shutdown

    App = FastAPI(
        title="Medical Insurance Prediction",
        lifespan=lifespan)
    App.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @App.get("/")
    def HealthCheck():
        return {"Status": "Active",
                "Message": "API is running and ready to accept requests.",
                "ModelStatus": "Model" in ModelComponent and "preprocessor" in ModelComponent
                }






 