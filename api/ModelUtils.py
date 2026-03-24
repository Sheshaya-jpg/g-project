# Data Analysis and Machine Learning utilities for the insurance model API
import pandas as pd

def FeatureEngineering(PatientData: dict) -> pd.DataFrame:
    # Covert to DataFrame for processing
    data = PatientData.model_dump()
    df = pd.DataFrame([PatientData])

    #Feature Engineeringg
    df['smoker_yes'] = df['smoker'] == 'yes'
    df['bmi_smoker'] = df['bmi'] * df['smoker_yes']
    df["age_smoker"] = df["age"] * df["smoker_yes"]
    df["age_bmi"] = df["age"] * df["bmi"]
    df["no_children"] = df["children"] + 1 
    
    return df
