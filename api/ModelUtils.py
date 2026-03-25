# Data Analysis and Machine Learning utilities for the insurance model API
import pandas as pd
from api.Format import DataRequest

def feature_engineering(patient: DataRequest) -> pd.DataFrame:
    # Convert request object to DataFrame for processing
    df = pd.DataFrame([patient.model_dump()])

    # Feature engineering
    df['smoker_yes'] = (df['smoker'] == 'yes').astype(int)
    df['bmi_smoker'] = df['bmi'] * df['smoker_yes']
    df['age_smoker'] = df['age'] * df['smoker_yes']
    df['bmi_age'] = df['bmi'] * df['age']
    df['no_children'] = (df['children'] == 0).astype(int)

    return df
