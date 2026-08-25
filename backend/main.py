from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

preprocessor = joblib.load('models/preprocessor.pkl')
model = joblib.load('models/logistic_regression.pkl')  
explainer = joblib.load('models/shap_explainer.pkl')

with open('models/model_config.json') as f:
    model_config = json.load(f)

with open('data/feature_names.json') as f:
    feature_names = json.load(f)

OPTIMAL_THRESHOLD = model_config['optimal_threshold']

print(f"Model loaded: {model_config['model_name']}, threshold: {OPTIMAL_THRESHOLD}")

class CustomerInput(BaseModel):
    """
    Defines exactly what fields the API expects, with types.
    Pydantic automatically validates incoming data against this — 
    if a field is missing or the wrong type, FastAPI returns a clear error 
    instead of crashing deep inside your prediction code.
    """
    gender: str
    SeniorCitizen: str 
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)  
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

def tenure_bucket(months):
    """Same function from Step 4 — MUST match exactly, or predictions will be wrong."""
    if months <= 12:
        return 'new'
    elif months <= 36:
        return 'mid'
    else:
        return 'long'

def prepare_input(customer: CustomerInput) -> pd.DataFrame:
    """Converts the validated Pydantic input into a DataFrame shaped exactly like X_train."""
    data = customer.dict()
    data['tenure_group'] = tenure_bucket(data['tenure'])
    
    
    data['SeniorCitizen'] = 1 if data['SeniorCitizen'] == 'Yes' else 0
    
    return pd.DataFrame([data])

@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        df_input = prepare_input(customer)
        X_processed = preprocessor.transform(df_input)  
        
        probability = model.predict_proba(X_processed)[0, 1]
        prediction = "Will Churn" if probability >= OPTIMAL_THRESHOLD else "Will Not Churn"
        
        return {
            "prediction": prediction,
            "churn_probability": round(float(probability), 4),
            "threshold_used": OPTIMAL_THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def get_top_reasons(shap_values_row, feature_names, top_n=3):
    """Same function from Step 9."""
    shap_series = pd.Series(shap_values_row, index=feature_names)
    churn_pushing = shap_series[shap_series > 0].sort_values(ascending=False)
    return churn_pushing.head(top_n)

@app.post("/explain")
def explain(customer: CustomerInput):
    try:
        df_input = prepare_input(customer)
        X_processed = preprocessor.transform(df_input)
        
        shap_values = explainer.shap_values(X_processed)
        top_reasons = get_top_reasons(shap_values[0], feature_names, top_n=3)
        
        return {
            "top_reasons": [
                {"feature": name, "impact": round(float(value), 4)}
                for name, value in top_reasons.items()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


RETENTION_ACTIONS = {
    "Contract_Month-to-month": "Offer a discount for upgrading to a 1-year or 2-year contract",
    "tenure_group_new": "Enroll in an early-tenure engagement/check-in program",
    "PaymentMethod_Electronic check": "Encourage switching to automatic payment with a small incentive",
    "InternetService_Fiber optic": "Investigate fiber service satisfaction; consider a loyalty discount",
}

@app.post("/retention-strategy")
def retention_strategy(customer: CustomerInput):
    try:
        df_input = prepare_input(customer)
        X_processed = preprocessor.transform(df_input)
        
        shap_values = explainer.shap_values(X_processed)
        top_reasons = get_top_reasons(shap_values[0], feature_names, top_n=1)
        
        if len(top_reasons) == 0:
            return {"recommendation": "No significant churn risk factors identified."}
        
        top_feature = top_reasons.index[0]
        recommendation = RETENTION_ACTIONS.get(
            top_feature, 
            "Review this customer's account for general retention outreach."
        )
        
        return {"top_driver": top_feature, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model": model_config['model_name']}