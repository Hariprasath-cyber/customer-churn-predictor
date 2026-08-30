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
    shap_series = pd.Series(shap_values_row, index=feature_names)
    churn_pushing = shap_series[shap_series > 0].sort_values(ascending=False)
    
    seen_categories = set()
    diversified = []
    for feat, val in churn_pushing.items():
        base_category = feat.split('_')[0]
        if base_category not in seen_categories:
            diversified.append((feat, val))
            seen_categories.add(base_category)
        if len(diversified) >= top_n:
            break
    
    return pd.Series(dict(diversified))

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
    "Contract_Month-to-month": "Offer a discount for upgrading to a 1-year or 2-year contract — this is the single strongest retention lever based on this analysis.",
    "tenure_group_new": "Enroll in an early-tenure engagement program: proactive check-in calls or onboarding support within the first 3 months.",
    "tenure_group_mid": "Monitor for renewal signals as the initial contract period nears its end; consider a loyalty perk before the 3-year mark.",
    "PaymentMethod_Electronic check": "Encourage switching to automatic payment (bank transfer or credit card) with a small one-time incentive or fee waiver.",
    "InternetService_Fiber optic": "Investigate fiber service satisfaction directly (call/survey); consider a service-quality follow-up or loyalty discount.",
    "OnlineSecurity_No": "Offer a free trial of online security add-on — customers without it show higher churn risk.",
    "TechSupport_No": "Offer a free trial of tech support add-on — lack of support access correlates with higher churn.",
    "PaperlessBilling_Yes": "No specific action — paperless billing is a minor factor; focus retention budget on contract/tenure drivers instead.",
    "SeniorCitizen": "Consider a simplified support channel or personal outreach — senior customers show a higher churn rate despite being a smaller segment.",
    "MonthlyCharges": "Review pricing tier fit — high monthly charges combined with short contracts is a common churn combination; consider bundling a discount with a contract upgrade rather than a price cut alone."
}

DEFAULT_RECOMMENDATION = "No single dominant risk factor identified — recommend a general retention check-in call to understand this customer's specific concerns."

@app.post("/retention-strategy")
def retention_strategy(customer: CustomerInput):
    try:
        df_input = prepare_input(customer)
        X_processed = preprocessor.transform(df_input)
        
        probability = model.predict_proba(X_processed)[0, 1]
        
        if probability < OPTIMAL_THRESHOLD:
            return {
                "top_drivers": [],
                "primary_recommendation": "This customer is not currently flagged as at-risk — no retention action needed.",
                "all_relevant_actions": []
            }
        
        shap_values = explainer.shap_values(X_processed)
        top_reasons = get_top_reasons(shap_values[0], feature_names, top_n=3)
        
        if len(top_reasons) == 0:
            return {
                "top_drivers": [],
                "primary_recommendation": "No significant churn risk factors identified for this customer.",
                "all_relevant_actions": []
            }
        
        recommendations = []
        top_drivers = []
        
        for feature_name in top_reasons.index:
            top_drivers.append(feature_name)
            action = RETENTION_ACTIONS.get(feature_name)
            if action is None:
                for key in RETENTION_ACTIONS:
                    if key in feature_name or feature_name in key:
                        action = RETENTION_ACTIONS[key]
                        break
            if action and action not in recommendations:
                recommendations.append(action)
        
        final_recommendation = recommendations[0] if recommendations else DEFAULT_RECOMMENDATION
        
        return {
            "top_drivers": top_drivers,
            "primary_recommendation": final_recommendation,
            "all_relevant_actions": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def health():
    return {"status": "ok", "model": model_config['model_name']}

