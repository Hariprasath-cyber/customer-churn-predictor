# Customer Churn Predictor

A deployable ML web app that predicts telecom customer churn, explains *why* using SHAP, and recommends a specific retention action — built to answer a business question, not just produce an accuracy score.

**[Live Demo](https://customer-churn-predictor-mmft6pg5gkwye6pvh3icz4.streamlit.app/)** · **[Business Report](./business_report.md)**

> Note: the backend runs on Render's free tier, which sleeps after inactivity — the first prediction after a period of no use may take 30–50 seconds to respond while it wakes up.

---

## What this project does

1. A user enters a customer's details (contract type, tenure, monthly charges, etc.) through a form
2. The trained model predicts **whether they'll churn**, with a confidence percentage
3. **SHAP** explains the top 3 reasons driving *that specific customer's* risk — not just a generic feature-importance chart
4. The app recommends a **concrete retention action** based on the customer's top risk driver (e.g., "offer a contract upgrade discount")

## Why this isn't just another churn-predictor tutorial clone

Most churn-prediction portfolio projects stop at "trained a model, got 80% accuracy." This one is built around a different question: **does this model actually save the business money?**

- **Cost-based model selection, not accuracy-based.** A missed churner (false negative) costs far more than an unnecessary retention offer (false positive). Every model was evaluated on estimated business cost using a cost-benefit matrix, and the decision threshold was optimized to minimize that cost — not defaulted to 0.5.
- **PR-AUC and Macro F1 over raw accuracy**, since the ~26.5% churn rate makes accuracy a misleading metric on its own.
- **Leakage-safe preprocessing** — a `scikit-learn` `Pipeline` + `ColumnTransformer` fit only on training data, avoiding a common and easy-to-miss data leakage bug.
- **Decoupled architecture** — the model is served through a separate FastAPI backend (with Pydantic-validated endpoints), not loaded directly inside the Streamlit app. This mirrors how production ML systems separate inference from UI.
- **Per-customer explainability** via SHAP, not just a single global feature-importance chart.

## Architecture

```
Streamlit (frontend, Streamlit Community Cloud)
        │
        ▼  HTTP requests
FastAPI backend (Render)
  ├── /predict            → churn probability + prediction
  ├── /explain             → top 3 SHAP-driven reasons
  ├── /retention-strategy  → business recommendation
  └── /health              → status check
```

## Tech stack

| Layer | Tools |
|---|---|
| Data & modeling | pandas, numpy, scikit-learn, xgboost |
| Explainability | shap (LinearExplainer) |
| Backend API | FastAPI, Pydantic, uvicorn |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud (frontend), Render (backend) |

## Project structure

```
customer-churn-predictor/
├── data/                  # raw and processed datasets
├── notebooks/
│   ├── 01_eda.ipynb                  # exploration, cleaning
│   ├── 02_feature_engineering.ipynb  # leakage-safe preprocessing pipeline
│   └── 03_model_training.ipynb       # 3-model comparison, cost-optimized threshold
├── models/                # saved preprocessor, models, SHAP explainer
├── backend/
│   └── main.py            # FastAPI service
├── app.py                 # Streamlit frontend
├── requirements.txt
└── business_report.md     # findings and recommendations, non-technical summary
```

## Key results

- **Churn rate:** ~26.5% of customers
- **Selected model:** Logistic Regression — chosen for lowest estimated business cost among Logistic Regression, Random Forest, and tuned XGBoost, at an optimized decision threshold (not the default 0.5)
- **Top churn drivers:** contract type (month-to-month), low tenure, electronic check payment method

See [`business_report.md`](./business_report.md) for the full findings and business recommendations.

## Running locally

```bash
# clone and install
git clone https://github.com/Hariprasath-cyber/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt

# terminal 1 — start the backend
python -m uvicorn backend.main:app --reload --port 8000

# terminal 2 — start the frontend
python -m streamlit run app.py
```

## Known limitations & future improvements

- Trained on a static, historical dataset (originally an IBM sample dataset) — a production version would need periodic retraining on live data.
- One test case surfaced a counterintuitive SHAP result worth further investigation — a reminder that one-hot-encoded linear model coefficients need careful interpretation, not blind trust.
- Planned next steps: customer segmentation into churn-risk personas, and an interactive "what-if" retention simulator in the app.

## Related project

**[Smart College Placement Predictor](https://github.com/Hariprasath-cyber/placement-predictor)** — a separate portfolio project using XGBoost/Random Forest/SHAP with a Hugging Face Spaces deployment, exploring a different explainability and deployment architecture than this one.

---

**Author:** Arunjunai Hari Prasath — [LinkedIn](https://linkedin.com/in/arunjunaihariprasath) · [GitHub](https://github.com/Hariprasath-cyber)