import streamlit as st
import requests

API_URL = "https://customer-churn-predictor-7xys.onrender.com"  

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer is likely to churn, understand why, and get a retention recommendation.")

st.divider()

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    monthly_charges = st.slider("Monthly Charges (₹)", min_value=0.0, max_value=150.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total Charges (₹)", min_value=0.0, value=800.0, step=10.0)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract_paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])


st.subheader("Additional Services")

col3, col4 = st.columns(2)

with col3:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col4:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])


st.divider()

if st.button("🔮 Predict Churn", type="primary"):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": contract_paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        with st.spinner("Analyzing customer..."):
            predict_response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            explain_response = requests.post(f"{API_URL}/explain", json=payload, timeout=10)
            strategy_response = requests.post(f"{API_URL}/retention-strategy", json=payload, timeout=10)

        if predict_response.status_code == 200:
            pred_data = predict_response.json()
            explain_data = explain_response.json()
            strategy_data = strategy_response.json()

            st.subheader("Prediction Result")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(label="Prediction", value=pred_data["prediction"])
            with col_b:
                st.metric(label="Churn Probability", value=f"{pred_data['churn_probability']*100:.1f}%")

            st.subheader("Top Reasons")
            for reason in explain_data["top_reasons"]:
                st.write(f"• **{reason['feature']}** (impact: {reason['impact']:.3f})")

            st.subheader("Recommended Action")
            st.success(strategy_data["primary_recommendation"])

            if len(strategy_data.get("all_relevant_actions", [])) > 1:
               with st.expander("See other contributing factors"):
                   for action in strategy_data["all_relevant_actions"][1:]:
                        st.write(f"• {action}")

        else:
            st.error(f"API error: {predict_response.json().get('detail', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to the backend. Make sure the FastAPI server is running (`uvicorn backend.main:app --reload --port 8000`).")
else:
    st.info("👆 Fill in the customer details above and click Predict.")

