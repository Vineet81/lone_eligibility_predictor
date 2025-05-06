import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and preprocessor
rf_model = joblib.load("rf_model.pkl")
log_model = joblib.load("logistic_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")

menu = ["Single Prediction", "Batch Prediction"]
choice = st.sidebar.selectbox("Select Mode", menu)

# Define input fields
def get_user_input():
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.selectbox("Loan Term (months)", [360, 120, 240, 180, 300, 84, 60, 36, 12])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Construct input DataFrame
    input_data = {
        "Gender": [Gender],
        "Married": [Married],
        "Dependents": [3 if Dependents == "3+" else int(Dependents)],
        "Education": [Education],
        "Self_Employed": [Self_Employed],
        "ApplicantIncome": [ApplicantIncome],
        "CoapplicantIncome": [CoapplicantIncome],
        "LoanAmount": [LoanAmount],
        "Loan_Amount_Term": [Loan_Amount_Term],
        "Credit_History": [Credit_History],
        "Property_Area": [Property_Area]
    }

    return pd.DataFrame(input_data)

if choice == "Single Prediction":
    st.subheader("🔍 Single Applicant Prediction")
    input_df = get_user_input()

    model_choice = st.radio("Choose Model", ["Random Forest", "Logistic Regression"])
    if st.button("Predict"):
        # Preprocess
        input_processed = preprocessor.transform(input_df)
        # Predict
        if model_choice == "Random Forest":
            pred = rf_model.predict(input_processed)[0]
        else:
            pred = log_model.predict(input_processed)[0]

        st.success(f"🎯 Prediction: {'Approved ✅' if pred == 'Y' else 'Rejected ❌'}")

elif choice == "Batch Prediction":
    st.subheader("📁 Batch Prediction")
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        batch_data = pd.read_csv(file)

        if "Loan_ID" in batch_data.columns:
            loan_ids = batch_data["Loan_ID"]
            batch_data.drop("Loan_ID", axis=1, inplace=True)
        else:
            loan_ids = None

        # Handle '3+'
        if "Dependents" in batch_data.columns:
            batch_data["Dependents"] = batch_data["Dependents"].replace("3+", 3).astype(int)

        # Predict with selected model
        model_choice = st.radio("Choose Model", ["Random Forest", "Logistic Regression"])
        if st.button("Batch Predict"):
            batch_processed = preprocessor.transform(batch_data)
            if model_choice == "Random Forest":
                preds = rf_model.predict(batch_processed)
            else:
                preds = log_model.predict(batch_processed)

            result_df = batch_data.copy()
            result_df["Prediction"] = preds

            if loan_ids is not None:
                result_df.insert(0, "Loan_ID", loan_ids)

            st.dataframe(result_df)

            # Download link
            csv = result_df.to_csv(index=False)
            st.download_button("Download Results", data=csv, file_name="batch_predictions.csv", mime="text/csv")

