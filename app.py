import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# load files
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_icon="🏦",
    layout="centered"
)
st.title("🏦 AI Loan Approval Prediction System")
st.markdown("Predict whether a loan application will be approved or rejected.")

st.write("Enter loan details")

# inputs
text = st.text_area("Purpose of Loan")

income = st.number_input("Income")

credit_score = st.number_input("Credit Score")

loan_amount = st.number_input("Loan Amount")

dti_ratio = st.number_input("DTI Ratio")

employment_status = st.selectbox(
    "Employment Status",
    ["employed", "unemployed"]
)

# encoding
if employment_status == "employed":
    employment_status = 1
else:
    employment_status = 0

# predict
if st.button("Predict"):

    # tfidf
    text_features = tfidf.transform([text])

    # numeric
    numeric_features = np.array([
        [
            income,
            credit_score,
            loan_amount,
            dti_ratio,
            employment_status
        ]
    ])

    # combine
    final_features = hstack([
        numeric_features,
        text_features
    ])

    # scaling
    final_features = scaler.transform(final_features)

    # prediction
    prediction = model.predict(final_features)

    # output
    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")