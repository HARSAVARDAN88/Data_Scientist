import streamlit as st
from model import predict_loan

st.set_page_config(page_title="Bike Loan Approval")

st.title("🏍 Bike Loan Approval Prediction")

income = st.number_input("Monthly Income", 5000, 100000, 20000)
age = st.number_input("Age", 18, 60, 25)
credit = st.number_input("Credit Score", 300, 900, 650)

if st.button("Check Loan Status"):
    result = predict_loan(income, age, credit)
    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
