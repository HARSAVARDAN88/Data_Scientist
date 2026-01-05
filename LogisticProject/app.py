import streamlit as st
from model import predict_health

st.set_page_config(page_title="Health Risk Predictor")

st.title("📱 Mobile Health Risk Prediction")

screen = st.slider("Screen Time (hours/day)", 0, 12, 5)
sleep = st.slider("Sleep Hours", 0, 10, 6)
eye = st.radio("Eye Strain", ["No", "Yes"])

eye_val = 1 if eye == "Yes" else 0

if st.button("Predict"):
    result = predict_health(screen, sleep, eye_val)
    if result == 1:
        st.error("⚠ Health Risk Detected")
    else:
        st.success("✅ No Health Risk")
